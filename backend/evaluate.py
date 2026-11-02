"""Benchmark K8Shield's Isolation Forest against a static rule-based baseline.

The baseline is the kind of threshold rule set a SOC writes by hand. An entity
window is flagged if ANY rule fires:

* request burst:         ``request_count > 100``
* auth failures:         ``error_code_rate >= 0.20``
* sensitive RBAC write:  ``rbac_escalation_flag == 1``
* interactive shell:     ``exec_ratio > 0`` (any ``pods/exec``, as in Falco's
  "Terminal shell in container")
* namespace sweep:       ``unique_namespaces_accessed > 10``

Both detectors are scored on the same held-out feature frame. The report gives:

* Precision, Recall, F1 and False Positive Rate (FPR) at each detector's
  default operating point.
* The relative FPR reduction, ``(FPR_rules - FPR_iforest) / FPR_rules``.
* A matched-recall comparison: the Isolation Forest's FPR when its threshold
  is set to reach the baseline's recall. This is the fairest single number,
  because it removes the effect of picking a ``contamination`` value.
* Recall per anomaly type.
* ``confusion_matrices.png`` and ``roc_curve.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render straight to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, roc_curve  # noqa: E402

from model_trainer import K8ShieldModel  # noqa: E402

logger = logging.getLogger(__name__)

# Chart styling: categorical slot 1 (blue) is K8Shield, slot 2 (orange) is the
# baseline. Text uses neutral ink tokens, never the series colour.
COLOR_IFOREST = "#2a78d6"
COLOR_RULES = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("k8s_blue", ["#eef4fc", "#9cc1ee", "#2a78d6", "#123f78"])


# --------------------------------------------------------------------------- #
# Baseline detector
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RuleThresholds:
    max_request_count: float = 100
    max_error_code_rate: float = 0.20
    flag_rbac_writes: bool = True
    flag_any_exec: bool = True
    max_unique_namespaces: float = 10


class RuleBasedDetector:
    """Static threshold rules over the same windowed features."""

    def __init__(self, thresholds: RuleThresholds | None = None):
        self.t = thresholds or RuleThresholds()

    def rule_hits(self, f: pd.DataFrame) -> pd.DataFrame:
        hits = {
            "request_burst": f["request_count"] > self.t.max_request_count,
            "auth_failures": f["error_code_rate"] >= self.t.max_error_code_rate,
            "namespace_sweep": f["unique_namespaces_accessed"] > self.t.max_unique_namespaces,
        }
        if self.t.flag_rbac_writes:
            hits["rbac_write"] = f["rbac_escalation_flag"] > 0
        if self.t.flag_any_exec and "exec_ratio" in f.columns:
            hits["exec_shell"] = f["exec_ratio"] > 0
        return pd.DataFrame(hits, index=f.index)

    def score(self, f: pd.DataFrame) -> np.ndarray:
        """Number of rules fired. A coarse score, used only to draw a ROC curve."""
        return self.rule_hits(f).sum(axis=1).to_numpy(dtype=float)

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        """True when the window is flagged."""
        return self.rule_hits(f).any(axis=1).to_numpy()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y_true: np.ndarray, y_flagged: np.ndarray) -> dict[str, float]:
    """Precision, recall, F1, FPR and confusion counts. Positive = anomaly."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_flagged).astype(bool)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[False, True]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "accuracy": float((tp + tn) / len(y)) if len(y) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def relative_reduction(baseline: float, new: float) -> float:
    return float((baseline - new) / baseline) if baseline > 0 else 0.0


def fpr_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> tuple[float, float]:
    """Lowest FPR at which ``scores`` reach ``target_recall``. Returns (fpr, threshold)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    ok = np.flatnonzero(tpr >= target_recall)
    i = ok[0] if ok.size else len(fpr) - 1
    return float(fpr[i]), float(thr[i])


@dataclass
class EvaluationResult:
    iforest: dict[str, float]
    rules: dict[str, float]
    iforest_auc: float
    rules_auc: float
    fpr_reduction: float
    matched_recall: dict[str, float]
    per_type_recall: dict[str, dict[str, float]]
    n_windows: int
    anomaly_rate: float
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_confusion_matrices(y: np.ndarray, preds: dict[str, np.ndarray], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(preds), figsize=(5.2 * len(preds), 4.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (name, p) in zip(axes, preds.items()):
        cm = confusion_matrix(y.astype(bool), p.astype(bool), labels=[False, True])
        row_share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(row_share, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
        for (i, j), count in np.ndenumerate(cm):
            dark = row_share[i, j] > 0.55
            ax.text(j, i, f"{count:,}\n{row_share[i, j]:.1%}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if dark else INK_PRIMARY)
        ax.set_xticks([0, 1], ["Normal", "Anomaly"])
        ax.set_yticks([0, 1], ["Normal", "Anomaly"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, color=INK_PRIMARY, fontsize=12, loc="left")
        ax.tick_params(colors=INK_SECONDARY, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices on the held-out test set (cell shade = share of actual row)",
                 color=INK_SECONDARY, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_roc(
    y: np.ndarray,
    if_scores: np.ndarray,
    rule_scores: np.ndarray,
    if_point: dict[str, float],
    rule_point: dict[str, float],
    path: Path,
) -> tuple[Path, float, float]:
    fpr_if, tpr_if, _ = roc_curve(y, if_scores)
    fpr_r, tpr_r, _ = roc_curve(y, rule_scores)
    auc_if, auc_r = auc(fpr_if, tpr_if), auc(fpr_r, tpr_r)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, xmax in ((ax_full, 1.0), (ax_zoom, 0.15)):
        _style_axes(ax)
        ax.grid(color=GRID, linewidth=0.6)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
        ax.plot(fpr_if, tpr_if, color=COLOR_IFOREST, linewidth=2,
                label=f"K8Shield Isolation Forest (AUC {auc_if:.3f})")
        ax.plot(fpr_r, tpr_r, color=COLOR_RULES, linewidth=2, drawstyle="steps-post",
                label=f"Static rules, by rules fired (AUC {auc_r:.3f})")
        ax.scatter([if_point["fpr"]], [if_point["recall"]], s=64, color=COLOR_IFOREST,
                   edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.scatter([rule_point["fpr"]], [rule_point["recall"]], s=64, color=COLOR_RULES, marker="s",
                   edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
    ax_zoom.annotate("K8Shield operating point", (if_point["fpr"], if_point["recall"]),
                     xytext=(8, -16), textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax_zoom.annotate("Rule baseline", (rule_point["fpr"], rule_point["recall"]),
                     xytext=(-10, -18), textcoords="offset points", color=INK_SECONDARY, fontsize=9, ha="right")
    ax_full.set_title("ROC curve", color=INK_PRIMARY, loc="left", fontsize=12)
    ax_zoom.set_title("Low-FPR region (FPR ≤ 0.15)", color=INK_PRIMARY, loc="left", fontsize=12)
    ax_full.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path, float(auc_if), float(auc_r)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(
    model: K8ShieldModel,
    test_features: pd.DataFrame,
    out_dir: str | Path = "reports",
    rules: RuleBasedDetector | None = None,
    make_plots: bool = True,
) -> EvaluationResult:
    """Score ``test_features`` with both detectors and write metrics and plots to ``out_dir``."""
    if "is_anomaly" not in test_features.columns:
        raise ValueError("Evaluation needs ground-truth labels (an 'is_anomaly' column)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rules = rules or RuleBasedDetector()

    y = test_features["is_anomaly"].to_numpy().astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Test set must contain both normal and anomalous windows")

    if_scores = model.anomaly_scores(test_features)
    if_flags = model.predict(test_features) == -1
    rule_scores = rules.score(test_features)
    rule_flags = rules.predict(test_features)

    m_if = binary_metrics(y, if_flags)
    m_rules = binary_metrics(y, rule_flags)
    matched_fpr, matched_thr = fpr_at_recall(y, if_scores, m_rules["recall"])

    per_type: dict[str, dict[str, float]] = {}
    types = test_features.get("anomaly_type", pd.Series("anomaly", index=test_features.index))
    for t in sorted(set(types[y == 1])):
        mask = (types == t).to_numpy() & (y == 1)
        per_type[t] = {
            "n": int(mask.sum()),
            "iforest_recall": float(if_flags[mask].mean()),
            "rules_recall": float(rule_flags[mask].mean()),
        }

    artifacts: dict[str, str] = {}
    auc_if = auc_r = float("nan")
    if make_plots:
        cm_path = plot_confusion_matrices(
            y, {"K8Shield (Isolation Forest)": if_flags, "Static rule baseline": rule_flags},
            out / "confusion_matrices.png")
        roc_path, auc_if, auc_r = plot_roc(y, if_scores, rule_scores, m_if, m_rules, out / "roc_curve.png")
        artifacts = {"confusion_matrices": str(cm_path), "roc_curve": str(roc_path)}
    else:
        fpr_a, tpr_a, _ = roc_curve(y, if_scores)
        fpr_b, tpr_b, _ = roc_curve(y, rule_scores)
        auc_if, auc_r = float(auc(fpr_a, tpr_a)), float(auc(fpr_b, tpr_b))

    result = EvaluationResult(
        iforest=m_if,
        rules=m_rules,
        iforest_auc=auc_if,
        rules_auc=auc_r,
        fpr_reduction=relative_reduction(m_rules["fpr"], m_if["fpr"]),
        matched_recall={
            "target_recall": m_rules["recall"],
            "iforest_fpr": matched_fpr,
            "iforest_score_threshold": matched_thr,
            "rules_fpr": m_rules["fpr"],
            "fpr_reduction": relative_reduction(m_rules["fpr"], matched_fpr),
        },
        per_type_recall=per_type,
        n_windows=int(len(y)),
        anomaly_rate=float(y.mean()),
        artifacts=artifacts,
    )
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(result.to_dict(), indent=2))
    report_path = out / "benchmark_report.md"
    report_path.write_text(format_report(result))
    result.artifacts.update({"metrics": str(metrics_path), "report": str(report_path)})
    return result


def format_report(r: EvaluationResult) -> str:
    """Render a Markdown benchmark table (also printed to the console)."""
    def row(name: str, m: dict[str, float], auc_v: float) -> str:
        return (f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                f"{m['fpr']:.4f} | {auc_v:.3f} | {m['fp']:,} |")

    lines = [
        f"Test set: {r.n_windows:,} entity windows, {r.anomaly_rate:.2%} anomalous",
        "",
        "| Detector | Precision | Recall | F1 | FPR | ROC AUC | False positives |",
        "|---|---:|---:|---:|---:|---:|---:|",
        row("Static rule baseline", r.rules, r.rules_auc),
        row("K8Shield (Isolation Forest)", r.iforest, r.iforest_auc),
        "",
        f"FPR reduction at default operating points: {r.fpr_reduction:.1%}",
        f"FPR reduction at matched recall ({r.matched_recall['target_recall']:.3f}): "
        f"{r.matched_recall['fpr_reduction']:.1%} "
        f"(Isolation Forest FPR {r.matched_recall['iforest_fpr']:.4f} vs rules {r.matched_recall['rules_fpr']:.4f})",
        "",
        "| Anomaly type | Windows | Isolation Forest recall | Rules recall |",
        "|---|---:|---:|---:|",
    ]
    for t, v in r.per_type_recall.items():
        lines.append(f"| {t} | {v['n']} | {v['iforest_recall']:.3f} | {v['rules_recall']:.3f} |")
    return "\n".join(lines) + "\n"


def summarize_runs(results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of the headline metrics over repeated runs."""
    def stat(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}

    keys = ("precision", "recall", "f1", "fpr")
    summary: dict[str, dict[str, float]] = {}
    for k in keys:
        summary[f"iforest_{k}"] = stat([r.iforest[k] for r in results])
        summary[f"rules_{k}"] = stat([r.rules[k] for r in results])
    summary["iforest_auc"] = stat([r.iforest_auc for r in results])
    summary["fpr_reduction"] = stat([r.fpr_reduction for r in results])
    summary["matched_recall_fpr_reduction"] = stat([r.matched_recall["fpr_reduction"] for r in results])
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained K8Shield model on a labelled audit log.")
    parser.add_argument("audit_log", help="Labelled test audit log (JSON Lines)")
    parser.add_argument("--model", default="models", help="Model bundle or directory")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from feature_extractor import FeatureExtractor, load_events

    model = K8ShieldModel.load(args.model)
    extractor = model.extractor or FeatureExtractor()
    if model.extractor is None:
        logger.warning("Model has no saved extractor state; fitting UA table on the test log")
        extractor.fit(load_events(args.audit_log))
    feats = extractor.transform(load_events(args.audit_log))
    print(format_report(evaluate(model, feats, args.out_dir)))


if __name__ == "__main__":
    _cli()
