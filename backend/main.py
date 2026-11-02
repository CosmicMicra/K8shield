"""K8Shield command-line entry point.

    python main.py run                    # full pipeline: generate -> features -> train -> evaluate -> ingest
    python main.py run --runs 5           # repeat with 5 independent seed pairs, report mean ± std
    python main.py generate --seed 7 -o data/raw/custom.jsonl
    python main.py train data/raw/train_audit.jsonl
    python main.py evaluate data/raw/test_audit.jsonl
    python main.py ingest data/raw/test_audit.jsonl --es-url http://localhost:9200

Run ``python main.py <command> -h`` for the options of each command.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from es_pipeline import DEFAULT_INDEX, AlertManager, make_sink, stream_to_index
from evaluate import EvaluationResult, evaluate, format_report, summarize_runs
from feature_extractor import FeatureConfig, FeatureExtractor, load_events
from log_generator import GeneratorConfig, generate_logs
from model_trainer import K8ShieldModel, ModelConfig, train_model

logger = logging.getLogger("k8shield")


@dataclass(frozen=True, slots=True)
class Paths:
    data: Path = Path("data")
    models: Path = Path("models")
    reports: Path = Path("reports")

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"


def _contamination(value: str) -> float | str:
    if value == "auto":
        return value
    try:
        v = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("contamination must be a float or 'auto'") from exc
    if not 0 < v <= 0.5:
        raise argparse.ArgumentTypeError("contamination must be in (0, 0.5]")
    return v


def _ratio(value: str) -> float:
    v = float(value)
    if not 0 <= v < 1:
        raise argparse.ArgumentTypeError("ratio must be in [0, 1)")
    return v


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_generate(args: argparse.Namespace) -> int:
    stats = generate_logs(args.output, GeneratorConfig(
        duration_minutes=args.minutes, anomaly_ratio=args.anomaly_ratio,
        stealth_ratio=args.stealth_ratio, seed=args.seed))
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    extractor = FeatureExtractor(FeatureConfig(args.window, args.stride, not args.core_only))
    features = extractor.fit_transform(load_events(args.audit_log))
    paths = Paths(models=Path(args.model_dir))
    paths.processed.mkdir(parents=True, exist_ok=True)
    features.to_csv(paths.processed / "train_features.csv", index=False)
    model = train_model(features, extractor,
                        ModelConfig(n_estimators=args.n_estimators, contamination=args.contamination,
                                    random_state=args.seed),
                        paths.models)
    print(json.dumps(model.metadata, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = K8ShieldModel.load(args.model)
    if model.extractor is None:
        logger.error("Model bundle has no feature-extractor state; retrain it with `python main.py train`")
        return 1
    features = model.extractor.transform(load_events(args.audit_log))
    result = evaluate(model, features, args.out_dir)
    print(format_report(result))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    model = K8ShieldModel.load(args.model)
    if model.extractor is None:
        logger.error("Model bundle has no feature-extractor state; retrain it with `python main.py train`")
        return 1
    scored = model.score(model.extractor.transform(load_events(args.audit_log)))
    sink = make_sink(args.es_url, args.index, args.mock_dir, username=args.es_user,
                     password=args.es_password, api_key=args.es_api_key)
    stats = stream_to_index(scored, model.feature_names, sink,
                            AlertManager(alerts_path=Path(args.mock_dir) / "alerts.jsonl",
                                         webhook_url=args.webhook_url, log_alerts=args.verbose_alerts),
                            model.metadata.get("model_version", "unknown"), replay_delay_s=args.replay_delay)
    print(json.dumps(stats, indent=2))
    return 0


def _run_once(args: argparse.Namespace, paths: Paths, train_seed: int, test_seed: int,
              out_dir: Path, ingest: bool, keep_logs: bool = True) -> EvaluationResult:
    """One full generate -> train -> evaluate (-> ingest) cycle on a fresh seed pair.

    ``keep_logs=False`` (repeat runs) keeps only the metrics: no logs, feature CSVs or model are left behind.
    """
    train_log = paths.raw / f"train_audit_seed{train_seed}.jsonl"
    test_log = paths.raw / f"test_audit_seed{test_seed}.jsonl"
    common = dict(duration_minutes=args.minutes, anomaly_ratio=args.anomaly_ratio, stealth_ratio=args.stealth_ratio)

    logger.info("[1/5] Generating audit logs (train seed %d, test seed %d)", train_seed, test_seed)
    generate_logs(train_log, GeneratorConfig(seed=train_seed, **common))
    generate_logs(test_log, GeneratorConfig(seed=test_seed, **common))

    logger.info("[2/5] Extracting %d-minute window features", args.window)
    extractor = FeatureExtractor(FeatureConfig(args.window, args.stride, not args.core_only))
    train_feats = extractor.fit_transform(load_events(train_log))
    test_feats = extractor.transform(load_events(test_log))  # UA table fitted on train only
    if keep_logs:  # repeat runs must not overwrite the primary run's artifacts
        paths.processed.mkdir(parents=True, exist_ok=True)
        train_feats.to_csv(paths.processed / "train_features.csv", index=False)
        test_feats.to_csv(paths.processed / "test_features.csv", index=False)

    logger.info("[3/5] Training Isolation Forest (unsupervised; labels unused)")
    model = train_model(train_feats, extractor,
                        ModelConfig(n_estimators=args.n_estimators, contamination=args.contamination,
                                    random_state=train_seed),
                        paths.models if keep_logs else None)

    logger.info("[4/5] Evaluating against the static rule baseline")
    result = evaluate(model, test_feats, out_dir, make_plots=True)

    if not keep_logs:  # repeat runs only need the metrics; raw logs are ~150 MB each
        train_log.unlink(missing_ok=True)
        test_log.unlink(missing_ok=True)

    if ingest:
        logger.info("[5/5] Streaming scored test windows to %s", "Elasticsearch" if args.es_url else "the mock index")
        scored = model.score(test_feats)
        mock_dir = paths.data / "es_mock"
        stream_to_index(scored, model.feature_names, make_sink(args.es_url, args.index, mock_dir),
                        AlertManager(alerts_path=mock_dir / "alerts.jsonl"),
                        model.metadata.get("model_version", "unknown"))
    return result


def cmd_run(args: argparse.Namespace) -> int:
    paths = Paths(Path(args.data_dir), Path(args.model_dir), Path(args.out_dir))
    t0 = time.perf_counter()
    results: list[EvaluationResult] = []
    for i in range(args.runs):
        train_seed, test_seed = args.seed + 2 * i, args.seed + 2 * i + 1
        out_dir = paths.reports if i == 0 else paths.reports / f"run_{i + 1}"
        results.append(_run_once(args, paths, train_seed, test_seed, out_dir, ingest=(i == 0), keep_logs=(i == 0)))

    print("\n=== K8Shield benchmark (run 1) ===\n")
    print(format_report(results[0]))
    if args.runs > 1:
        summary = summarize_runs(results)
        (paths.reports / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"=== Mean ± std over {args.runs} independent runs ===")
        for k, v in summary.items():
            print(f"  {k:<32} {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"\nArtifacts: {paths.models}/k8shield_model.joblib, {paths.reports}/metrics.json, "
          f"{paths.reports}/confusion_matrices.png, {paths.reports}/roc_curve.png")
    print(f"Done in {time.perf_counter() - t0:.1f}s")
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="k8shield", description="K8Shield: Kubernetes audit-log anomaly detection.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def gen_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--minutes", type=int, default=24 * 60, help="Simulated duration per log (default: 1 day)")
        sp.add_argument("--anomaly-ratio", type=_ratio, default=0.05, help="Anomalous entity-window share")
        sp.add_argument("--stealth-ratio", type=float, default=0.3, help="Share of attacks that blend in")

    def feat_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--window", type=int, default=5, help="Window length in minutes")
        sp.add_argument("--stride", type=int, default=None, help="Stride in minutes (sliding windows)")
        sp.add_argument("--core-only", action="store_true", help="Use only the six core features")

    def model_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--contamination", type=_contamination, default=0.05)
        sp.add_argument("--n-estimators", type=int, default=300)

    g = sub.add_parser("generate", help="Generate a synthetic audit log")
    g.add_argument("-o", "--output", default="data/raw/audit.jsonl")
    g.add_argument("--seed", type=int, default=42)
    gen_opts(g)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="Extract features and train the model")
    t.add_argument("audit_log")
    t.add_argument("--model-dir", default="models")
    t.add_argument("--seed", type=int, default=42)
    feat_opts(t)
    model_opts(t)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="Benchmark a trained model on a labelled log")
    e.add_argument("audit_log")
    e.add_argument("--model", default="models")
    e.add_argument("--out-dir", default="reports")
    e.set_defaults(func=cmd_evaluate)

    i = sub.add_parser("ingest", help="Score a log and stream results to Elasticsearch or the mock index")
    i.add_argument("audit_log")
    i.add_argument("--model", default="models")
    i.add_argument("--es-url", default=None)
    i.add_argument("--es-user", default=None)
    i.add_argument("--es-password", default=None)
    i.add_argument("--es-api-key", default=None)
    i.add_argument("--index", default=DEFAULT_INDEX)
    i.add_argument("--mock-dir", default="data/es_mock")
    i.add_argument("--webhook-url", default=None)
    i.add_argument("--replay-delay", type=float, default=0.0, help="Seconds between batches (demo streaming)")
    i.add_argument("--verbose-alerts", action="store_true", help="Log every alert to the console")
    i.set_defaults(func=cmd_ingest)

    r = sub.add_parser("run", help="Run the whole pipeline end to end")
    r.add_argument("--seed", type=int, default=42, help="Train seed; test seed is seed+1")
    r.add_argument("--runs", type=int, default=1, help="Repeat with independent seeds and report mean ± std")
    r.add_argument("--data-dir", default="data")
    r.add_argument("--model-dir", default="models")
    r.add_argument("--out-dir", default="reports")
    r.add_argument("--es-url", default=None, help="Elasticsearch URL (omit to use the mock index)")
    r.add_argument("--index", default=DEFAULT_INDEX)
    gen_opts(r)
    feat_opts(r)
    model_opts(r)
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
