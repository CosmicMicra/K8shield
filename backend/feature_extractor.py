"""Turn raw Kubernetes audit events into windowed behavioural features.

Events are grouped by entity, a (``source_ip``, ``user``) pair, and by time
window. The default is a 5-minute tumbling window. Set ``stride_minutes``
lower than ``window_minutes`` to get overlapping sliding windows.

Feature set
-----------
Core features, as specified for K8Shield:

``request_count``
    Total API requests the entity made in the window.
``verb_entropy``
    Shannon entropy (bits) of the HTTP methods behind the audit verbs:
    get/list/watch -> GET, create -> POST, update -> PUT, patch -> PATCH,
    delete/deletecollection -> DELETE.
``rbac_escalation_flag``
    1 if the window has a sensitive RBAC call: a write to a
    (Cluster)Role/(Cluster)RoleBinding, or any request that references
    ``cluster-admin``.
``unique_namespaces_accessed``
    Number of distinct namespaces the entity targeted.
``unusual_user_agent_score``
    Rarity of the client user agent, from 0 (ubiquitous) to 1 (never seen
    while fitting). Rarity is ``log(1/p) / log(N)``, where ``p`` is the share
    of training entity windows that used the agent and ``N`` is the number of
    training entity windows. Frequencies are counted per entity window, not
    per request, so a single burst cannot make its own client look "common".
``error_code_rate``
    Share of responses with HTTP 401 or 403.

Extended context features, which can be turned off with
``FeatureConfig(include_extended=False)``:

``exec_ratio``
    Share of requests to ``pods/exec``, ``pods/attach`` or ``pods/portforward``.
``secret_access_ratio``
    Share of requests that touch ``secrets``.
``external_source_flag``
    1 if the source IP is not private, loopback or link-local.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from log_generator import GROUND_TRUTH_ANNOTATION

logger = logging.getLogger(__name__)

CORE_FEATURES: tuple[str, ...] = (
    "request_count",
    "verb_entropy",
    "rbac_escalation_flag",
    "unique_namespaces_accessed",
    "unusual_user_agent_score",
    "error_code_rate",
)
EXTENDED_FEATURES: tuple[str, ...] = (
    "exec_ratio",
    "secret_access_ratio",
    "external_source_flag",
)
ENTITY_KEYS: tuple[str, ...] = ("window_start", "source_ip", "user")

VERB_TO_HTTP: dict[str, str] = {
    "get": "GET",
    "list": "GET",
    "watch": "GET",
    "create": "POST",
    "update": "PUT",
    "patch": "PATCH",
    "delete": "DELETE",
    "deletecollection": "DELETE",
}
RBAC_RESOURCES = frozenset({"roles", "rolebindings", "clusterroles", "clusterrolebindings"})
RBAC_WRITE_VERBS = frozenset({"create", "update", "patch", "delete", "deletecollection"})
INTERACTIVE_SUBRESOURCES = frozenset({"exec", "attach", "portforward"})
AUTH_ERROR_CODES = frozenset({401, 403})


class FeatureExtractionError(RuntimeError):
    """Raised when audit input cannot be parsed into features."""


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    window_minutes: int = 5
    stride_minutes: int | None = None  # None means tumbling (stride == window)
    include_extended: bool = True

    def __post_init__(self) -> None:
        stride = self.stride_minutes or self.window_minutes
        if self.window_minutes <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        if self.window_minutes % stride:
            raise ValueError("window_minutes must be a multiple of stride_minutes")

    @property
    def stride(self) -> int:
        return self.stride_minutes or self.window_minutes

    @property
    def feature_names(self) -> list[str]:
        return list(CORE_FEATURES + (EXTENDED_FEATURES if self.include_extended else ()))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def iter_audit_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield audit events from a JSON Lines file or a JSON ``EventList``.

    Malformed lines are skipped with a warning, so one corrupt record does not
    stop a long replay.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audit log not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[" or (first == "{" and _looks_like_event_list(path)):
            payload = json.load(fh)
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            yield from (e for e in items if isinstance(e, dict))
            return
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d skipped malformed JSON (%s)", path, lineno, exc)
                continue
            if isinstance(event, dict):
                yield event


def _looks_like_event_list(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as fh:
        head = fh.read(4096)
    return '"EventList"' in head.split("\n", 1)[0]


@lru_cache(maxsize=65536)
def _is_external_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local)


def _references_cluster_admin(event: dict[str, Any]) -> bool:
    req = event.get("requestObject") or {}
    role_ref = (req.get("roleRef") or {}) if isinstance(req, dict) else {}
    if role_ref.get("name") == "cluster-admin":
        return True
    name = (event.get("objectRef") or {}).get("name") or ""
    return "cluster-admin" in name


def events_to_frame(events: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten audit events into one row per request with event-level flags."""
    rows: list[tuple[Any, ...]] = []
    skipped = 0
    for ev in events:
        try:
            ts = ev.get("requestReceivedTimestamp") or ev["stageTimestamp"]
            obj = ev.get("objectRef") or {}
            verb = str(ev.get("verb", "")).lower()
            resource = obj.get("resource") or ""
            sub = obj.get("subresource")
            code = int((ev.get("responseStatus") or {}).get("code", 0))
            ips = ev.get("sourceIPs") or ["unknown"]
            annotations = ev.get("annotations") or {}
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        rbac_flag = (resource in RBAC_RESOURCES and verb in RBAC_WRITE_VERBS) or _references_cluster_admin(ev)
        rows.append(
            (
                ts,
                ips[0],
                (ev.get("user") or {}).get("username", "unknown"),
                verb,
                VERB_TO_HTTP.get(verb, "OTHER"),
                resource,
                obj.get("namespace"),
                code,
                ev.get("userAgent") or "unknown",
                rbac_flag,
                sub in INTERACTIVE_SUBRESOURCES,
                resource == "secrets",
                code in AUTH_ERROR_CODES,
                annotations.get(GROUND_TRUTH_ANNOTATION),
            )
        )
    if skipped:
        logger.warning("Skipped %d events with missing or invalid fields", skipped)
    if not rows:
        raise FeatureExtractionError("No valid audit events found")

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp", "source_ip", "user", "verb", "http_verb", "resource", "namespace",
            "code", "user_agent", "is_rbac", "is_exec", "is_secret", "is_auth_error", "anomaly_type",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df


def load_events(path: str | Path) -> pd.DataFrame:
    """Read an audit log file and return the flattened per-request frame."""
    return events_to_frame(iter_audit_events(path))


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """Fit/transform extractor for windowed audit features.

    ``fit`` learns the user-agent frequency table, the only learned state.
    ``transform`` turns any event frame into one row per entity window. Fit on
    training data only, so rarity scores on test data are not leaked.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.ua_frequency_: dict[str, float] | None = None
        self.n_fit_groups_: int = 0

    # ---- windowing -------------------------------------------------------- #
    def _windowed(self, events: pd.DataFrame) -> pd.DataFrame:
        """Assign each event to every window that contains it."""
        cfg = self.config
        stride_ns = cfg.stride * 60 * 1_000_000_000
        k = cfg.window_minutes // cfg.stride
        ts_ns = events["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
        base = (ts_ns // stride_ns) * stride_ns
        if k == 1:
            out = events.copy()
            out["window_start"] = base
        else:
            starts = (base[:, None] - np.arange(k, dtype="int64")[None, :] * stride_ns).ravel()
            out = events.iloc[np.repeat(np.arange(len(events)), k)].reset_index(drop=True)
            out["window_start"] = starts
        out["window_start"] = pd.to_datetime(out["window_start"], utc=True)
        return out

    # ---- fitting ---------------------------------------------------------- #
    def fit(self, events: pd.DataFrame) -> FeatureExtractor:
        windowed = self._windowed(events)
        groups = windowed[[*ENTITY_KEYS, "user_agent"]].drop_duplicates()
        n_groups = int(windowed[list(ENTITY_KEYS)].drop_duplicates().shape[0])
        counts = groups["user_agent"].value_counts()
        self.ua_frequency_ = (counts / n_groups).to_dict()
        self.n_fit_groups_ = n_groups
        logger.info("Fitted UA frequency table: %d agents over %d entity windows", len(counts), n_groups)
        return self

    def _ua_rarity(self, agents: pd.Series) -> pd.Series:
        if self.ua_frequency_ is None:
            raise FeatureExtractionError("FeatureExtractor is not fitted; call fit() first")
        denom = np.log(max(self.n_fit_groups_, 2))
        freq = agents.map(self.ua_frequency_).astype(float)
        rarity = np.log(1.0 / freq) / denom
        return rarity.fillna(1.0).clip(0.0, 1.0)

    # ---- transform -------------------------------------------------------- #
    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Return one row per (window_start, source_ip, user) with features and labels."""
        if events.empty:
            raise FeatureExtractionError("Cannot extract features from an empty event frame")
        df = self._windowed(events)
        df["ua_rarity"] = self._ua_rarity(df["user_agent"])
        keys = list(ENTITY_KEYS)
        g = df.groupby(keys, sort=True, observed=True)

        feats = g.agg(
            request_count=("verb", "size"),
            rbac_escalation_flag=("is_rbac", "max"),
            unique_namespaces_accessed=("namespace", "nunique"),
            unusual_user_agent_score=("ua_rarity", "max"),
            error_code_rate=("is_auth_error", "mean"),
            exec_ratio=("is_exec", "mean"),
            secret_access_ratio=("is_secret", "mean"),
        )
        feats["verb_entropy"] = self._verb_entropy(df, keys).reindex(feats.index).fillna(0.0)
        feats["rbac_escalation_flag"] = feats["rbac_escalation_flag"].astype(int)
        feats = feats.reset_index()
        feats["external_source_flag"] = feats["source_ip"].map(_is_external_ip).astype(int)
        feats["window_end"] = feats["window_start"] + pd.Timedelta(minutes=self.config.window_minutes)

        # Ground truth (only present for synthetic or labelled data).
        labelled = df["anomaly_type"].notna()
        if labelled.any():
            types = df.loc[labelled].groupby(keys, observed=True)["anomaly_type"].first()
            feats = feats.merge(types.rename("anomaly_type").reset_index(), on=keys, how="left")
            feats["is_anomaly"] = feats["anomaly_type"].notna().astype(int)
            feats["anomaly_type"] = feats["anomaly_type"].fillna("normal")

        meta = ["window_start", "window_end", "source_ip", "user"]
        label_cols = [c for c in ("is_anomaly", "anomaly_type") if c in feats.columns]
        ordered = feats[meta + self.config.feature_names + label_cols]
        return ordered.astype({c: float for c in self.config.feature_names}).reset_index(drop=True)

    def fit_transform(self, events: pd.DataFrame) -> pd.DataFrame:
        return self.fit(events).transform(events)

    @staticmethod
    def _verb_entropy(df: pd.DataFrame, keys: list[str]) -> pd.Series:
        counts = df.groupby([*keys, "http_verb"], observed=True).size().unstack(fill_value=0)
        p = counts.to_numpy(dtype=float, copy=True)
        p /= p.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
        return pd.Series(ent + 0.0, index=counts.index)  # + 0.0 normalises -0.0

    # ---- persistence ------------------------------------------------------ #
    def to_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "ua_frequency": self.ua_frequency_,
            "n_fit_groups": self.n_fit_groups_,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FeatureExtractor:
        fx = cls(FeatureConfig(**state["config"]))
        fx.ua_frequency_ = state["ua_frequency"]
        fx.n_fit_groups_ = state["n_fit_groups"]
        return fx


def extract_features(path: str | Path, extractor: FeatureExtractor | None = None) -> pd.DataFrame:
    """Load ``path`` and return its feature frame (fitting a new extractor if none is given)."""
    events = load_events(path)
    if extractor is None:
        return FeatureExtractor().fit_transform(events)
    return extractor.transform(events)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Extract windowed features from K8s audit logs.")
    parser.add_argument("input", help="Audit log (JSON Lines or EventList JSON)")
    parser.add_argument("-o", "--output", default="data/processed/features.csv")
    parser.add_argument("--window", type=int, default=5, help="Window length in minutes")
    parser.add_argument("--stride", type=int, default=None, help="Stride in minutes (sliding windows)")
    parser.add_argument("--core-only", action="store_true", help="Use only the six core features")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fx = FeatureExtractor(FeatureConfig(args.window, args.stride, not args.core_only))
    feats = fx.fit_transform(load_events(args.input))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(args.output, index=False)
    print(f"Wrote {len(feats)} feature rows to {args.output}")


if __name__ == "__main__":
    _cli()
