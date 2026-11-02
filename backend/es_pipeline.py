"""Stream scored K8Shield windows into Elasticsearch, or a local mock index, and raise alerts.

Two interchangeable sinks are available.

``ElasticsearchSink``
    Talks to a real cluster through the REST ``_bulk`` API using only the
    standard library (``urllib``), so the official client is not needed. It
    creates the index with an explicit mapping, sends documents in chunks,
    retries transient failures with exponential backoff, and reports per-item
    bulk errors.
``LocalIndexSink``
    A file-backed mock that writes one NDJSON document per line. It supports
    simple ``search``, ``count`` and ``top_alerts`` queries, so the pipeline
    runs end-to-end on a laptop with no services at all.

Each document looks like this::

    {
      "@timestamp": "2026-01-06T13:05:00+00:00",   # window start
      "window_end": "2026-01-06T13:10:00+00:00",
      "source_ip": "185.220.101.23",
      "user": "system:serviceaccount:default:default",
      "anomaly_score": 0.71,
      "is_flagged": true,
      "severity": "high",
      "features": { "request_count": 412.0, ... },
      "model_version": "1.0.0",
      "event": {"kind": "alert", "module": "k8shield"}
    }

``AlertManager`` logs flagged documents. It can append them to
``alerts.jsonl`` and POST them to a webhook (Slack-compatible ``{"text": ...}``).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INDEX = "k8shield-anomalies"

INDEX_MAPPING: dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "@timestamp": {"type": "date"},
            "window_end": {"type": "date"},
            "source_ip": {"type": "ip"},
            "user": {"type": "keyword"},
            "anomaly_score": {"type": "float"},
            "is_flagged": {"type": "boolean"},
            "severity": {"type": "keyword"},
            "model_version": {"type": "keyword"},
            "features": {"type": "object", "dynamic": True},
            "event": {"properties": {"kind": {"type": "keyword"}, "module": {"type": "keyword"}}},
        }
    },
}


class SinkError(RuntimeError):
    """Raised when documents cannot be indexed."""


# --------------------------------------------------------------------------- #
# Document construction
# --------------------------------------------------------------------------- #
def severity_for(score: float, flagged: bool) -> str:
    if not flagged:
        return "none"
    if score >= 0.70:
        return "critical"
    if score >= 0.62:
        return "high"
    return "medium"


def build_documents(
    scored: pd.DataFrame,
    feature_names: list[str],
    model_version: str = "unknown",
) -> Iterator[dict[str, Any]]:
    """Turn a scored feature frame into index documents, in time order.

    Ground-truth columns (``is_anomaly``, ``anomaly_type``) are left out on
    purpose: a production index must not hold labels the detector never sees.
    """
    required = {"window_start", "source_ip", "user", "anomaly_score", "is_flagged"}
    missing = required - set(scored.columns)
    if missing:
        raise KeyError(f"Scored frame is missing columns: {sorted(missing)}")
    ordered = scored.sort_values(["window_start", "anomaly_score"], ascending=[True, False])
    for row in ordered.itertuples(index=False):
        rec = row._asdict()
        flagged = bool(rec["is_flagged"])
        score = float(rec["anomaly_score"])
        doc: dict[str, Any] = {
            "@timestamp": pd.Timestamp(rec["window_start"]).isoformat(),
            "window_end": pd.Timestamp(rec["window_end"]).isoformat() if "window_end" in rec else None,
            "source_ip": rec["source_ip"],
            "user": rec["user"],
            "anomaly_score": round(score, 6),
            "is_flagged": flagged,
            "severity": severity_for(score, flagged),
            "features": {f: float(rec[f]) for f in feature_names if f in rec},
            "model_version": model_version,
            "event": {"kind": "alert" if flagged else "metric", "module": "k8shield"},
        }
        if doc["window_end"] is None:
            del doc["window_end"]
        yield doc


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #
class DocumentSink(Protocol):
    def ensure_index(self) -> None: ...
    def index_batch(self, docs: list[dict[str, Any]]) -> int: ...
    def close(self) -> None: ...


@dataclass
class ElasticsearchSink:
    """Minimal Elasticsearch ``_bulk`` client built on ``urllib``."""

    url: str = "http://localhost:9200"
    index: str = DEFAULT_INDEX
    username: str | None = None
    password: str | None = None
    api_key: str | None = None
    verify_tls: bool = True
    timeout_s: float = 10.0
    max_retries: int = 4

    def _request(self, method: str, path: str, body: bytes | None = None,
                 content_type: str = "application/json", attempts: int | None = None) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.url.rstrip('/')}/{path.lstrip('/')}", data=body, method=method)
        req.add_header("Content-Type", content_type)
        if self.api_key:
            req.add_header("Authorization", f"ApiKey {self.api_key}")
        elif self.username:
            token = base64.b64encode(f"{self.username}:{self.password or ''}".encode()).decode()
            req.add_header("Authorization", f"Basic {token}")
        ctx = None if self.verify_tls else ssl._create_unverified_context()  # noqa: S323 (opt-in dev clusters)

        delay = 1.0
        max_attempts = attempts or self.max_retries
        for attempt in range(1, max_attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s, context=ctx) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                # 4xx (except 429) are permanent: bad mapping, auth, etc.
                if exc.code != 429 and exc.code < 500:
                    raise SinkError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
                err: Exception = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                err = exc
            if attempt == max_attempts:
                raise SinkError(f"{method} {path} failed after {attempt} attempts: {err}") from err
            logger.warning("Elasticsearch %s %s failed (%s); retrying in %.0fs", method, path, err, delay)
            time.sleep(delay)
            delay *= 2
        raise SinkError("unreachable")  # pragma: no cover

    def ping(self) -> bool:
        try:
            info = self._request("GET", "/", attempts=1)  # fail fast so the mock fallback is quick
            logger.info("Connected to Elasticsearch %s", info.get("version", {}).get("number", "?"))
            return True
        except SinkError as exc:
            logger.debug("Elasticsearch ping failed: %s", exc)
            return False

    def ensure_index(self) -> None:
        try:
            self._request("PUT", f"/{self.index}", json.dumps(INDEX_MAPPING).encode())
            logger.info("Created index %s", self.index)
        except SinkError as exc:
            if "resource_already_exists_exception" not in str(exc):
                raise

    def index_batch(self, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0
        lines: list[str] = []
        for d in docs:
            lines.append(json.dumps({"index": {"_index": self.index}}))
            lines.append(json.dumps(d))
        body = ("\n".join(lines) + "\n").encode()
        resp = self._request("POST", "/_bulk", body, content_type="application/x-ndjson")
        if resp.get("errors"):
            failed = [i["index"]["error"] for i in resp.get("items", []) if "error" in i.get("index", {})]
            logger.error("%d/%d bulk items failed; first error: %s", len(failed), len(docs), failed[:1])
            return len(docs) - len(failed)
        return len(docs)

    def close(self) -> None:
        try:
            self._request("POST", f"/{self.index}/_refresh")
        except SinkError as exc:
            logger.warning("Index refresh failed: %s", exc)


@dataclass
class LocalIndexSink:
    """File-backed mock index (one NDJSON document per line)."""

    root: str | Path = "data/es_mock"
    index: str = DEFAULT_INDEX
    reset: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.root) / f"{self.index}.ndjson"
        self._fh = None

    def ensure_index(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        (self.path.parent / f"{self.index}.mapping.json").write_text(json.dumps(INDEX_MAPPING, indent=2))
        self._fh = self.path.open("w" if self.reset else "a", encoding="utf-8")

    def index_batch(self, docs: list[dict[str, Any]]) -> int:
        if self._fh is None:
            self.ensure_index()
        for d in docs:
            self._fh.write(json.dumps(d) + "\n")
        return len(docs)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    # ---- read side: tiny query helpers ------------------------------------ #
    def _docs(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def count(self, **term: Any) -> int:
        return sum(1 for _ in self.search(**term))

    def search(self, **term: Any) -> Iterator[dict[str, Any]]:
        """Exact-match filter on top-level fields, e.g. ``search(is_flagged=True)``."""
        for d in self._docs():
            if all(d.get(k) == v for k, v in term.items()):
                yield d

    def top_alerts(self, n: int = 10) -> list[dict[str, Any]]:
        return sorted(self.search(is_flagged=True), key=lambda d: d["anomaly_score"], reverse=True)[:n]


# --------------------------------------------------------------------------- #
# Alerting
# --------------------------------------------------------------------------- #
@dataclass
class AlertManager:
    """Send flagged documents to the log, an alerts file and an optional webhook."""

    alerts_path: str | Path | None = "data/es_mock/alerts.jsonl"
    webhook_url: str | None = None
    min_severity: str = "medium"
    log_alerts: bool = False
    _order = ("none", "medium", "high", "critical")

    def __post_init__(self) -> None:
        self.sent = 0
        if self.alerts_path:
            Path(self.alerts_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.alerts_path).write_text("")

    def handle(self, doc: dict[str, Any]) -> bool:
        if not doc["is_flagged"] or self._order.index(doc["severity"]) < self._order.index(self.min_severity):
            return False
        msg = (f"[K8Shield] {doc['severity'].upper()} anomaly score={doc['anomaly_score']:.3f} "
               f"user={doc['user']} ip={doc['source_ip']} window={doc['@timestamp']}")
        if self.log_alerts:
            logger.warning(msg)
        if self.alerts_path:
            with Path(self.alerts_path).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(doc) + "\n")
        if self.webhook_url:
            self._post_webhook(msg)
        self.sent += 1
        return True

    def _post_webhook(self, text: str) -> None:
        req = urllib.request.Request(self.webhook_url, data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.error("Webhook delivery failed: %s", exc)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _chunks(it: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def make_sink(es_url: str | None = None, index: str = DEFAULT_INDEX, mock_dir: str | Path = "data/es_mock",
              **es_kwargs: Any) -> DocumentSink:
    """Return an ``ElasticsearchSink`` if ``es_url`` answers, otherwise a ``LocalIndexSink``."""
    if es_url:
        sink = ElasticsearchSink(url=es_url, index=index, **es_kwargs)
        if sink.ping():
            return sink
        logger.warning("Elasticsearch at %s is unreachable; falling back to the local mock index", es_url)
    return LocalIndexSink(root=mock_dir, index=index)


def stream_to_index(
    scored: pd.DataFrame,
    feature_names: list[str],
    sink: DocumentSink,
    alerts: AlertManager | None = None,
    model_version: str = "unknown",
    batch_size: int = 500,
    replay_delay_s: float = 0.0,
) -> dict[str, int]:
    """Stream documents window by window into ``sink``, alerting on flagged ones.

    ``replay_delay_s`` sleeps between batches to mimic a live feed in a demo.
    """
    sink.ensure_index()
    indexed = flagged = 0
    try:
        for batch in _chunks(build_documents(scored, feature_names, model_version), batch_size):
            indexed += sink.index_batch(batch)
            for doc in batch:
                if doc["is_flagged"]:
                    flagged += 1
                    if alerts:
                        alerts.handle(doc)
            if replay_delay_s:
                time.sleep(replay_delay_s)
    finally:
        sink.close()
    stats = {"indexed": indexed, "flagged": flagged, "alerts_sent": alerts.sent if alerts else 0}
    logger.info("Indexed %d documents into %s (%d flagged)", indexed, type(sink).__name__, flagged)
    return stats


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Score an audit log and stream results to Elasticsearch.")
    parser.add_argument("audit_log")
    parser.add_argument("--model", default="models")
    parser.add_argument("--es-url", default=None, help="e.g. http://localhost:9200 (omit for mock index)")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--webhook-url", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from feature_extractor import load_events
    from model_trainer import K8ShieldModel

    model = K8ShieldModel.load(args.model)
    if model.extractor is None:
        raise SystemExit("Model bundle has no feature-extractor state; retrain with `python main.py train`")
    scored = model.score(model.extractor.transform(load_events(args.audit_log)))
    stats = stream_to_index(scored, model.feature_names, make_sink(args.es_url, args.index),
                            AlertManager(webhook_url=args.webhook_url), model.metadata.get("model_version", "?"))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
