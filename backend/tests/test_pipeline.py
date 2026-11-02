"""Fast end-to-end and unit tests for K8Shield (run with ``pytest -q``)."""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from es_pipeline import AlertManager, LocalIndexSink, build_documents, stream_to_index  # noqa: E402
from evaluate import RuleBasedDetector, binary_metrics, evaluate, fpr_at_recall  # noqa: E402
from feature_extractor import (  # noqa: E402
    CORE_FEATURES,
    FeatureConfig,
    FeatureExtractor,
    events_to_frame,
    load_events,
)
from log_generator import (  # noqa: E402
    AUDIT_API_VERSION,
    GROUND_TRUTH_ANNOTATION,
    AuditLogGenerator,
    GeneratorConfig,
    build_audit_event,
)
from model_trainer import K8ShieldModel, ModelConfig  # noqa: E402


@pytest.fixture(scope="module")
def logs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("logs")
    train, test = d / "train.jsonl", d / "test.jsonl"
    AuditLogGenerator(GeneratorConfig(duration_minutes=360, seed=1)).write_jsonl(train)
    AuditLogGenerator(GeneratorConfig(duration_minutes=360, seed=2)).write_jsonl(test)
    return train, test


@pytest.fixture(scope="module")
def features(logs: tuple[Path, Path]) -> tuple[FeatureExtractor, pd.DataFrame, pd.DataFrame]:
    fx = FeatureExtractor()
    return fx, fx.fit_transform(load_events(logs[0])), fx.transform(load_events(logs[1]))


def _event(verb: str, minute: int = 0, **kw):
    return build_audit_event(
        ts=datetime(2026, 1, 1, 0, minute, tzinfo=UTC), username="u", groups=[], source_ip="10.0.0.1",
        user_agent=kw.pop("ua", "kubectl"), verb=verb, resource=kw.pop("resource", "pods"),
        namespace=kw.pop("namespace", "default"), name=None, subresource=None, code=kw.pop("code", 200), **kw)


# ---- generator --------------------------------------------------------------
def test_generator_emits_valid_audit_events(logs):
    with logs[0].open() as fh:
        events = [json.loads(next(fh)) for _ in range(200)]
    for ev in events:
        assert ev["apiVersion"] == AUDIT_API_VERSION and ev["kind"] == "Event"
        assert {"auditID", "verb", "user", "sourceIPs", "requestURI", "responseStatus"} <= ev.keys()
    timestamps = [ev["requestReceivedTimestamp"] for ev in events]
    assert timestamps == sorted(timestamps)


def test_generator_respects_anomaly_ratio():
    gen = AuditLogGenerator(GeneratorConfig(duration_minutes=720, anomaly_ratio=0.10, seed=3))
    for _ in gen.generate():
        pass
    s = gen.stats
    share = s["anomalous_entity_windows"] / (s["anomalous_entity_windows"] + s["normal_entity_windows"])
    assert 0.07 < share < 0.13


def test_zero_anomaly_ratio_has_no_labels():
    gen = AuditLogGenerator(GeneratorConfig(duration_minutes=30, anomaly_ratio=0.0, seed=4))
    assert not any(GROUND_TRUTH_ANNOTATION in e["annotations"] for e in gen.generate())


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        GeneratorConfig(anomaly_ratio=1.5)


# ---- features ---------------------------------------------------------------
def test_verb_entropy_and_error_rate():
    events = [_event("get"), _event("create", code=403), _event("patch"), _event("delete", code=401)]
    fx = FeatureExtractor(FeatureConfig(include_extended=False))
    row = fx.fit_transform(events_to_frame(events)).iloc[0]
    assert math.isclose(row["verb_entropy"], 2.0)  # four equally likely HTTP methods
    assert math.isclose(row["error_code_rate"], 0.5)
    assert row["request_count"] == 4
    assert list(fx.config.feature_names) == list(CORE_FEATURES)


def test_rbac_flag_and_unseen_user_agent():
    events = [_event("create", resource="clusterrolebindings", namespace=None, ua="evil/1.0")]
    fx = FeatureExtractor().fit(events_to_frame([_event("get")]))
    row = fx.transform(events_to_frame(events)).iloc[0]
    assert row["rbac_escalation_flag"] == 1
    assert row["unusual_user_agent_score"] == 1.0  # never seen at fit time


def test_sliding_windows_duplicate_events():
    events = [_event("get", minute=0), _event("get", minute=3)]
    tumbling = FeatureExtractor(FeatureConfig(5)).fit_transform(events_to_frame(events))
    sliding = FeatureExtractor(FeatureConfig(5, 1)).fit_transform(events_to_frame(events))
    assert len(tumbling) == 1 and len(sliding) > 1
    assert sliding["request_count"].max() == 2


def test_feature_frame_is_clean(features):
    _, train, _ = features
    assert {"window_start", "source_ip", "user", "is_anomaly", "anomaly_type"} <= set(train.columns)
    assert np.isfinite(train[FeatureConfig().feature_names].to_numpy()).all()
    assert 0.0 < train["is_anomaly"].mean() < 0.15


# ---- model ------------------------------------------------------------------
def test_model_train_save_load_roundtrip(features, tmp_path):
    fx, train, test = features
    model = K8ShieldModel(ModelConfig(n_estimators=50), extractor=fx).fit(train)
    model_path, scaler_path = model.save(tmp_path)
    assert model_path.exists() and scaler_path.exists()
    loaded = K8ShieldModel.load(tmp_path)
    np.testing.assert_allclose(model.anomaly_scores(test), loaded.anomaly_scores(test))
    assert set(np.unique(loaded.predict(test))) <= {-1, 1}
    assert loaded.extractor is not None and loaded.extractor.ua_frequency_ == fx.ua_frequency_


def test_unfitted_model_raises(features):
    with pytest.raises(RuntimeError):
        K8ShieldModel().anomaly_scores(features[1])


# ---- evaluation -------------------------------------------------------------
def test_binary_metrics():
    m = binary_metrics(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert math.isclose(m["precision"], 0.5) and math.isclose(m["fpr"], 1 / 3)


def test_fpr_at_recall_monotone():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.5, 0.9])
    assert fpr_at_recall(y, s, 0.5)[0] == 0.0
    assert fpr_at_recall(y, s, 1.0)[0] == pytest.approx(1 / 3)


def test_evaluate_writes_artifacts(features, tmp_path):
    fx, train, test = features
    model = K8ShieldModel(ModelConfig(n_estimators=50), extractor=fx).fit(train)
    result = evaluate(model, test, tmp_path)
    for name in ("metrics.json", "benchmark_report.md", "confusion_matrices.png", "roc_curve.png"):
        assert (tmp_path / name).exists()
    assert 0.0 <= result.iforest["fpr"] <= 1.0
    assert result.iforest_auc > 0.8
    assert RuleBasedDetector().predict(test).dtype == bool


# ---- Elasticsearch mock -----------------------------------------------------
def test_stream_to_local_index(features, tmp_path):
    fx, train, test = features
    model = K8ShieldModel(ModelConfig(n_estimators=50), extractor=fx).fit(train)
    scored = model.score(test)
    sink = LocalIndexSink(root=tmp_path)
    stats = stream_to_index(scored, model.feature_names, sink, AlertManager(alerts_path=tmp_path / "a.jsonl"))
    assert stats["indexed"] == len(scored) == sink.count()
    assert sink.count(is_flagged=True) == int(scored["is_flagged"].sum()) == stats["flagged"]
    doc = next(build_documents(scored, model.feature_names))
    assert {"@timestamp", "source_ip", "user", "anomaly_score", "is_flagged"} <= doc.keys()
    assert "is_anomaly" not in doc  # labels never leak into the index
