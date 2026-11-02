"""Train, score with, and persist the K8Shield Isolation Forest detector.

The model is a scikit-learn ``Pipeline(StandardScaler -> IsolationForest)``.
Training is unsupervised: ground-truth labels, if present, are ignored here
and only used later by ``evaluate.py``.

Scoring convention
------------------
``anomaly_score``
    ``-IsolationForest.score_samples(x)``, the anomaly score ``s(x, n)`` from
    Liu et al. (2008), in (0, 1]. Higher is more anomalous; values near 1 are
    clear outliers and values well below 0.5 are clearly normal.
``prediction``
    ``1`` for normal, ``-1`` for anomaly. This is scikit-learn's convention,
    with the threshold set by ``contamination``.

Artifacts written by ``K8ShieldModel.save``
-------------------------------------------
``k8shield_model.joblib``
    A bundle with the full pipeline, the fitted feature-extractor state (the
    user-agent table), the feature names, the hyper-parameters and metadata.
``k8shield_scaler.joblib``
    The fitted ``StandardScaler`` saved on its own, for consumers that only
    need the scaling step.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

MODEL_FILENAME = "k8shield_model.joblib"
SCALER_FILENAME = "k8shield_scaler.joblib"
MODEL_VERSION = "1.0.0"


class ModelNotFittedError(RuntimeError):
    """Raised when scoring is attempted before training or loading."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    n_estimators: int = 300
    max_samples: int | float | str = "auto"
    contamination: float | str = 0.05
    max_features: float = 1.0
    random_state: int = 42
    n_jobs: int | None = -1

    def __post_init__(self) -> None:
        if isinstance(self.contamination, float) and not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5] or 'auto'")


class K8ShieldModel:
    """Isolation Forest anomaly detector over windowed audit features."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        feature_names: list[str] | None = None,
        extractor: FeatureExtractor | None = None,
    ):
        self.config = config or ModelConfig()
        self.feature_names = feature_names
        self.extractor = extractor
        self.pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] = {}

    # ---- training --------------------------------------------------------- #
    def _build_pipeline(self) -> Pipeline:
        c = self.config
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "iforest",
                    IsolationForest(
                        n_estimators=c.n_estimators,
                        max_samples=c.max_samples,
                        contamination=c.contamination,
                        max_features=c.max_features,
                        random_state=c.random_state,
                        n_jobs=c.n_jobs,
                    ),
                ),
            ]
        )

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None:
            raise ModelNotFittedError("feature_names unknown; train or load a model first")
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise KeyError(f"Feature frame is missing columns: {missing}")
        x = features[self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            logger.warning("Non-finite feature values found; replacing them with 0")
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def fit(self, features: pd.DataFrame) -> K8ShieldModel:
        """Fit the scaler and Isolation Forest on a feature frame (labels ignored)."""
        if self.feature_names is None:
            self.feature_names = (
                self.extractor.config.feature_names if self.extractor else
                [c for c in features.select_dtypes("number").columns if c != "is_anomaly"]
            )
        if len(features) < 10:
            raise ValueError(f"Need at least 10 training rows, got {len(features)}")
        x = self._matrix(features)
        self.pipeline = self._build_pipeline().fit(x)
        train_scores = -self.pipeline.score_samples(x)
        self.metadata = {
            "model_version": MODEL_VERSION,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "n_train_rows": int(len(x)),
            "feature_names": list(self.feature_names),
            "train_score_mean": float(train_scores.mean()),
            "train_flag_rate": float((self.pipeline.predict(x) == -1).mean()),
            "threshold_score": float(-self.pipeline.named_steps["iforest"].offset_),
        }
        logger.info(
            "Trained IsolationForest on %d rows x %d features (flag rate %.2f%%, score threshold %.3f)",
            len(x), x.shape[1], 100 * self.metadata["train_flag_rate"], self.metadata["threshold_score"],
        )
        return self

    # ---- scoring ---------------------------------------------------------- #
    def _require_fitted(self) -> Pipeline:
        if self.pipeline is None:
            raise ModelNotFittedError("Model is not trained; call fit() or K8ShieldModel.load()")
        return self.pipeline

    def anomaly_scores(self, features: pd.DataFrame) -> np.ndarray:
        """Anomaly score in (0, 1]; higher is more anomalous."""
        return -self._require_fitted().score_samples(self._matrix(features))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """``1`` for normal, ``-1`` for anomaly."""
        return self._require_fitted().predict(self._matrix(features))

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``features`` with ``anomaly_score``, ``prediction`` and ``is_flagged``."""
        out = features.copy()
        out["anomaly_score"] = self.anomaly_scores(features)
        out["prediction"] = self.predict(features)
        out["is_flagged"] = out["prediction"] == -1
        return out

    # ---- persistence ------------------------------------------------------ #
    def save(self, model_dir: str | Path) -> tuple[Path, Path]:
        """Save the model bundle and the standalone scaler. Returns both paths."""
        pipeline = self._require_fitted()
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": pipeline,
            "feature_names": self.feature_names,
            "model_config": asdict(self.config),
            "extractor_state": self.extractor.to_state() if self.extractor else None,
            "metadata": self.metadata,
        }
        model_path = model_dir / MODEL_FILENAME
        scaler_path = model_dir / SCALER_FILENAME
        joblib.dump(bundle, model_path, compress=3)
        joblib.dump(pipeline.named_steps["scaler"], scaler_path)
        logger.info("Saved model to %s and scaler to %s", model_path, scaler_path)
        return model_path, scaler_path

    @classmethod
    def load(cls, path: str | Path) -> K8ShieldModel:
        """Load a bundle from a ``.joblib`` file or the directory that holds it."""
        path = Path(path)
        if path.is_dir():
            path = path / MODEL_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found: {path}. Run `python main.py train` first.")
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise ValueError(f"{path} is not a K8Shield model bundle")
        saved_version = bundle.get("metadata", {}).get("sklearn_version")
        if saved_version and saved_version != sklearn.__version__:
            logger.warning("Model trained with scikit-learn %s, running %s", saved_version, sklearn.__version__)
        extractor = (
            FeatureExtractor.from_state(bundle["extractor_state"]) if bundle.get("extractor_state") else None
        )
        model = cls(ModelConfig(**bundle["model_config"]), bundle["feature_names"], extractor)
        model.pipeline = bundle["pipeline"]
        model.metadata = bundle.get("metadata", {})
        return model


def train_model(
    features: pd.DataFrame,
    extractor: FeatureExtractor | None = None,
    config: ModelConfig | None = None,
    model_dir: str | Path | None = None,
) -> K8ShieldModel:
    """Train a model and optionally save it to ``model_dir``."""
    model = K8ShieldModel(config, extractor=extractor).fit(features)
    if model_dir is not None:
        model.save(model_dir)
    return model


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train the K8Shield Isolation Forest.")
    parser.add_argument("audit_log", help="Training audit log (JSON Lines)")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--contamination", default="0.05", help="float or 'auto'")
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from feature_extractor import load_events

    contamination: float | str = args.contamination if args.contamination == "auto" else float(args.contamination)
    extractor = FeatureExtractor()
    features = extractor.fit_transform(load_events(args.audit_log))
    train_model(features, extractor, ModelConfig(n_estimators=args.n_estimators, contamination=contamination),
                args.model_dir)


if __name__ == "__main__":
    _cli()
