from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .dataset import LABEL_FAKE, LABEL_REAL, Sample, build, load_samples
from .text_utils import preprocess

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "truthlens_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42

def build_pipeline(variant: str = "logreg") -> Pipeline:
    
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=preprocess,
                    tokenizer=None,
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{2,}\b",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=60000,
                ),
            )
        ]
    )

    if variant == "svm":
        clf: Any = CalibratedClassifierCV(
            LinearSVC(C=0.3, random_state=RANDOM_STATE, max_iter=5000), cv=3
        )
    elif variant == "nb":
        clf = ComplementNB(alpha=0.3)
    else:
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)

    return Pipeline([("features", features), ("clf", clf)])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_fake": round(float(precision_score(y_true, y_pred, pos_label=LABEL_FAKE)), 4),
        "recall_fake": round(float(recall_score(y_true, y_pred, pos_label=LABEL_FAKE)), 4),
        "precision_real": round(float(precision_score(y_true, y_pred, pos_label=LABEL_REAL)), 4),
        "recall_real": round(float(recall_score(y_true, y_pred, pos_label=LABEL_REAL)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=["FAKE", "REAL"]),
    }


def train(variant: str = "logreg", holdout: bool = True) -> dict[str, Any]:
    """Train the pipeline, evaluate it, and persist artifacts."""
    samples = load_samples()
    if not samples:
        samples = build()

    texts = np.array([s.text for s in samples], dtype=object)
    labels = np.array([s.label for s in samples])

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    pipeline = build_pipeline(variant)
    pipeline.fit(X_train, y_train)

    if holdout:
        y_pred = pipeline.predict(X_test)
        metrics = _metrics(y_test, y_pred)
        print(
            f"[{variant}] holdout accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )
    else:
        metrics = {"note": "trained on full dataset, no holdout evaluation"}

    final_model = build_pipeline(variant)
    final_model.fit(texts, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH, compress=3)

    payload = {
        "variant": variant,
        "trained_at_samples": int(len(samples)),
        "label_map": {"FAKE": LABEL_FAKE, "REAL": LABEL_REAL},
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

@dataclass
class Prediction:
    label: str
    confidence: float
    fake_score: float
    real_score: float
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "is_fake": self.label == "FAKE",
            "confidence": round(self.confidence, 4),
            "confidence_percent": f"{round(self.confidence * 100, 1)}%",
            "scores": {"fake": round(self.fake_score, 4), "real": round(self.real_score, 4)},
            "word_count": self.word_count,
        }

class Predictor:
    """Lazy-loading wrapper around the persisted pipeline."""

    def __init__(self, path: Path = MODEL_PATH):
        self.path = path
        self._model: Pipeline | None = None
        self.metrics: dict[str, Any] = {}
        metrics_file = METRICS_PATH
        if metrics_file.exists():
            try:
                self.metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.metrics = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.path}. Run: python train.py"
            )
        self._model = joblib.load(self.path)

    def predict(self, text: str) -> Prediction:
        self.load()
        assert self._model is not None

        cleaned = preprocess(text)
        words = len(text.split())
        if not cleaned:
            raise ValueError("Input text is empty after preprocessing.")

        proba = self._model.predict_proba([cleaned])[0]
        classes = list(self._model.classes_)
        fake_score = float(proba[classes.index(LABEL_FAKE)]) if LABEL_FAKE in classes else 0.0
        real_score = float(proba[classes.index(LABEL_REAL)]) if LABEL_REAL in classes else 0.0
        label = "FAKE" if fake_score >= real_score else "REAL"

        return Prediction(
            label=label,
            confidence=max(fake_score, real_score),
            fake_score=fake_score,
            real_score=real_score,
            word_count=words,
        )

_predictor: Predictor | None = None

def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor

__all__ = [
    "Predictor",
    "Prediction",
    "build_pipeline",
    "train",
    "get_predictor",
    "MODEL_PATH",
    "METRICS_PATH",
]