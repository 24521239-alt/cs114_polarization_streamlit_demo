"""Reusable wrapper for models trained on a pre-fitted TF-IDF vectorizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _as_text_list(texts: str | Iterable[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(item) for item in texts]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class PrefitTextClassifier:
    """Text classifier with a TF-IDF vectorizer fitted outside a Pipeline.

    This wrapper is useful when a notebook fits TF-IDF before splitting the
    matrix, then trains the classifier on the resulting train matrix. Keeping
    the fitted vectorizer and classifier together lets the Streamlit app call
    predict/predict_proba with plain text input.
    """

    vectorizer: Any
    classifier: Any

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(getattr(self.classifier, "classes_", np.array([0, 1])))

    def _transform(self, texts: str | Iterable[str]) -> Any:
        return self.vectorizer.transform(_as_text_list(texts))

    def predict(self, texts: str | Iterable[str]) -> np.ndarray:
        return self.classifier.predict(self._transform(texts))

    def predict_proba(self, texts: str | Iterable[str]) -> np.ndarray:
        features = self._transform(texts)

        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(features)

        if not hasattr(self.classifier, "decision_function"):
            predictions = self.classifier.predict(features)
            probabilities = np.zeros((len(predictions), len(self.classes_)), dtype=float)
            for row_idx, predicted_label in enumerate(predictions):
                class_idx = int(np.where(self.classes_ == predicted_label)[0][0])
                probabilities[row_idx, class_idx] = 1.0
            return probabilities

        scores = np.asarray(self.classifier.decision_function(features), dtype=float)
        if scores.ndim == 1:
            positive_probability = _sigmoid(scores)
            return np.column_stack([1.0 - positive_probability, positive_probability])

        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
