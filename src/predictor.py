"""Prediction utilities for the polarization detection demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.constants import LABEL_NAMES
from src.preprocessing import clean_text, is_too_short


@dataclass(frozen=True)
class PredictionResult:
    raw_text: str
    cleaned_text: str
    predicted_label: int
    predicted_name: str
    probabilities: dict[int, float]
    too_short: bool


def _get_probabilities(model: Any, cleaned_text: str) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The loaded model does not support probability prediction.")

    classes = getattr(model, "classes_", np.array([0, 1]))
    proba = model.predict_proba([cleaned_text])[0]

    return {int(label): float(prob) for label, prob in zip(classes, proba)}


def predict_text(model: Any, raw_text: str) -> PredictionResult:
    cleaned_text = clean_text(raw_text)
    probabilities = _get_probabilities(model, cleaned_text)

    if probabilities:
        predicted_label = max(probabilities, key=probabilities.get)
    else:
        predicted_label = int(model.predict([cleaned_text])[0])

    return PredictionResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        predicted_label=predicted_label,
        predicted_name=LABEL_NAMES.get(predicted_label, str(predicted_label)),
        probabilities=probabilities,
        too_short=is_too_short(cleaned_text),
    )
