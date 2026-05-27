"""Prediction utilities for the polarization detection demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.constants import LABEL_NAMES
from src.preprocessing import clean_text

MIN_CLEAN_TOKENS = 3
MIN_ACTIVE_FEATURES = 2


@dataclass(frozen=True)
class InputValidation:
    is_valid: bool
    message: str | None
    token_count: int
    active_feature_count: int | None


@dataclass(frozen=True)
class PredictionResult:
    raw_text: str
    cleaned_text: str
    predicted_label: int | None
    predicted_name: str | None
    probabilities: dict[int, float]
    validation: InputValidation


def _get_vectorizer(model: Any) -> Any | None:
    vectorizer = getattr(model, "vectorizer", None)
    if vectorizer is not None and hasattr(vectorizer, "transform"):
        return vectorizer

    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        for step in named_steps.values():
            if hasattr(step, "vocabulary_") and hasattr(step, "transform"):
                return step

    steps = getattr(model, "steps", None)
    if steps:
        for _, step in steps:
            if hasattr(step, "vocabulary_") and hasattr(step, "transform"):
                return step

    return None


def _count_active_features(model: Any, cleaned_text: str) -> int | None:
    vectorizer = _get_vectorizer(model)
    if vectorizer is None:
        return None

    vector = vectorizer.transform([cleaned_text])
    return int(vector.nnz)


def validate_text(model: Any, cleaned_text: str) -> InputValidation:
    token_count = len(cleaned_text.split())

    if token_count < MIN_CLEAN_TOKENS:
        return InputValidation(
            is_valid=False,
            message=(
                "Văn bản quá ngắn hoặc chưa đủ ngữ cảnh để phân tích. "
                "Vui lòng nhập một câu tiếng Anh đầy đủ hơn."
            ),
            token_count=token_count,
            active_feature_count=None,
        )

    active_feature_count = _count_active_features(model, cleaned_text)
    if active_feature_count is not None and active_feature_count < MIN_ACTIVE_FEATURES:
        return InputValidation(
            is_valid=False,
            message=(
                "Văn bản chưa có đủ đặc trưng phù hợp với dữ liệu huấn luyện. "
                "Vui lòng nhập một câu có ngữ cảnh rõ hơn."
            ),
            token_count=token_count,
            active_feature_count=active_feature_count,
        )

    return InputValidation(
        is_valid=True,
        message=None,
        token_count=token_count,
        active_feature_count=active_feature_count,
    )


def _get_probabilities(model: Any, cleaned_text: str) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The loaded model does not support probability prediction.")

    classes = getattr(model, "classes_", np.array([0, 1]))
    proba = model.predict_proba([cleaned_text])[0]
    probability_map = {int(label): float(prob) for label, prob in zip(classes, proba)}

    return {label: probability_map.get(label, 0.0) for label in LABEL_NAMES}


def predict_text(model: Any, raw_text: str) -> PredictionResult:
    cleaned_text = clean_text(raw_text)
    validation = validate_text(model, cleaned_text)

    if not validation.is_valid:
        return PredictionResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            predicted_label=None,
            predicted_name=None,
            probabilities={},
            validation=validation,
        )

    probabilities = _get_probabilities(model, cleaned_text)
    predicted_label = max(probabilities, key=probabilities.get)

    return PredictionResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        predicted_label=predicted_label,
        predicted_name=LABEL_NAMES.get(predicted_label, str(predicted_label)),
        probabilities=probabilities,
        validation=validation,
    )
