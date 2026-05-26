"""Model loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.constants import METADATA_PATH, MODEL_FILES, MODELS_DIR


def load_model(model_name: str) -> Any:
    """Load a trained pipeline by display name."""

    if model_name not in MODEL_FILES:
        valid = ", ".join(MODEL_FILES)
        raise ValueError(f"Unknown model '{model_name}'. Valid models: {valid}")

    model_path = MODELS_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    """Read model metadata generated during training."""

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))
