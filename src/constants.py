"""Shared constants for the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

LABEL_NAMES: dict[int, str] = {
    0: "Non-Polarized",
    1: "Polarized",
}

LABEL_DESCRIPTIONS: dict[int, str] = {
    0: "Văn bản không chứa hoặc không thể hiện quan điểm phân cực rõ ràng.",
    1: "Văn bản chứa quan điểm phân cực hoặc thể hiện sự chia rẽ lập trường.",
}

MODEL_FILES: dict[str, str] = {
    "XGBoost + TF-IDF": "xgb_optuna_model.pkl",
    "SVM + TF-IDF": "svm_optuna_model.pkl",
    "Logistic Regression + TF-IDF": "lr_optuna_model.pkl",
}


class DemoExample(TypedDict):
    label: str
    text: str
    expected_label: int | None


DEMO_EXAMPLES: dict[str, DemoExample] = {
    "Tự nhập": {
        "label": "Tự nhập",
        "text": "",
        "expected_label": None,
    },
    "Mẫu nhãn 0 - tin tức": {
        "label": "Mẫu nhãn 0 - tin tức",
        "text": "The report said the council will meet on Friday to discuss the new budget.",
        "expected_label": 0,
    },
    "Mẫu nhãn 0 - thông báo": {
        "label": "Mẫu nhãn 0 - thông báo",
        "text": "The local newspaper reported that the city council approved the new road project.",
        "expected_label": 0,
    },
    "Mẫu nhãn 1 - tranh luận": {
        "label": "Mẫu nhãn 1 - tranh luận",
        "text": "The system is corrupt, the election was rigged, and those criminals are destroying the country.",
        "expected_label": 1,
    },
    "Mẫu nhãn 1 - phát ngôn mạnh": {
        "label": "Mẫu nhãn 1 - phát ngôn mạnh",
        "text": "The apartheid state must be stopped because ethnic cleansing is unacceptable.",
        "expected_label": 1,
    },
}
