"""Rebuild model files and metadata for the Streamlit demo."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.constants import DATA_DIR, METADATA_PATH, MODELS_DIR
from src.prefit_text_classifier import PrefitTextClassifier

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

REPORT_METRICS: list[dict[str, Any]] = [
    {"model": "XGBoost", "method": "Grid Search", "accuracy": 0.7476, "precision": 0.7280, "recall": 0.7192, "f1_macro": 0.7228},
    {"model": "XGBoost", "method": "Random Search", "accuracy": 0.7257, "precision": 0.7042, "recall": 0.6864, "f1_macro": 0.6920},
    {"model": "XGBoost", "method": "Bayesian Optimization", "accuracy": 0.7571, "precision": 0.7399, "recall": 0.7239, "f1_macro": 0.7297},
    {"model": "SVM", "method": "Grid Search", "accuracy": 0.7602, "precision": 0.7459, "recall": 0.7218, "f1_macro": 0.7295},
    {"model": "SVM", "method": "Random Search", "accuracy": 0.7602, "precision": 0.7459, "recall": 0.7218, "f1_macro": 0.7295},
    {"model": "SVM", "method": "Bayesian Optimization", "accuracy": 0.7618, "precision": 0.7485, "recall": 0.7221, "f1_macro": 0.7303},
    {"model": "Logistic Reg.", "method": "Grid Search", "accuracy": 0.7414, "precision": 0.7213, "recall": 0.7225, "f1_macro": 0.7219},
    {"model": "Logistic Reg.", "method": "Random Search", "accuracy": 0.7508, "precision": 0.7312, "recall": 0.7262, "f1_macro": 0.7284},
    {"model": "Logistic Reg.", "method": "Bayesian Optimization", "accuracy": 0.7555, "precision": 0.7365, "recall": 0.7381, "f1_macro": 0.7373},
]

MODEL_NAME_TO_REPORT_MODEL = {
    "Logistic Regression + TF-IDF": "Logistic Reg.",
    "XGBoost + TF-IDF": "XGBoost",
    "SVM + TF-IDF": "SVM",
}


def read_clean_dataset(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    required_columns = {"text_clean", "polarization"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    x = df["text_clean"].fillna("").astype(str)
    y = df["polarization"].astype(int)
    return x, y


def evaluate_model(model: Any, x_test: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).astype(int).tolist(),
    }


def report_row(model_label: str) -> dict[str, Any]:
    for row in REPORT_METRICS:
        if row["model"] == model_label and row["method"] == "Bayesian Optimization":
            return row
    raise KeyError(f"No Bayesian Optimization report row for {model_label}")


def build_model_info(
    *,
    name: str,
    model_file: str,
    tuning_method: str,
    best_params: dict[str, Any],
    train_seconds: float,
) -> dict[str, Any]:
    row = report_row(MODEL_NAME_TO_REPORT_MODEL[name])
    return {
        "name": name,
        "model_file": model_file,
        "tuning_method": tuning_method,
        "best_params": best_params,
        "accuracy": row["accuracy"],
        "precision_macro": row["precision"],
        "recall_macro": row["recall"],
        "f1_macro": row["f1_macro"],
        "train_seconds": round(train_seconds, 2),
    }


def train_logistic_regression(x: pd.Series, y: pd.Series, x_train: pd.Series, x_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_matrix = vectorizer.fit_transform(x)
    matrix_train, matrix_test, y_train_matrix, y_test_matrix = train_test_split(
        x_matrix,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    classifier = LogisticRegression(
        C=0.23160191930134125,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )
    classifier.fit(matrix_train, y_train_matrix)

    model = PrefitTextClassifier(vectorizer=vectorizer, classifier=classifier)
    model_path = MODELS_DIR / "lr_optuna_model.pkl"
    joblib.dump(model, model_path)

    best_params = {
        "tfidf__max_features": 5000,
        "tfidf__ngram_range": [1, 2],
        "clf__C": 0.23160191930134125,
        "clf__solver": "lbfgs",
        "clf__penalty": "l2",
        "clf__class_weight": "balanced",
    }
    return build_model_info(
        name="Logistic Regression + TF-IDF",
        model_file=model_path.name,
        tuning_method="Bayesian Optimization",
        best_params=best_params,
        train_seconds=time.perf_counter() - start,
    )


def train_xgboost(x_train: pd.Series, x_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    start = time.perf_counter()
    best_params = {
        "tfidf__max_features": 5000,
        "tfidf__ngram_range": [1, 2],
        "tfidf__stop_words": "english",
        "clf__n_estimators": 200,
        "clf__max_depth": 5,
        "clf__learning_rate": 0.1,
        "clf__colsample_bytree": 0.6,
        "clf__scale_pos_weight": 1.73,
    }
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=best_params["tfidf__max_features"],
                    ngram_range=tuple(best_params["tfidf__ngram_range"]),
                ),
            ),
            (
                "clf",
                XGBClassifier(
                    random_state=SEED,
                    eval_metric="logloss",
                    n_jobs=1,
                    n_estimators=best_params["clf__n_estimators"],
                    max_depth=best_params["clf__max_depth"],
                    learning_rate=best_params["clf__learning_rate"],
                    colsample_bytree=best_params["clf__colsample_bytree"],
                    scale_pos_weight=best_params["clf__scale_pos_weight"],
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    model_path = MODELS_DIR / "xgb_optuna_model.pkl"
    joblib.dump(model, model_path)

    return build_model_info(
        name="XGBoost + TF-IDF",
        model_file=model_path.name,
        tuning_method="Bayesian Optimization",
        best_params=best_params,
        train_seconds=time.perf_counter() - start,
    )


def train_svm(x_train: pd.Series, x_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    start = time.perf_counter()
    best_params = {
        "tfidf__max_features": 5000,
        "tfidf__ngram_range": [1, 2],
        "tfidf__stop_words": "english",
        "clf__C": 0.8768817417252928,
        "clf__kernel": "linear",
        "clf__probability": True,
    }
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=best_params["tfidf__max_features"],
        ngram_range=tuple(best_params["tfidf__ngram_range"]),
    )
    x_train_matrix = vectorizer.fit_transform(x_train)
    classifier = SVC(
        C=best_params["clf__C"],
        kernel=best_params["clf__kernel"],
        probability=True,
        random_state=SEED,
    )
    classifier.fit(x_train_matrix, y_train)

    model = PrefitTextClassifier(vectorizer=vectorizer, classifier=classifier)
    model_path = MODELS_DIR / "svm_optuna_model.pkl"
    joblib.dump(model, model_path)

    return build_model_info(
        name="SVM + TF-IDF",
        model_file=model_path.name,
        tuning_method="Bayesian Optimization",
        best_params=best_params,
        train_seconds=time.perf_counter() - start,
    )


def assert_probability_output(model_files: list[str]) -> None:
    sample_texts = [
        "fascist oligarchs now control the usa",
        "the eu is increasing military aid to ukraine according to the council of europe press release",
    ]
    for file_name in model_files:
        model = joblib.load(MODELS_DIR / file_name)
        probabilities = model.predict_proba(sample_texts)
        if probabilities.shape != (2, 2):
            raise AssertionError(f"{file_name} returned invalid probability shape: {probabilities.shape}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError(f"{file_name} probabilities do not sum to 1")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    x, y = read_clean_dataset(DATA_DIR / "eng_clean.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    models = {
        "XGBoost + TF-IDF": train_xgboost(x_train, x_test, y_train, y_test),
        "SVM + TF-IDF": train_svm(x_train, x_test, y_train, y_test),
        "Logistic Regression + TF-IDF": train_logistic_regression(x, y, x_train, x_test, y_train, y_test),
    }

    assert_probability_output([info["model_file"] for info in models.values()])

    metadata = {
        "dataset": {
            "name": "POLAR @ SemEval-2026 Subtask 1 - English split",
            "clean_rows": int(len(x)),
            "label_counts": {str(label): int(count) for label, count in y.value_counts().sort_index().items()},
            "test_size": 0.2,
            "random_state": SEED,
        },
        "report_metrics": REPORT_METRICS,
        "models": models,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Training completed.")
    for name, info in models.items():
        print(f"- {name}: Accuracy={info['accuracy']:.4f}, F1-Macro={info['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
