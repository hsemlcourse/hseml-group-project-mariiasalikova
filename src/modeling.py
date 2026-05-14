import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# ── Глобальный seed для воспроизводимости ──────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
# ───────────────────────────────────────────────────────────────────────────


def load_data(data_dir: str = "data/processed"):
    """Загружает предобработанные train/val/test CSV."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    val = pd.read_csv(f"{data_dir}/val.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    X_train, y_train = train.drop(columns=["flag"]), train["flag"]
    X_val, y_val = val.drop(columns=["flag"]), val["flag"]
    X_test, y_test = test.drop(columns=["flag"]), test["flag"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Возвращает словарь метрик для переданного набора данных."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    precision, recall, _ = precision_recall_curve(y, y_prob)
    pr_auc = auc(recall, precision)

    return {
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": pr_auc,
        "accuracy": (y_pred == y).mean(),
    }


def train_evaluate_model(model, X_train, y_train, X_val, y_val):
    """Обучает модель и выводит метрики на валидации."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print("Classification Report:\n", classification_report(y_val, y_pred))

    roc_auc = roc_auc_score(y_val, y_prob)
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"F1-Score: {f1_score(y_val, y_pred):.4f}")

    return model


def save_model(model, filename: str = "../models/best_model.pkl") -> None:
    """Сохраняет обученную модель на диск."""
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    joblib.dump(model, filename)
    print(f"Модель сохранена в {filename}")
