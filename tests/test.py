import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from preprocessing import clean_column_names, feature_engineering, load_and_clean_data


def test_clean_column_names():
    df = pd.DataFrame({"Test Column Name": [1, 2], "Another.Col": [3, 4]})
    cleaned_df = clean_column_names(df)
    assert list(cleaned_df.columns) == ["test_column_name", "another_col"]


def test_feature_engineering():
    df = pd.DataFrame(
        {
            "sent_tnx": [10, 0],
            "received_tnx": [5, 0],
            "total_ether_sent": [100.0, 0.0],
        }
    )
    processed = feature_engineering(df)

    # Создание ratio
    assert "ratio_sent_received" in processed.columns
    assert np.isclose(processed["ratio_sent_received"].iloc[0], 10 / (5 + 1e-5))

    # Логарифмирование
    assert "log_total_ether_sent" in processed.columns
    assert processed["log_total_ether_sent"].iloc[1] == pytest.approx(0.0, abs=1e-5)


def test_drop_duplicates():
    """Проверяем, что load_and_clean_data удаляет дубликаты."""
    import tempfile

    # Создаём временный CSV с дубликатами
    df = pd.DataFrame(
        {
            "Index": [1, 2, 2],
            "Address": ["0xA", "0xB", "0xB"],
            "FLAG": [0, 1, 1],
            "sent_tnx": [5, 10, 10],
            "received_tnx": [3, 2, 2],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        cleaned = load_and_clean_data(tmp_path)
        # После удаления дубликата должно остаться 2 строки
        assert len(cleaned) == 2, f"Ожидалось 2 строки, получено {len(cleaned)}"
    finally:
        os.unlink(tmp_path)


def test_no_constant_columns():
    """Проверяем, что константные колонки удаляются."""
    import tempfile

    df = pd.DataFrame(
        {
            "Index": [1, 2],
            "FLAG": [0, 1],
            "sent_tnx": [5, 10],
            "const_col": [0, 0],  # константная колонка
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        cleaned = load_and_clean_data(tmp_path)
        assert "const_col" not in cleaned.columns
    finally:
        os.unlink(tmp_path)


def test_feature_engineering_no_nan():
    """FE не должен порождать NaN в признаках."""
    df = pd.DataFrame(
        {
            "sent_tnx": [0, 5, 100],
            "received_tnx": [0, 0, 3],
            "total_ether_sent": [0.0, 50.0, 1000.0],
            "total_ether_received": [0.0, 20.0, 500.0],
            "total_ether_balance": [-10.0, 30.0, 200.0],
        }
    )
    processed = feature_engineering(df)
    assert not processed.isnull().any().any(), "FE породил NaN-значения!"