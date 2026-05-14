import random
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Глобальный seed для воспроизводимости ──────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
# ───────────────────────────────────────────────────────────────────────────


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит названия колонок к snake_case для единообразия."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )
    return df


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Загружает сырые данные, чистит и возвращает DataFrame."""
    df = pd.read_csv(filepath)
    df = clean_column_names(df)

    # Удаляем дубликаты
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    print(f"Удалено дубликатов: {n_before - n_after} (было {n_before}, стало {n_after})")

    # Удаляем нерелевантные признаки
    cols_to_drop = ["unnamed:_0", "index", "address"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    # Удаляем категориальные токены с большим количеством None/уникальных значений
    token_cols = ["erc20_most_sent_token_type", "erc20_most_rec_token_type"]
    cols_to_drop.extend([c for c in token_cols if c in df.columns])

    df = df.drop(columns=cols_to_drop)

    # Заполнение пропусков (ERC20 транзакции без данных → 0)
    df = df.fillna(0)

    # Удаление колонок с нулевой дисперсией (константы)
    nunique = df.nunique()
    cols_to_drop_const = nunique[nunique == 1].index.tolist()
    if cols_to_drop_const:
        print(f"Удалены константные столбцы: {cols_to_drop_const}")
    df = df.drop(columns=cols_to_drop_const)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Генерация новых признаков."""
    # Отношение отправленных к полученным транзакциям
    df["ratio_sent_received"] = df["sent_tnx"] / (df["received_tnx"] + 1e-5)

    # Логарифмирование денежных сумм (борьба с выбросами)
    money_cols = ["total_ether_sent", "total_ether_received", "total_ether_balance"]
    for col in money_cols:
        if col in df.columns:
            min_val = df[col].min()
            shift = abs(min_val) + 1 if min_val < 0 else 1
            df[f"log_{col}"] = np.log(df[col] + shift)

    return df


def split_and_scale(
    df: pd.DataFrame,
    target_col: str = "flag",
    output_dir: str = "../data/processed",
) -> None:
    """Разбиение на train/val/test с последующим шкалированием."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Стратифицированный сплит (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=SEED
    )

    # Шкалирование только по train
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    os.makedirs(output_dir, exist_ok=True)

    pd.concat([y_train.reset_index(drop=True), X_train_scaled], axis=1).to_csv(
        f"{output_dir}/train.csv", index=False
    )
    pd.concat([y_val.reset_index(drop=True), X_val_scaled], axis=1).to_csv(
        f"{output_dir}/val.csv", index=False
    )
    pd.concat([y_test.reset_index(drop=True), X_test_scaled], axis=1).to_csv(
        f"{output_dir}/test.csv", index=False
    )

    print("Данные успешно обработаны и сохранены")


if __name__ == "__main__":
    raw_path = "../data/raw/transaction_dataset.csv"
    if os.path.exists(raw_path):
        df = load_and_clean_data(raw_path)
        df = feature_engineering(df)
        split_and_scale(df)
    else:
        print(f"Файл {raw_path} с исходным датасетом не найден в data/raw/")
