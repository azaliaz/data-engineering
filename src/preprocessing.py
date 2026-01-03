"""
preprocessing.py

Более надёжная предобработка:
- обрабатывает CSV криптовалют (ожидает колонку "timestamp" и "price")
- обрабатывает CSV акций (устойчиво к метаданным в первых строках от yfinance)
- считает return, log_return, volatility_20
- сохраняет результаты в data/processed/crypto и data/processed/stocks
"""

import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

CRYPTO_DIR = "data/raw/crypto"
STOCK_DIR = "data/raw/stocks"

PROCESSED_DIR = "data/processed"
CRYPTO_PROC_DIR = os.path.join(PROCESSED_DIR, "crypto")
STOCK_PROC_DIR = os.path.join(PROCESSED_DIR, "stocks")
os.makedirs(CRYPTO_PROC_DIR, exist_ok=True)
os.makedirs(STOCK_PROC_DIR, exist_ok=True)



def safe_to_numeric(series):
    """Безопасно конвертирует в числа, заменяет некорректные на NaN."""
    return pd.to_numeric(series, errors="coerce")


def compute_returns_and_volatility(df, price_col, vol_window=20):
    """
    Добавляет в df колонки:
      - return  (простая дневная доходность)
      - log_return
      - volatility_<vol_window> (скользящая std лог-доходностей)
    Возвращает DataFrame (копия).
    """
    df = df.copy()
    # Простая доходность
    df["return"] = df[price_col].pct_change()
    # Лог-доходность
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    # Скользящая волатильность (стд лог-доходностей)
    df[f"volatility_{vol_window}"] = df["log_return"].rolling(window=vol_window, min_periods=1).std()
    return df



def preprocess_crypto(file_path):
    """
    Ожидается CSV с колонками ['timestamp','price'] или индекс timestamp.
    Возвращает обработанный DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.exception(f"Error reading crypto file {file_path}: {e}")
        return None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    elif "ts_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True, errors="coerce")
        df = df.set_index("timestamp")
    else:
        first_col = df.columns[0]
        try:
            df[first_col] = pd.to_datetime(df[first_col], utc=True, errors="coerce")
            df = df.set_index(first_col)
        except Exception:
            logging.debug(f"Cannot parse timestamp column automatically for {file_path}")

    price_col = None
    for c in df.columns:
        if c.lower() in ("price", "close", "adj close", "adj_close"):
            price_col = c
            break
    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            price_col = numeric_cols[0]
    if price_col is None:
        logging.warning(f"No price column detected in {file_path}; skipping.")
        return None

    df[price_col] = safe_to_numeric(df[price_col])
    df = df.sort_index()
    df = df.dropna(subset=[price_col])
    if df.empty:
        logging.warning(f"No valid rows after dropping NaNs in {file_path}")
        return None

    # переименуем колонку цены на единое имя 'price'
    df = df.rename(columns={price_col: "price"})

    # рассчёт доходностей и волатильности
    df = compute_returns_and_volatility(df, price_col="price", vol_window=20)

    return df


def preprocess_stock(file_path):
    """
    Корректно читает разные варианты CSV от yfinance:
    - если есть колонка 'Date' -> read normally
    - если файл содержит 2–3 строки метаданных (Ticker, header, Date line) -> пропускаем и читаем
    Затем рассчитываем return, log_return и volatility_20 на основе 'Adj Close'.
    """
    df = None
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    except Exception:
        candidate_names = [
            ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
            ["Date", "Price", "Adj Close", "Close", "High", "Low", "Open", "Volume"],
        ]
        for skip in (1, 2, 3, 4):
            for names in candidate_names:
                try:
                    df_try = pd.read_csv(file_path, skiprows=skip, names=names, parse_dates=["Date"], index_col="Date")
                    # проверим, есть ли в df_try числовые колонки и не все NaN
                    if df_try.shape[0] > 0 and df_try.select_dtypes(include=[np.number]).shape[1] >= 1:
                        df = df_try
                        logging.debug(f"Read stock {file_path} with skiprows={skip} and names={names}")
                        break
                except Exception:
                    continue
            if df is not None:
                break

    if df is None:
        logging.warning(f"Failed to parse stock CSV {file_path}")
        return None

    # нормализуем названия колонок (возможно есть лишние пробелы)
    df.columns = [c.strip() for c in df.columns]

    # Ищем 'Adj Close' или 'Adj Close' под разными именами
    adj_candidates = [c for c in df.columns if c.lower().replace(" ", "") in ("adjclose", "adj_close", "adjclose*")]
    adj_col = adj_candidates[0] if adj_candidates else None

    # Если нет Adj Close — попытаемся взять Close или Price
    if adj_col is None:
        close_candidates = [c for c in df.columns if c.lower() in ("close", "price", "last")]
        if close_candidates:
            adj_col = close_candidates[0]

    if adj_col is None:
        logging.warning(f"No suitable price column (Adj Close/Close) in {file_path}; skipping.")
        return None

    # Приводим к числу и удаляем NaN
    df[adj_col] = safe_to_numeric(df[adj_col])
    df = df.sort_index()
    df = df.dropna(subset=[adj_col])
    if df.empty:
        logging.warning(f"No valid rows after dropping NaNs in {file_path}")
        return None

    # Переименовать колонку в 'Adj Close' для единообразия
    df = df.rename(columns={adj_col: "Adj Close"})

    # Рассчитать доходности и волатильность на основе 'Adj Close'
    df = compute_returns_and_volatility(df, price_col="Adj Close", vol_window=20)

    return df


def main():
    # Crypto
    crypto_files = [f for f in os.listdir(CRYPTO_DIR) if f.endswith(".csv") and not f.startswith("top_")]
    logging.info(f"Found {len(crypto_files)} crypto files to process in {CRYPTO_DIR}")
    for fname in sorted(crypto_files):
        path = os.path.join(CRYPTO_DIR, fname)
        try:
            df = preprocess_crypto(path)
            if df is None:
                logging.info(f"Skipped crypto file (no data): {fname}")
                continue
            out_path = os.path.join(CRYPTO_PROC_DIR, fname)
            df.to_csv(out_path)
            logging.info(f"Processed crypto: {fname}, rows: {len(df)}")
        except Exception as e:
            logging.exception(f"Error processing crypto file {fname}: {e}")

    stock_files = [f for f in os.listdir(STOCK_DIR) if f.endswith(".csv")]
    logging.info(f"Found {len(stock_files)} stock files to process in {STOCK_DIR}")
    for fname in sorted(stock_files):
        path = os.path.join(STOCK_DIR, fname)
        try:
            df = preprocess_stock(path)
            if df is None:
                logging.info(f"Skipped stock file (no data/parse fail): {fname}")
                continue
            out_path = os.path.join(STOCK_PROC_DIR, fname)
            df.to_csv(out_path)
            logging.info(f"Processed stock: {fname}, rows: {len(df)}")
        except Exception as e:
            logging.exception(f"Error processing stock file {fname}: {e}")

    logging.info("Preprocessing finished!")


if __name__ == "__main__":
    main()
