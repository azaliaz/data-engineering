"""
collect_data.py

Собирает данные:
- для криптовалют через CoinGecko API (no API key) — автоматически берет top N по market_cap
- для акций через yfinance (Yahoo Finance)

Сохраняет CSV в data/raw/{type}/
"""

import os
import time
from datetime import datetime, timedelta
import logging
from datetime import datetime, timezone
import math
import requests
import pandas as pd

# Для акций
try:
    import yfinance as yf
except Exception:
    yf = None

# ========== НАСТРОЙКИ ==========
DATA_DIR = "data/raw"
CRYPTO_DIR = os.path.join(DATA_DIR, "crypto")
STOCK_DIR = os.path.join(DATA_DIR, "stocks")
os.makedirs(CRYPTO_DIR, exist_ok=True)
os.makedirs(STOCK_DIR, exist_ok=True)

COINGECKO_API = "https://api.coingecko.com/api/v3"


CONFIG = {
    "crypto": {
        "coin_ids": [],
        "top_n": 100,
        "vs_currency": "usd",
        "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "rate_limit_sleep": 15,
        "page_sleep": 1.0
    },
    "stocks": {
        "tickers": [
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "GOOGL",  # Alphabet (Google)
            "AMZN",  # Amazon
            "TSLA",  # Tesla
            "META",  # Meta (Facebook)
            "NVDA",  # NVIDIA
            "NFLX",  # Netflix
            "INTC",  # Intel
            "CSCO",  # Cisco
            "ADBE",  # Adobe
            "PYPL",  # PayPal
            "ORCL",  # Oracle
            "CRM",  # Salesforce
            "IBM"  # IBM
        ],
        "start_date": "2022-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
    },
    "download": {
        "skip_existing": True,
        "force_download": False
    }
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")



def to_unix(ts: str) -> int:
    """Преобразовать YYYY-MM-DD в unix (seconds)."""
    dt = datetime.fromisoformat(ts)
    return int(dt.replace(tzinfo=timezone.utc).timestamp())



def get_top_coin_ids(n: int, vs_currency: str = "usd", per_page: int = 250, page_sleep: float = 1.0):
    """
    Возвращает список coin_id (строк) топ n по рыночной капитализации.
    per_page: максимальное значение CoinGecko позволяет до 250.
    """
    logging.info(f"Fetching top {n} coin ids from CoinGecko (vs_currency={vs_currency})")
    coin_ids = []
    page = 1
    while len(coin_ids) < n:
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": min(per_page, 250),
            "page": page,
            "sparkline": "false"
        }
        url = f"{COINGECKO_API}/coins/markets"
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                logging.error(f"Failed to fetch markets page {page}: {r.status_code} {r.text}")
                break
            data = r.json()
            if not data:
                logging.info("No more data from CoinGecko markets endpoint.")
                break
            for item in data:
                coin_ids.append(item.get("id"))
                if len(coin_ids) >= n:
                    break
            logging.info(f"Collected {len(coin_ids)} ids so far (page {page})")
            page += 1
            time.sleep(page_sleep)
        except Exception as e:
            logging.exception(f"Exception while fetching top coins page {page}: {e}")
            break
    coin_ids = [c for c in coin_ids if c][:n]
    top_csv = os.path.join(CRYPTO_DIR, f"top_{len(coin_ids)}_coin_ids.csv")
    pd.DataFrame({"coin_id": coin_ids}).to_csv(top_csv, index=False)
    logging.info(f"Saved top coin ids to {top_csv}")
    return coin_ids


def get_coin_market_chart_range(coin_id: str, vs_currency: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками: timestamp (UTC), price
    Использует /coins/{id}/market_chart/range
    """
    start_unix = to_unix(start_date)
    end_dt = datetime.fromisoformat(end_date)
    end_dt = end_dt.replace(hour=23, minute=59, second=59)
    end_unix = int(end_dt.replace(tzinfo=timezone.utc).timestamp())

    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart/range"
    params = {"vs_currency": vs_currency, "from": start_unix, "to": end_unix}
    logging.info(f"CoinGecko request for {coin_id}: {params}")
    r = requests.get(url, params=params, timeout=60)

    if r.status_code != 200:
        logging.error(f"CoinGecko API error {r.status_code} for {coin_id}: {r.text}")
        return pd.DataFrame()

    j = r.json()
    prices = j.get("prices", [])
    if not prices:
        logging.warning(f"No price data returned for {coin_id}")
        return pd.DataFrame()

    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df[["timestamp", "price"]]
    df = df.set_index("timestamp").sort_index()
    return df


def save_crypto_csv(df: pd.DataFrame, coin_id: str, vs_currency: str, out_dir=CRYPTO_DIR):
    fname = f"{coin_id}_{vs_currency}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=True)
    logging.info(f"Saved crypto CSV: {path}")
    return path


def get_stock_history_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Возвращает DataFrame с историей (Open, High, Low, Close, Adj Close, Volume)
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")
    logging.info(f"Downloading stock data for {ticker} {start_date}..{end_date} via yfinance")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df.empty:
        logging.warning(f"No data for {ticker}")
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def save_stock_csv(df: pd.DataFrame, ticker: str, out_dir=STOCK_DIR):
    fname = f"{ticker}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=True)
    logging.info(f"Saved stock CSV: {path}")
    return path



def collect_all(config=CONFIG):
    crypto_cfg = config.get("crypto", {})
    coin_ids = crypto_cfg.get("coin_ids") or []
    top_n = crypto_cfg.get("top_n", 0)
    vs_currency = crypto_cfg.get("vs_currency", "usd")
    start_date = crypto_cfg.get("start_date")
    end_date = crypto_cfg.get("end_date")
    sleep = float(crypto_cfg.get("rate_limit_sleep", 1.2))
    page_sleep = float(crypto_cfg.get("page_sleep", 1.0))
    skip_existing = config.get("download", {}).get("skip_existing", True)
    force_download = config.get("download", {}).get("force_download", False)

    if not coin_ids and top_n and top_n > 0:
        coin_ids = get_top_coin_ids(top_n, vs_currency=vs_currency, page_sleep=page_sleep)
        logging.info(f"Using top {len(coin_ids)} coins for download.")

    logging.info(f"Starting crypto download for {len(coin_ids)} coins.")
    for idx, coin in enumerate(coin_ids, start=1):
        try:
            out_fname = os.path.join(CRYPTO_DIR, f"{coin}_{vs_currency}.csv")
            if skip_existing and os.path.exists(out_fname) and not force_download:
                logging.info(f"[{idx}/{len(coin_ids)}] Skipping {coin} (file exists): {out_fname}")
                continue

            df = get_coin_market_chart_range(coin, vs_currency, start_date, end_date)
            if df.empty:
                logging.warning(f"[{idx}/{len(coin_ids)}] No data for {coin}. Skipping.")
            else:
                save_crypto_csv(df, coin, vs_currency)
            time.sleep(sleep)
        except Exception as e:
            logging.exception(f"[{idx}/{len(coin_ids)}] Failed to fetch crypto {coin}: {e}")

    stock_cfg = config.get("stocks", {})
    tickers = stock_cfg.get("tickers", [])
    s_start = stock_cfg.get("start_date")
    s_end = stock_cfg.get("end_date")

    logging.info(f"Starting stock download for {len(tickers)} tickers.")
    for ticker in tickers:
        try:
            out_fname = os.path.join(STOCK_DIR, f"{ticker}.csv")
            if skip_existing and os.path.exists(out_fname) and not force_download:
                logging.info(f"Skipping {ticker} (file exists): {out_fname}")
                continue

            df = get_stock_history_yfinance(ticker, s_start, s_end)
            if not df.empty:
                save_stock_csv(df, ticker)
        except Exception as e:
            logging.exception(f"Failed to fetch stock {ticker}: {e}")


if __name__ == "__main__":
    logging.info("Starting data collection")
    collect_all()
    logging.info("Data collection finished")
