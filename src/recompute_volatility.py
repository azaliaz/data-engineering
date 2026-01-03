import os
import pandas as pd
import numpy as np
from glob import glob

PROCESSED_CRYPTO = "data/processed/crypto"
PROCESSED_STOCKS = "data/processed/stocks"

VOL_WINDOW = 20
MIN_PERIODS = 20
DDOF = 1

def recompute_vol(df, price_col, vol_window=VOL_WINDOW, min_periods=MIN_PERIODS, ddof=DDOF):
    df = df.copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df[f"volatility_{vol_window}"] = df["log_return"].rolling(window=vol_window, min_periods=min_periods).std(ddof=ddof)
    return df

def process_dir(dir_path, price_col_candidates):
    files = sorted(glob(os.path.join(dir_path, "*.csv")))
    for p in files:
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            price_col = None
            for cand in price_col_candidates:
                if cand in df.columns:
                    price_col = cand
                    break
            if price_col is None:
                numeric = df.select_dtypes(include=[np.number]).columns
                if len(numeric) == 0:
                    print(f"Skip (no numeric cols): {p}")
                    continue
                price_col = numeric[0]
            df = recompute_vol(df, price_col)
            df.to_csv(p)
            print(f"Recomputed volatility: {os.path.basename(p)} rows:{len(df)}")
        except Exception as e:
            print(f"Error for {p}: {e}")

process_dir(PROCESSED_CRYPTO, ["price", "Price", "Adj Close", "Close"])
process_dir(PROCESSED_STOCKS, ["Adj Close", "Adj_Close", "Close", "price"])
