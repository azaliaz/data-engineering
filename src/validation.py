import os, pandas as pd, numpy as np
CRYPTO = "data/processed/crypto"
STOCKS = "data/processed/stocks"

def sample_info(path, n=5):
    files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
    for f in files[:n]:
        df = pd.read_csv(os.path.join(path,f), index_col=0, parse_dates=True)
        print(f, "rows:", len(df), "start:", df.index.min(), "end:", df.index.max(),
              "na_return:", df['return'].isna().sum() if 'return' in df.columns else 'no_return',
              "vol_col:", [c for c in df.columns if c.startswith("volatility")])

print("Crypto sample:")
sample_info(CRYPTO, n=6)
print("Stocks sample:")
sample_info(STOCKS, n=6)
