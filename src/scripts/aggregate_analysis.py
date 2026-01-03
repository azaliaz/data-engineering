import os
import glob
import pandas as pd
import numpy as np

os.makedirs("reports/tables", exist_ok=True)

def build_returns(dirpath):
    files = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
    series_list = []
    for p in files:
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Skip read error {p}: {e}")
            continue

        if 'price' in df.columns:
            pricecol = 'price'
        elif 'Adj Close' in df.columns:
            pricecol = 'Adj Close'
        else:
            numeric = df.select_dtypes(include=[np.number]).columns
            if len(numeric) == 0:
                print(f"Skip (no numeric cols) {p}")
                continue
            pricecol = numeric[0]

        name = os.path.basename(p).replace('.csv','')
        s = df[pricecol].rename(name).astype(float)

        prev = s.shift(1)
        prev = prev.replace(0, np.nan)
        lr = np.log(s / prev)
        lr = lr.replace([np.inf, -np.inf], np.nan)
        series_list.append(lr)
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1)

crypto_lr = build_returns("data/processed/crypto")
stocks_lr = build_returns("data/processed/stocks")

all_lr = pd.concat([crypto_lr, stocks_lr], axis=1).dropna(how='all')

corr = all_lr.corr()
corr.to_csv("reports/tables/returns_correlation.csv")

vol_crypto = crypto_lr.std(skipna=True) * (365 ** 0.5)
vol_stocks = stocks_lr.std(skipna=True) * (252 ** 0.5)

ann_vol = pd.concat([vol_crypto, vol_stocks]).sort_values(ascending=False)
ann_vol.to_csv("reports/tables/annualized_volatility.csv", header=["annualized_volatility"])

print("Saved correlation and annualized volatility to reports/tables/")
