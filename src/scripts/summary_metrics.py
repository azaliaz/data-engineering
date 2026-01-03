import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as st
from pathlib import Path
TRADING_DAYS = 252

def max_drawdown_from_logreturns(logr: pd.Series) -> float:

    if logr.empty:
        return np.nan
    cum = np.exp(logr.cumsum())
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return dd.min()

out = []
paths = glob.glob("data/processed/crypto/*.csv") + glob.glob("data/processed/stocks/*.csv")
for path in glob.glob("data/processed/crypto/*.csv") + glob.glob("data/processed/stocks/*.csv"):
    name = os.path.basename(path).replace('.csv','')
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if 'log_return' not in df.columns:
        continue

    s = df['log_return'].dropna()
    if s.empty:
        continue

    # определить число торговых дней
    if 'crypto' in path:
        trading_days = 365
    else:
        trading_days = 252

    daily_mean = s.mean()
    daily_std = s.std()
    ann_vol = daily_std * (trading_days ** 0.5)
    ann_return = np.exp(daily_mean * trading_days) - 1
    mdd = max_drawdown_from_logreturns(s)
    skewness = float(st.skew(s))

    out.append((name, len(df), ann_return, ann_vol, mdd, skewness))


summary = pd.DataFrame(out, columns=['asset','n_rows','ann_return','ann_vol','max_drawdown','skew'])
out_path = Path("reports/tables/summary_metrics.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
summary.sort_values('ann_vol', ascending=False).to_csv(out_path, index=False)
print("Saved summary_metrics.csv")
