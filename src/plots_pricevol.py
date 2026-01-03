import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYPTO_DIR = os.path.join(BASE_DIR, "data", "processed", "crypto")
STOCK_DIR = os.path.join(BASE_DIR, "data", "processed", "stocks")
OUT_TABLES = os.path.join(BASE_DIR, "reports", "tables")
OUT_FIGS = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)

TRADING_DAYS_STOCK = 252
TRADING_DAYS_CRYPTO = 365
TOP_N = 10
FIG_DPI = 150


def read_df_safe(path):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        df = pd.read_csv(path)
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.set_index(first_col)
    if pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.sort_index()
    return df


def get_price_series(df):
    if 'price' in df.columns:
        return df['price'].astype(float)
    if 'Adj Close' in df.columns:
        return df['Adj Close'].astype(float)
    numeric = df.select_dtypes(include=[np.number]).columns
    return df[numeric[0]].astype(float) if len(numeric) else None


def compute_log_returns(df):
    s = get_price_series(df)
    if s is None:
        return pd.Series(dtype=float)
    prev = s.shift(1).replace(0, np.nan)
    lr = np.log(s / prev)
    return lr.replace([np.inf, -np.inf], np.nan)


# ==============================================================
# 1) Annualized volatility table
# ==============================================================

rows = []
all_paths = []

for p in glob.glob(os.path.join(CRYPTO_DIR, "*.csv")):
    all_paths.append((p, "crypto"))

for p in glob.glob(os.path.join(STOCK_DIR, "*.csv")):
    all_paths.append((p, "stock"))

for path, typ in sorted(all_paths):
    asset = os.path.basename(path).replace(".csv", "")
    df = read_df_safe(path)

    if 'log_return' in df.columns and df['log_return'].notna().sum() > 0:
        s = df['log_return'].dropna().astype(float)
    else:
        s = compute_log_returns(df).dropna()

    if s.empty:
        ann_vol = np.nan
    else:
        days = TRADING_DAYS_CRYPTO if typ == "crypto" else TRADING_DAYS_STOCK
        ann_vol = float(s.std(ddof=1) * math.sqrt(days))

    rows.append({
        "asset": asset,
        "type": typ,
        "ann_vol": ann_vol,
        "n_obs": len(df)
    })

vol_df = pd.DataFrame(rows).sort_values("ann_vol", ascending=False)
vol_csv = os.path.join(OUT_TABLES, "annualized_volatility.csv")
vol_df.to_csv(vol_csv, index=False)
print(f"Saved annualized vol table -> {vol_csv}")

# используем TOP-N, но БЕЗ bar chart
top_df = vol_df.dropna(subset=["ann_vol"]).head(TOP_N)


# ==============================================================
# 2) Price + 20d volatility plots (ТОП-N)
# ==============================================================

def plot_price_and_vol(path, asset_name):
    df = read_df_safe(path)
    price = get_price_series(df)
    if price is None or price.dropna().empty:
        return

    if 'volatility_20' in df.columns and df['volatility_20'].notna().sum() > 0:
        vol20 = df['volatility_20'].astype(float)
    else:
        lr = df['log_return'] if 'log_return' in df.columns else compute_log_returns(df)
        vol20 = lr.rolling(20).std()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.plot(price.index, price.values, label="Price", linewidth=1.5)
    ax1.plot(price.index, price.rolling(20).mean(), "--", label="MA(20)", linewidth=1)
    ax1.set_title(f"{asset_name} — Price & 20D Volatility")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(vol20.index, vol20.values, color="orange", label="Volatility (20d)")
    ax2.set_ylabel("Vol (20d)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out = os.path.join(OUT_FIGS, f"{asset_name}_price_vol.png")
    plt.savefig(out, dpi=FIG_DPI)
    plt.close()
    print(f"Saved {out}")


for _, row in top_df.iterrows():
    asset = row["asset"]
    c = os.path.join(CRYPTO_DIR, asset + ".csv")
    s = os.path.join(STOCK_DIR, asset + ".csv")

    if os.path.exists(c):
        plot_price_and_vol(c, asset)
    elif os.path.exists(s):
        plot_price_and_vol(s, asset)
    else:
        print(f"File for {asset} not found")

print("All done.")
