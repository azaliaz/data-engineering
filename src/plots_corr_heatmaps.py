import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYPTO_DIR = os.path.join(BASE_DIR, "data", "processed", "crypto")
STOCK_DIR = os.path.join(BASE_DIR, "data", "processed", "stocks")
OUT_FIGS = os.path.join(BASE_DIR, "reports", "figures_cor")
OUT_TABLES = os.path.join(BASE_DIR, "reports", "tables_cor")
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)

TOP_N = 30
MIN_COMMON_OBS = 30
FIG_DPI = 200
ANNOTATE_TOP = False
USE_MASK_TRIANGLE = False

sns.set_theme(style="white")


def read_price_series(path):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        df = pd.read_csv(path)
        date_col = None
        for c in df.columns:
            if "date" in c.lower() or "timestamp" in c.lower():
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    if 'price' in df.columns:
        s = df['price'].astype(float)
    elif 'Adj Close' in df.columns:
        s = df['Adj Close'].astype(float)
    else:
        numeric = df.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            return pd.Series(dtype=float)
        s = df[numeric[0]].astype(float)
    s = s.sort_index()
    return s

def compute_log_return_series_from_price(s):
    prev = s.shift(1).replace(0, np.nan)
    lr = np.log(s / prev)
    lr = lr.replace([np.inf, -np.inf], np.nan)
    return lr

def collect_returns(min_obs=MIN_COMMON_OBS):
    series_list = {}
    paths = sorted(glob.glob(os.path.join(CRYPTO_DIR, "*.csv"))) + sorted(glob.glob(os.path.join(STOCK_DIR, "*.csv")))
    for p in paths:
        name = os.path.basename(p).replace('.csv','')
        s_price = read_price_series(p)
        if s_price.empty:
            continue
        # попробовать взять готовые log_return
        try:
            df_try = pd.read_csv(p, index_col=0, parse_dates=True)
            if 'log_return' in df_try.columns and df_try['log_return'].notna().sum() > 0:
                lr = df_try['log_return'].astype(float)
                lr.index = pd.to_datetime(lr.index, errors='coerce')
                lr = lr.sort_index()
            else:
                lr = compute_log_return_series_from_price(s_price)
        except Exception:
            lr = compute_log_return_series_from_price(s_price)

        if lr.dropna().shape[0] >= min_obs:
            series_list[name] = lr
    if not series_list:
        raise SystemExit("No return series found.")
    all_lr = pd.concat(series_list.values(), axis=1)
    all_lr.columns = list(series_list.keys())
    return all_lr


def adaptive_figsize(n):
    per = 0.4
    w = max(12, min(80, n * per + 6))
    h = w
    return (w, h)

def thin_tick_labels(ax, max_labels=40):
    labels = ax.get_xticklabels()
    n = len(labels)
    if n == 0:
        return
    step = max(1, int(math.ceil(n / max_labels)))
    for i, lbl in enumerate(labels):
        lbl.set_visible((i % step) == 0)
    labels_y = ax.get_yticklabels()
    for i, lbl in enumerate(labels_y):
        lbl.set_visible((i % step) == 0)

def plot_heatmap_matrix(mat, outpath, title="", annotate=True, mask_triangle=False):
    n = mat.shape[0]
    figsize = adaptive_figsize(n)
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    vmin, vmax = -1.0, 1.0
    mask = np.triu(np.ones_like(mat, dtype=bool)) if mask_triangle else None
    ax = sns.heatmap(
        mat,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=False,
        linewidths=0.3,
        linecolor="white",
        annot=annotate,
        fmt=".2f" if annotate else "",
        cbar_kws={'label': 'Correlation', 'shrink': 0.6},
        mask=mask
    )
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    thin_tick_labels(ax, max_labels=40 if n>50 else 80)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("Saved", outpath)

print("Collect returns...")
all_lr = collect_returns(min_obs=MIN_COMMON_OBS)
print(f"Collected {all_lr.shape[1]} series, range {all_lr.index.min()} -> {all_lr.index.max()}")


print("Computing full correlation...")
corr_full = all_lr.corr()
corr_full.to_csv(os.path.join(OUT_TABLES, "returns_correlation_full.csv"))
print("Saved returns_correlation_full.csv")

full_out = os.path.join(OUT_FIGS, "corr_full.png")
plot_heatmap_matrix(
    corr_full,
    full_out,
    title="Корреляции лог-доходностей (полная матрица)",
    annotate=False,
    mask_triangle=USE_MASK_TRIANGLE
)

print("Compute annualized volatility for selection...")
ann_vol = {}
for name in all_lr.columns:
    s = all_lr[name].dropna()
    trading_days = 365 if ('usd' in name.lower() or 'coin' in name.lower() or name.islower()) else 252
    ann_vol[name] = s.std(ddof=1) * math.sqrt(trading_days)
ann_series = pd.Series(ann_vol).dropna().sort_values(ascending=False)
ann_series.to_csv(os.path.join(OUT_TABLES, "annualized_volatility_from_returns.csv"), header=["ann_vol"])
print("Saved annualized_volatility_from_returns.csv")

top_names = ann_series.head(TOP_N).index.tolist()
print("Top assets selected:", len(top_names))

corr_top = corr_full.loc[top_names, top_names]
top_out = os.path.join(OUT_FIGS, f"corr_top{TOP_N}.png")
plot_heatmap_matrix(
    corr_top,
    top_out,
    title=f"Корреляции (топ-{TOP_N} по годовой волатильности)",
    annotate=ANNOTATE_TOP,
    mask_triangle=False
)

print("Done.")
