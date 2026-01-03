import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[1]
TABLES_DIR = SRC / "reports" / "tables_top10"
FIGURES_DIR = SRC / "reports" / "figures_top10"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = SRC / "reports" / "tables" / "summary_metrics.csv"

if not SUMMARY_PATH.exists():
    raise SystemExit(f"Файл не найден: {SUMMARY_PATH}")

summary = pd.read_csv(SUMMARY_PATH)

cols_lower = {c.lower(): c for c in summary.columns}
required = ("asset", "ann_return", "ann_vol")
missing = [c for c in required if c not in cols_lower]
if missing:
    raise SystemExit(f"Отсутствуют колонки: {missing}")

asset_col = cols_lower["asset"]
ann_return_col = cols_lower["ann_return"]
ann_vol_col = cols_lower["ann_vol"]


top_return = summary.sort_values(ann_return_col, ascending=False).head(10)
top_vol = summary.sort_values(ann_vol_col, ascending=False).head(10)

top_return.to_csv(TABLES_DIR / "top10_return.csv", index=False)
top_vol.to_csv(TABLES_DIR / "top10_volatility.csv", index=False)

def plot_top(
    df,
    value_col,
    title,
    filename,
    color="skyblue",
    log_scale=False
):
    assets = df[asset_col].astype(str)
    values = df[value_col]

    plt.figure(figsize=(10, 6))
    plt.barh(assets, values, color=color)
    plt.gca().invert_yaxis()

    if log_scale:
        if (values <= 0).any():
            plt.close()
            raise ValueError(f"log-scale невозможен: {value_col} содержит <= 0")
        plt.xscale("log")
        plt.xlabel(f"{value_col.replace('_',' ').title()} (log scale)")
    else:
        plt.xlabel(value_col.replace('_',' ').title())

    plt.title(title)

    for i, v in enumerate(values):
        plt.text(
            v,
            i,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=9
        )

    plt.tight_layout()
    outpath = FIGURES_DIR / filename
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

p1 = plot_top(
    top_return,
    ann_return_col,
    "Top 10 Assets by Annualized Return",
    "top10_return.png",
    color="green",
    log_scale=True
)

p2 = plot_top(
    top_vol,
    ann_vol_col,
    "Top 10 Assets by Annualized Volatility",
    "top10_volatility.png",
    color="red",
    log_scale=False
)

print(f"\nSaved CSVs to: {TABLES_DIR}")
print(f"Saved figures to: {FIGURES_DIR}")
print(f"Figures: {p1}, {p2}")
