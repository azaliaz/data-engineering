import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


DATA_FOLDERS = ["data/processed/stocks", "data/processed/crypto"]

FIG_DIR = Path("reports/figures_plot_single")
FIG_DIR.mkdir(parents=True, exist_ok=True)


assets = []
for folder in DATA_FOLDERS:
    folder_path = Path(folder)
    if folder_path.exists():
        assets += list(folder_path.glob("*.csv"))

if not assets:
    raise SystemExit("Нет файлов CSV в data/processed/stocks и data/processed/crypto")


def plot_asset(df, asset_name):
    plt.figure(figsize=(12,5))
    price_col = df['price'] if 'price' in df.columns else df.get('Adj Close')
    plt.plot(df.index, price_col, label='Price', color='blue', linewidth=2)
    plt.title(f"{asset_name} Price", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{asset_name}_price.png")
    plt.close()

    plt.figure(figsize=(12,5))
    if 'volatility_20' in df.columns:
        vcol = 'volatility_20'
    else:
        vcol = next((c for c in df.columns if c.startswith("volatility")), None)
        if vcol is None:
            print(f"Нет колонки волатильности для {asset_name}, пропускаем")
            return
    plt.plot(df.index, df[vcol], label=vcol, color='orange', linewidth=2)
    plt.title(f"{asset_name} Volatility", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Volatility", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{asset_name}_vol.png")
    plt.close()



for asset_file in assets:
    asset_name = asset_file.stem.replace("_usd", "")  # для крипты убираем _usd
    try:
        df = pd.read_csv(asset_file, index_col=0, parse_dates=True)
        plot_asset(df, asset_name)
        print(f"Сгенерированы графики для {asset_name}")
    except Exception as e:
        print(f"Ошибка при обработке {asset_name}: {e}")

print(f"Все графики сохранены в {FIG_DIR}")
