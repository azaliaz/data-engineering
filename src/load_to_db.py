import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://analytics_user:analytics@localhost:5432/analytics_db"
)

engine = create_engine(DATABASE_URL)

DATASETS = [
    (os.path.join(BASE_DIR, "data/processed/crypto"), "crypto"),
    (os.path.join(BASE_DIR, "data/processed/stocks"), "stock"),
]


def load_csv(path, asset_type):
    asset = os.path.basename(path).replace(".csv", "")
    # читаем CSV без parse_dates, чтобы сначала узнать названия колонок
    df = pd.read_csv(path)

    # ищем колонку с датой
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "timestamp" in c.lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date/timestamp column found in {path}")

    # теперь преобразуем найденный столбец в datetime
    df[date_col] = pd.to_datetime(df[date_col])

    df["asset"] = asset
    df["asset_type"] = asset_type

    # Переименовываем колонку даты в "timestamp", чтобы соответствовать БД
    df = df.rename(columns={date_col: "timestamp"})

    # Если волатильность отсутствует, вычисляем 20-дневную скользящую std
    if "volatility_20" not in df.columns or df["volatility_20"].isnull().all():
        if "log_return" in df.columns:
            df["volatility_20"] = df["log_return"].rolling(20).std()
        else:
            df["volatility_20"] = None

    # Обязательные поля
    for c in ["price", "return", "log_return"]:
        if c not in df.columns:
            df[c] = None

    df = df[["asset", "asset_type", "timestamp", "price", "return", "log_return", "volatility_20"]]

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO asset_prices
                    (asset, asset_type, timestamp, price, return, log_return, volatility_20)
                    VALUES
                    (:asset, :asset_type, :timestamp, :price, :return, :log_return, :volatility_20)
                    ON CONFLICT (asset, timestamp) DO UPDATE SET
                        price = EXCLUDED.price,
                        return = EXCLUDED.return,
                        log_return = EXCLUDED.log_return,
                        volatility_20 = EXCLUDED.volatility_20;
                    """
                ),
                row.to_dict(),
            )

    print(f"Loaded {asset} ({asset_type}), rows={len(df)}")


def main():
    for base, asset_type in DATASETS:
        for path in glob.glob(os.path.join(base, "*.csv")):
            load_csv(path, asset_type)

if __name__ == "__main__":
    main()
