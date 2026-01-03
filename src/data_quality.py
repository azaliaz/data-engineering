import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CRYPTO = os.path.join(BASE_DIR, "data/processed/crypto")
PROCESSED_STOCKS = os.path.join(BASE_DIR, "data/processed/stocks")
OUT_DIR = os.path.join(os.path.dirname("src"), "reports", "tables")
os.makedirs(OUT_DIR, exist_ok=True)


CHECK_DB = True
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://analytics_user:analytics@localhost:5432/analytics_db")


MISSING_THRESHOLD = 0.05
SHORT_SERIES_DAYS = 30

def find_date_column(df):
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "timestamp" in lc or lc == "index":
            return c
    return None

def infer_freq_from_index(idx):
    try:
        s = pd.infer_freq(idx)
        return s
    except Exception:
        return None

def file_level_checks(dirpath):
    rows = []
    files = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
    for p in files:
        name = os.path.basename(p)
        try:
            df = pd.read_csv(p)
        except Exception as e:
            rows.append({
                "file": name,
                "asset": name.replace(".csv",""),
                "error": f"read_error: {e}",
                "rows": np.nan,
                "start_date": None,
                "end_date": None,
                "n_unique_dates": np.nan,
                "dup_dates": np.nan,
                "price_null_rate": np.nan,
                "return_null_rate": np.nan,
                "log_return_null_rate": np.nan,
                "volatility20_null_rate": np.nan,
                "inferred_freq": None,
                "notes": "read_failed"
            })
            continue

        date_col = find_date_column(df)
        if date_col is None:
            date_col = df.columns[0]

        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        df = df.set_index(date_col)
        df = df.sort_index()

        n_rows = len(df)
        n_unique_dates = df.index.nunique()
        start = df.index.min()
        end = df.index.max()

        dup_count = n_rows - n_unique_dates

        price_null = df["price"].isna().mean() if "price" in df.columns else np.nan
        return_null = df["return"].isna().mean() if "return" in df.columns else np.nan
        logreturn_null = df["log_return"].isna().mean() if "log_return" in df.columns else np.nan
        vol20_null = df["volatility_20"].isna().mean() if "volatility_20" in df.columns else np.nan

        freq = infer_freq_from_index(df.index)

        notes = []
        if n_rows == 0:
            notes.append("empty")
        if dup_count > 0:
            notes.append(f"dup_index:{dup_count}")
        if (price_null if not np.isnan(price_null) else 1.0) > MISSING_THRESHOLD:
            notes.append("many_price_nulls")
        if (return_null if not np.isnan(return_null) else 1.0) > MISSING_THRESHOLD:
            notes.append("many_return_nulls")
        if (vol20_null if not np.isnan(vol20_null) else 1.0) > 0.9:
            notes.append("no_volatility")
        if (end is not None) and ((end - start).days < SHORT_SERIES_DAYS):
            notes.append("short_series")

        rows.append({
            "file": name,
            "asset": name.replace(".csv",""),
            "rows": n_rows,
            "start_date": (start.isoformat() if pd.notna(start) else None),
            "end_date": (end.isoformat() if pd.notna(end) else None),
            "n_unique_dates": int(n_unique_dates) if not pd.isna(n_unique_dates) else np.nan,
            "dup_dates": int(dup_count),
            "price_null_rate": float(price_null) if not np.isnan(price_null) else np.nan,
            "return_null_rate": float(return_null) if not np.isnan(return_null) else np.nan,
            "log_return_null_rate": float(logreturn_null) if not np.isnan(logreturn_null) else np.nan,
            "volatility20_null_rate": float(vol20_null) if not np.isnan(vol20_null) else np.nan,
            "inferred_freq": freq,
            "notes": ";".join(notes) if notes else ""
        })

    df_report = pd.DataFrame(rows)
    return df_report

def db_level_checks(database_url):
    engine = create_engine(database_url)
    q_assets = "SELECT DISTINCT asset FROM asset_prices;"
    assets = [r[0] for r in engine.execute(text(q_assets)).fetchall()]

    rows = []
    for asset in assets:
        q = text("""
            SELECT
                COUNT(*) AS n_rows,
                MIN(timestamp) AS start_ts,
                MAX(timestamp) AS end_ts,
                SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS price_nulls,
                SUM(CASE WHEN return IS NULL THEN 1 ELSE 0 END) AS return_nulls,
                SUM(CASE WHEN log_return IS NULL THEN 1 ELSE 0 END) AS log_return_nulls,
                SUM(CASE WHEN volatility_20 IS NULL THEN 1 ELSE 0 END) AS vol_nulls
            FROM asset_prices
            WHERE asset = :asset
        """)
        res = engine.execute(q, {"asset": asset}).fetchone()
        n_rows = res["n_rows"]
        start_ts = res["start_ts"]
        end_ts = res["end_ts"]
        price_nulls = res["price_nulls"]
        return_nulls = res["return_nulls"]
        logreturn_nulls = res["log_return_nulls"]
        vol_nulls = res["vol_nulls"]

        qdup = text("""
            SELECT COUNT(*) FROM (
                SELECT asset, timestamp, COUNT(*) AS c
                FROM asset_prices
                WHERE asset = :asset
                GROUP BY asset, timestamp
                HAVING COUNT(*) > 1
            ) t
        """)
        dup_count = engine.execute(qdup, {"asset": asset}).scalar()

        rows.append({
            "asset": asset,
            "n_rows": int(n_rows),
            "start_ts": start_ts.isoformat() if start_ts is not None else None,
            "end_ts": end_ts.isoformat() if end_ts is not None else None,
            "price_null_rate": price_nulls / n_rows if n_rows else None,
            "return_null_rate": return_nulls / n_rows if n_rows else None,
            "log_return_null_rate": logreturn_nulls / n_rows if n_rows else None,
            "volatility20_null_rate": vol_nulls / n_rows if n_rows else None,
            "dup_count": int(dup_count)
        })
    return pd.DataFrame(rows)


def main():
    print("Running file-level DQ checks...")
    files_crypto = file_level_checks(PROCESSED_CRYPTO)
    files_stocks = file_level_checks(PROCESSED_STOCKS)
    files_df = pd.concat([files_crypto, files_stocks], ignore_index=True)

    files_csv = os.path.join(OUT_DIR, "data_quality_files.csv")
    files_df.to_csv(files_csv, index=False)
    print(f"Saved file-level DQ -> {files_csv}")

    db_df = pd.DataFrame()
    if CHECK_DB:
        try:
            print("Running DB-level DQ checks...")
            db_df = db_level_checks(DATABASE_URL)
            db_csv = os.path.join(OUT_DIR, "data_quality_db.csv")
            db_df.to_csv(db_csv, index=False)
            print(f"Saved DB-level DQ -> {db_csv}")
        except Exception as e:
            print("DB checks failed:", e)
            # CHECK_DB = False

    summary = files_df.copy()
    if not db_df.empty:
        summary = summary.merge(db_df, left_on="asset", right_on="asset", how="left", suffixes=("_file", "_db"))

    summary_csv = os.path.join(OUT_DIR, "data_quality_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved merged summary -> {summary_csv}")

    md_lines = []
    md_lines.append("# Data Quality Report\n")
    md_lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
    md_lines.append("## Key file-level issues\n")
    ff = files_df[files_df['notes'] != ""].sort_values(['notes','rows'])
    if ff.empty:
        md_lines.append("No notable file-level issues found.\n")
    else:
        for _, r in ff.iterrows():
            md_lines.append(f"- **{r['asset']}** ({r['file']}): {r['notes']}; rows={r['rows']}, {r['start_date']} → {r['end_date']}\n")

    if not db_df.empty:
        md_lines.append("\n## Key DB-level issues\n")
        db_issues = db_df[(db_df['dup_count'] > 0) | (db_df['price_null_rate'] > MISSING_THRESHOLD) | (db_df['log_return_null_rate'] > MISSING_THRESHOLD)]
        if db_issues.empty:
            md_lines.append("No notable DB-level issues found.\n")
        else:
            for _, r in db_issues.iterrows():
                md_lines.append(f"- **{r['asset']}**: rows={r['n_rows']}, dup={r['dup_count']}, price_null_rate={r['price_null_rate']:.3f}, log_return_null_rate={r['log_return_null_rate']:.3f}\n")

    md_lines.append("\n## Next recommended actions\n")
    md_lines.append("- Investigate files with `many_price_nulls` or `many_return_nulls` and decide imputation or removal.\n")
    md_lines.append("- Fix duplicates for listed assets (DB-level) or deduplicate CSVs.\n")
    md_lines.append("- Compute volatility_20 if missing (script already computes it by default).\n")

    md_path = os.path.join(OUT_DIR, "data_quality_summary.md")
    with open(md_path, "w", encoding="utf8") as f:
        f.writelines([l + "\n" for l in md_lines])

    print(f"Saved markdown summary -> {md_path}")
    print("Done.")

if __name__ == "__main__":
    main()
