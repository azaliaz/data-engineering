# Data Quality Report

Generated: 2026-01-02T20:07:07.290478 UTC

## Key file-level issues

- **AAPL** (AAPL.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **ADBE** (ADBE.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **AMZN** (AMZN.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **CRM** (CRM.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **CSCO** (CSCO.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **GOOGL** (GOOGL.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **IBM** (IBM.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **INTC** (INTC.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **META** (META.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **MSFT** (MSFT.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **NFLX** (NFLX.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **NVDA** (NVDA.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **ORCL** (ORCL.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **PYPL** (PYPL.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **TSLA** (TSLA.csv): many_price_nulls; rows=999, 2022-01-03T00:00:00+00:00 → 2025-12-24T00:00:00+00:00

- **midnight-3_usd** (midnight-3_usd.csv): many_return_nulls;no_volatility;short_series; rows=18, 2025-12-09T00:00:00+00:00 → 2025-12-26T00:00:00+00:00


## Next recommended actions

- Investigate files with `many_price_nulls` or `many_return_nulls` and decide imputation or removal.

- Fix duplicates for listed assets (DB-level) or deduplicate CSVs.

- Compute volatility_20 if missing (script already computes it by default).

