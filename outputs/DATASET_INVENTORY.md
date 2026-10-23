# Dataset Inventory (uploaded files)

| Dataset | Rows | Columns | Coverage/role |
|---|---:|---:|---|
| silver_daily_ohlcv_2000_2025.csv | 6,783 | 13 | Daily OHLCV and derived returns, 2000–2025 |
| silver_macroeconomic_monthly.csv | 312 | 19 | Monthly macro indicators, 2000–2025 |
| silver_supply_demand_annual.csv | 26 | 20 | Annual supply-demand fundamentals |
| silver_sentiment_weekly.csv | 1,043 | 16 | Weekly positioning, news/social and options sentiment |
| silver_futures_contracts.csv | 27,390 | 11 | Futures term structure by date and contract |

These counts were read directly from the supplied CSVs. Review `data_quality_audit.csv` for engineered-table missingness. The date of observation is not necessarily the date a macro/fundamental value became public; release-date-aware joins are required before production use.
