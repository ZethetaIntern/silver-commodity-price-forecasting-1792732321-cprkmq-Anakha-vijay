# 15-Day Delivery Plan

| Day | Deliverable | Acceptance check |
|---|---|---|
| 1 | Data inventory and provenance | Five CSVs profiled; row counts, dates, types and missingness documented |
| 2 | Cleaning and alignment | Duplicates, date ordering, gaps, units and missing values reviewed |
| 3 | Technical and lag features | SMA/EMA, RSI, MACD, ATR, volatility and lagged returns generated |
| 4 | Macro, futures, sentiment and fundamentals | Features joined by known observation date; release-lag assumptions documented |
| 5 | Stationarity and exploratory analysis | ADF/KPSS, ACF/PACF, regime/volatility review |
| 6 | ARIMA/SARIMA | Baselines and residual diagnostics |
| 7 | Prophet | Trend/changepoints/seasonality, only where installed and valid future regressors exist |
| 8 | XGBoost | Time-ordered validation, feature importance, parameter tuning |
| 9 | LSTM | Sequence windows, train-only scalers, early stopping, repeatable seeds |
| 10 | Ensemble layer | Inverse-validation-error weighting, Ridge stacking, regime switching |
| 11 | Backtesting and risk | Rolling-origin metrics, naive baseline, drawdown and cost-aware P&L |
| 12 | Power BI semantic model | Date dimension, relationships, DAX validation and report pages |
| 13 | Trading Arena | Virtual positions, scenario cards, fees/slippage, scoring and leaderboards |
| 14 | QA and stress tests | Flash crash, crisis, missing data, stale forecasts and outlier checks |
| 15 | Final package and presentation | Reproducible setup, model cards, dashboard screenshots, limitations and demo |

## Arena game design
- Roles: Analyst, Risk Manager, Portfolio Manager, Market Surveillance.
- Rounds: normal market, crisis sell-off, flash crash/rebound, supply shock, sentiment squeeze, manipulation attempt.
- Score: risk-adjusted net P&L, drawdown discipline, position sizing, forecast calibration, and anomaly detection.
- Penalties: excessive exposure, ignored stop/risk limits, unexplained outlier response, and overconfidence in a single model.
- Integrity: simulated prices only; seeded scenarios; no broker API; explicit distinction between suspicious activity and proven manipulation.
