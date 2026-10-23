# Silver Intelligence: Forecasting + Power BI + Trading Arena

An educational end-to-end starter implementation based on the supplied silver market datasets and project brief.

## Contents
- `src/data_pipeline.py`: merges the five datasets and creates technical, macro, futures, sentiment, and lag features.
- `src/train_models.py`: chronological holdout benchmarks for persistence, XGBoost (or sklearn fallback), ARIMA, SARIMA, Prophet, and LSTM when dependencies are available.
- `app.py`: Streamlit dashboard with market overview, model metrics, portfolio risk sandbox, and gamified scenario round.
- `powerbi/Silver_DAX_Measures.dax`: starting DAX measures for Power BI.
- `data/`: supplied CSV datasets copied into the project.
- `outputs/`: generated features, audit, forecasts, metrics, and model artifacts.

## Quick start
Use Python 3.10 or 3.11 in a virtual environment. TensorFlow and Prophet can require platform-specific setup; they are optional at runtime and are skipped if unavailable.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
python src/data_pipeline.py
python src/train_models.py
streamlit run app.py
```

If TensorFlow installation is not suitable for your machine, remove it from the install command; the other models still run. For a quick dashboard preview before training, run the app and it will tell you which outputs are missing.

## Power BI setup
1. Import `outputs/silver_features.csv`, `outputs/forecast_predictions.csv`, and the supplied source tables.
2. Create a dedicated Date table spanning the daily series; mark it as the Date table.
3. Relate Date[Date] to SilverDaily[Date], and create appropriate one-to-many relationships to forecast, macro, and portfolio tables.
4. Add measures from `powerbi/Silver_DAX_Measures.dax`. Define `[Selected Forecast Price]` using the selected model and date context.
5. Suggested report pages: Executive Overview, Forecast Comparison, Risk & Drawdown, Futures Curve, Macro & Sentiment, Supply/Demand, Portfolio Monitor, Trading Arena Scorecard.

## Modeling and data caveats
- Inputs are aligned on observation dates and forward-filled. For a production-grade backtest, use release/availability timestamps for macro and fundamentals; a calendar date alone may not represent when the information was public.
- Annual supply/demand values are mapped to calendar year for this starter. Replace with publication-date-aware availability to avoid look-ahead bias.
- The starter's forecast benchmark is a holdout evaluation. A production version should add rolling-origin validation, tuning on validation folds only, forecast intervals, transaction costs, slippage, and model drift monitoring.
- Forecasts and DAX signals are not investment advice. The arena is virtual and educational; no brokerage or live order execution is connected.
- Check data provenance, units, missingness, duplicate dates, and whether volume represents comparable contracts before using results externally.
