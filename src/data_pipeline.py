"""Data preparation for silver commodity forecasting. Uses only information available by each date."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "outputs"
FILES = {
    "daily": "silver_daily_ohlcv_2000_2025.csv",
    "macro": "silver_macroeconomic_monthly.csv",
    "supply": "silver_supply_demand_annual.csv",
    "sentiment": "silver_sentiment_weekly.csv",
    "futures": "silver_futures_contracts.csv",
}

def load_and_engineer(data_dir=DATA):
    daily = pd.read_csv(data_dir / FILES['daily'], parse_dates=['Date']).sort_values('Date').drop_duplicates('Date').set_index('Date')
    macro = pd.read_csv(data_dir / FILES['macro'], parse_dates=['Date']).sort_values('Date').drop_duplicates('Date').set_index('Date')
    sentiment = pd.read_csv(data_dir / FILES['sentiment'], parse_dates=['Week_Ending']).sort_values('Week_Ending').drop_duplicates('Week_Ending').set_index('Week_Ending')
    supply = pd.read_csv(data_dir / FILES['supply'])
    futures = pd.read_csv(data_dir / FILES['futures'], parse_dates=['Trade_Date']).sort_values(['Trade_Date','Months_To_Expiry'])

    df = daily.copy()
    # Monthly macro values are only carried forward after their recorded date.
    macro_cols = [c for c in macro.columns if c not in ['Silver_Price_USD']]
    df = df.join(macro[macro_cols].reindex(df.index, method='ffill'))
    # Weekly sentiment is available only from its observation date onward.
    df = df.join(sentiment.drop(columns=['Silver_Price_Close'], errors='ignore').reindex(df.index, method='ffill'))
    # Annual fundamentals: map each observation to its year, available only from that year onward.
    supply = supply.set_index('Year')
    supply_cols = list(supply.columns)
    years = pd.DataFrame({'Year': df.index.year}, index=df.index)
    for col in supply_cols:
        years[col] = years['Year'].map(supply[col])
    df = df.join(years.drop(columns='Year'))
    # Futures term structure summary by date; forward fill only after the date exists.
    fut = futures.groupby('Trade_Date').apply(lambda g: pd.Series({
        'Futures_1M': g.loc[g['Months_To_Expiry'].idxmin(), 'Futures_Price'],
        'Futures_12M': g.iloc[(g['Months_To_Expiry']-12).abs().argsort()[:1]]['Futures_Price'].iloc[0],
        'Futures_Basis_Mean': g['Basis'].mean(),
        'Futures_OI_Total': g['Open_Interest'].sum(),
        'Futures_Volume_Total': g['Volume'].sum(),
        'Futures_Carry_Mean': g['Implied_Carry_Rate_Annual'].mean(),
    }), include_groups=False)
    df = df.join(fut.reindex(df.index, method='ffill'))

    c = df['Close']
    df['Return_1D'] = c.pct_change()
    df['Log_Return_1D'] = np.log(c).diff()
    for w in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{w}'] = c.rolling(w).mean()
        df[f'Volatility_{w}'] = df['Log_Return_1D'].rolling(w).std() * np.sqrt(252)
    df['EMA_12'] = c.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = c.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI_14'] = 100 - 100/(1 + gain / loss.replace(0, np.nan))
    tr = pd.concat([df['High']-df['Low'], (df['High']-c.shift()).abs(), (df['Low']-c.shift()).abs()], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume'].rolling(20).mean()
    if 'Gold_Price_USD' in df and 'Silver_Price_USD' in df:
        df['Gold_Silver_Ratio'] = df['Gold_Price_USD'] / c
    if 'DXY_Index' in df:
        df['DXY_3M_Momentum'] = df['DXY_Index'].pct_change(63)
    if 'US_10Y_Yield' in df and 'Fed_Funds_Rate' in df:
        df['Yield_Curve_Slope'] = df['US_10Y_Yield'] - df['Fed_Funds_Rate']
    if 'Futures_1M' in df and 'Futures_12M' in df:
        df['Term_Structure_Slope'] = (df['Futures_12M'] - df['Futures_1M']) / c
    if 'CFTC_NonCommercial_Net' in df:
        df['CFTC_Net_3Y_Min'] = df['CFTC_NonCommercial_Net'].rolling(756, min_periods=100).min()
        df['CFTC_Net_3Y_Max'] = df['CFTC_NonCommercial_Net'].rolling(756, min_periods=100).max()
        den = (df['CFTC_Net_3Y_Max'] - df['CFTC_Net_3Y_Min']).replace(0, np.nan)
        df['COT_Index_3Y'] = (df['CFTC_NonCommercial_Net'] - df['CFTC_Net_3Y_Min']) / den
    for lag in [1,2,3,5,10,20]:
        df[f'Return_Lag_{lag}'] = df['Return_1D'].shift(lag)
    df['Target_Close_t1'] = c.shift(-1)
    df['Target_Return_t1'] = c.pct_change().shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / 'silver_features.csv', index_label='Date')
    audit = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(x) for x in df.dtypes],
        'missing_count': df.isna().sum().values,
        'missing_pct': (df.isna().mean().values*100).round(2),
    })
    audit.to_csv(OUT / 'data_quality_audit.csv', index=False)
    return df

if __name__ == '__main__':
    frame = load_and_engineer()
    print(f'Prepared {len(frame):,} daily rows, {frame.shape[1]} columns')
    print(frame[['Close','Target_Close_t1']].dropna().tail())
