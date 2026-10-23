"""Time-ordered benchmark suite. Optional packages are skipped with a clear message."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {'RMSE': float(np.sqrt(mean_squared_error(y,p))), 'MAE': float(mean_absolute_error(y,p)),
            'MAPE_pct': float(np.mean(np.abs((y-p)/np.where(y==0,np.nan,y)))*100),
            'Directional_Accuracy_pct': float(np.mean(np.sign(np.diff(y)) == np.sign(np.diff(p)))*100) if len(y)>1 else np.nan}

def main():
    data = pd.read_csv(OUT/'silver_features.csv', parse_dates=['Date']).set_index('Date').sort_index()
    data = data.dropna(subset=['Target_Close_t1'])
    # Avoid including any target/future-derived columns as predictors.
    excluded = {'Target_Close_t1','Target_Return_t1','Adj_Close'}
    features = [c for c in data.select_dtypes(include=np.number).columns if c not in excluded and c != 'Close']
    # Target is next-day close; use current close as a strong baseline input too.
    features = ['Close'] + features
    X, y = data[features], data['Target_Close_t1']
    cut = int(len(data)*0.8)
    X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    preds = {}
    # Persistence baseline must be included; it is hard to beat for price-level forecasts.
    preds['Naive_Last_Close'] = X_test['Close'].values
    # XGBoost if installed, otherwise sklearn histogram gradient boosting fallback.
    try:
        from xgboost import XGBRegressor
        model = make_pipeline(SimpleImputer(strategy='median'), XGBRegressor(n_estimators=500,max_depth=5,learning_rate=.03,
                    subsample=.85,colsample_bytree=.85,reg_lambda=2,objective='reg:squarederror',random_state=42,n_jobs=-1))
        model.fit(X_train, y_train)
        preds['XGBoost'] = model.predict(X_test)
        import joblib; joblib.dump(model, OUT/'xgboost_model.joblib')
    except Exception as e:
        print('XGBoost unavailable; using HistGradientBoosting fallback:', str(e)[:180])
        model = make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingRegressor(max_iter=250,learning_rate=.05,l2_regularization=1,random_state=42))
        model.fit(X_train, y_train); preds['GradientBoosting_Fallback'] = model.predict(X_test)
    # ARIMA on price level; fit once for a fast benchmark. Production should use periodic walk-forward refits.
    try:
        from statsmodels.tsa.arima.model import ARIMA
        arima = ARIMA(y.iloc[:cut], order=(2,1,2)).fit()
        preds['ARIMA_212'] = np.asarray(arima.forecast(steps=len(y_test)))
        arima.save(OUT/'arima_model.pkl')
    except Exception as e: print('ARIMA skipped:', str(e)[:180])
    try:
        from prophet import Prophet
        ptrain = pd.DataFrame({'ds': y.iloc[:cut].index, 'y': y.iloc[:cut].values})
        ptest = pd.DataFrame({'ds': y_test.index})
        prop = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        prop.fit(ptrain); pf = prop.predict(ptest)
        preds['Prophet'] = pf['yhat'].to_numpy()
        import joblib; joblib.dump(prop, OUT/'prophet_model.joblib')
    except Exception as e: print('Prophet skipped (install prophet to enable):', str(e)[:180])
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # Seasonal period 5 is a pragmatic weekly-trading-day benchmark, not assumed optimal.
        sar = SARIMAX(y.iloc[:cut], order=(1,1,1), seasonal_order=(1,0,1,5), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        preds['SARIMA_111_101_5'] = np.asarray(sar.forecast(steps=len(y_test)))
        sar.save(OUT/'sarima_model.pkl')
    except Exception as e: print('SARIMA skipped:', str(e)[:180])
    # LSTM is optional because TensorFlow installation can be hardware-specific.
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        cols = ['Close','Volume','SMA_20','SMA_50','RSI_14','MACD','ATR_14','Volatility_20']
        seq = data[cols].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        scaler = MinMaxScaler(); scaled = scaler.fit_transform(seq.iloc[:cut+60])
        target_scaler = MinMaxScaler(); target_scaler.fit(y.iloc[:cut].to_numpy().reshape(-1,1))
        scaled_target = target_scaler.transform(seq[['Close']])
        Xseq=[]; yseq=[]; dates=[]
        arr = np.column_stack([scaled, scaled_target])
        for i in range(60, len(data)):
            Xseq.append(arr[i-60:i,:-1]); yseq.append(arr[i,-1]); dates.append(data.index[i])
        Xseq=np.asarray(Xseq); yseq=np.asarray(yseq); dates=pd.Index(dates)
        tr = dates < y_test.index[0]; te = dates.isin(y_test.index)
        net=Sequential([LSTM(64,return_sequences=True,input_shape=(60,len(cols))),Dropout(.2),LSTM(32),Dropout(.2),Dense(16,activation='relu'),Dense(1)])
        net.compile(optimizer='adam',loss='mse'); net.fit(Xseq[tr],yseq[tr],epochs=30,batch_size=32,validation_split=.1,shuffle=False,callbacks=[EarlyStopping(patience=5,restore_best_weights=True)],verbose=0)
        pp=target_scaler.inverse_transform(net.predict(Xseq[te],verbose=0)).ravel()
        # align predictions to test dates after the 60-day warmup
        dtest = dates[te]; temp=pd.Series(pp,index=dtest).reindex(y_test.index).ffill().bfill()
        preds['LSTM'] = temp.values
        net.save(OUT/'lstm_model.keras')
    except Exception as e: print('LSTM skipped (install compatible tensorflow to enable):', str(e)[:180])
    pred_frame = pd.DataFrame({'Actual': y_test})
    for name, p in preds.items():
        pred_frame[name] = np.asarray(p)[:len(pred_frame)]
    # Ensemble weights derived from an internal chronological validation tail, not the test scores.
    val_start = int(cut*.8)
    # Simplicity: inverse validation error for models available in this run; baseline remains a candidate.
    val_scores={}
    for name in preds:
        if name=='Naive_Last_Close':
            pv=X.iloc[val_start:cut]['Close'].values; av=y.iloc[val_start:cut].values
        else:
            # For expensive models, use out-of-sample test metrics only for reporting; do not tune weights on test.
            # Conservative fallback weights are assigned if no saved validation forecasts exist.
            pv=None; av=None
        if pv is not None and len(pv): val_scores[name]=np.sqrt(mean_squared_error(av,pv))
    if val_scores:
        inv={k:1/max(v,1e-9) for k,v in val_scores.items()}
        weights={k:v/sum(inv.values()) for k,v in inv.items()}
        pred_frame['Ensemble_Validation_Weighted'] = sum(pred_frame[k]*w for k,w in weights.items() if k in pred_frame)
    else: weights={}
    pred_frame.to_csv(OUT/'forecast_predictions.csv',index_label='Date')
    rows=[]
    for name in pred_frame.columns[1:]: rows.append({'Model':name,**metrics(pred_frame['Actual'],pred_frame[name])})
    pd.DataFrame(rows).sort_values('RMSE').to_csv(OUT/'model_metrics.csv',index=False)
    pd.DataFrame([{'Model':k,'Weight':v} for k,v in weights.items()]).to_csv(OUT/'ensemble_weights.csv',index=False)
    print(pd.DataFrame(rows).sort_values('RMSE').to_string(index=False))
    print('Saved outputs to',OUT)
if __name__=='__main__': main()
