from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; DATA=ROOT/'data'
st.set_page_config(page_title='Silver Intelligence',page_icon='🥈',layout='wide')
st.title('🥈 Silver Intelligence | Forecasting & Commodity Trading Arena')
st.caption('Research and training prototype — forecasts and simulated trades are not investment advice.')

def read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()
features=read_csv(OUT/'silver_features.csv',parse_dates=['Date'])
preds=read_csv(OUT/'forecast_predictions.csv',parse_dates=['Date'])
metrics=read_csv(OUT/'model_metrics.csv')
tab1,tab2,tab3,tab4=st.tabs(['Market Overview','Model Lab','Risk Monitor','Trading Arena'])
with tab1:
    if features.empty:
        st.warning('Run `python src/data_pipeline.py` first to prepare the datasets.')
    else:
        features=features.sort_values('Date'); last=features.iloc[-1]
        a,b,c,d=st.columns(4)
        a.metric('Latest close (USD/oz)',f"${last['Close']:,.2f}")
        b.metric('Daily change',f"{last.get('Returns_Pct',np.nan):.2f}%")
        bval=last.get('Volatility_20',np.nan); c.metric('20-day annualised volatility',f'{bval:.1%}' if pd.notna(bval) else 'N/A')
        d.metric('RSI (14)',f"{last.get('RSI_14',np.nan):.1f}")
        st.subheader('Silver price history')
        fig=go.Figure(); fig.add_trace(go.Scatter(x=features.Date,y=features.Close,name='Close',line=dict(width=1.5)))
        for col in ['SMA_20','SMA_50','SMA_200']:
            if col in features: fig.add_trace(go.Scatter(x=features.Date,y=features[col],name=col,line=dict(width=1)))
        fig.update_layout(height=480,xaxis_title='Date',yaxis_title='USD per troy ounce',legend_orientation='h'); st.plotly_chart(fig,use_container_width=True)
        if not preds.empty:
            st.subheader('Out-of-sample forecasts')
            choices=[c for c in preds.columns if c not in ['Date','Actual']]
            selected=st.multiselect('Models',choices,default=choices[:4])
            fig=go.Figure(); fig.add_trace(go.Scatter(x=preds.Date,y=preds.Actual,name='Actual',line=dict(width=2)))
            for col in selected: fig.add_trace(go.Scatter(x=preds.Date,y=preds[col],name=col,line=dict(width=1)))
            fig.update_layout(height=420,yaxis_title='USD/oz'); st.plotly_chart(fig,use_container_width=True)
with tab2:
    st.subheader('Forecast model comparison')
    if metrics.empty: st.info('Run `python src/train_models.py` to train available models and generate metrics.')
    else:
        st.dataframe(metrics.sort_values('RMSE'),use_container_width=True,hide_index=True)
        fig=px.bar(metrics.sort_values('RMSE'),x='Model',y='RMSE',title='Test-period RMSE (lower is better)'); st.plotly_chart(fig,use_container_width=True)
        st.caption('Compare models on identical dates. Price-level persistence is a required baseline; report directional accuracy and transaction-cost-adjusted returns before judging trading utility.')
with tab3:
    st.subheader('Portfolio risk sandbox')
    capital=st.number_input('Virtual portfolio equity ($)',min_value=1000.0,value=100000.0,step=5000.0)
    exposure=st.slider('Silver exposure (% of equity)',0,100,25)
    shock=st.slider('Hypothetical overnight price shock (%)',-30,30,-5)
    vol=st.slider('Annualised volatility assumption (%)',1,100,30)
    pnl=capital*exposure/100*shock/100
    st.metric('Scenario P&L',f'${pnl:,.2f}',f'{pnl/capital:.2%} of equity')
    st.metric('One-day 95% parametric VaR (approx.)',f"${capital*exposure/100*(vol/100)/np.sqrt(252)*1.645:,.2f}")
    st.caption('Illustrative normal-return VaR; it can understate tail risk and does not model margin, liquidity, options, or correlations.')
with tab4:
    st.subheader('Commodity Trading Arena — practice mode')
    st.write('Start with virtual capital, choose a scenario, and place simulated trades. No broker connection or real orders are used.')
    scenario=st.selectbox('Market scenario',['Normal market','Crisis sell-off','Flash crash and rebound','Supply shock','Sentiment squeeze'])
    initial=st.number_input('Starting virtual balance ($)',min_value=1000,value=100000,step=5000,key='arena_cash')
    move_map={'Normal market':1.0,'Crisis sell-off':-8.0,'Flash crash and rebound':-12.0,'Supply shock':6.0,'Sentiment squeeze':10.0}
    move=move_map[scenario]
    st.info(f'Scenario card: {scenario}. Training shock used for this round: {move:+.1f}%. In a production game, shocks should be generated from seeded historical/event distributions.')
    side=st.radio('Simulated position', ['Stay in cash','Long silver','Short silver'],horizontal=True)
    size=st.slider('Exposure (% of starting balance)',0,100,20)
    signed={'Stay in cash':0,'Long silver':1,'Short silver':-1}[side]
    result=initial*(size/100)*signed*(move/100)
    st.metric('Round P&L',f'${result:,.2f}',f'{result/initial:.2%} of starting balance')
    st.metric('Ending virtual equity',f'${initial+result:,.2f}')
    st.caption('Educational scenario only; this simplified round excludes slippage, fees, leverage, margin calls, and order-book effects.')
