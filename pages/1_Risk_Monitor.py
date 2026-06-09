# ======================================================
#                  MODEL IMPORTS
# ======================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import sqlite3
from modules.model_engine import CostSensitiveEnsemble
import streamlit.components.v1 as components

st.set_page_config(page_title="Predicting Crash", layout="wide")

# ==========================================
#               DATA INGESTION
# ==========================================
@st.cache_resource
def load_engine():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'crash_predictor.pkl')
    return joblib.load(model_path)

@st.cache_data
def fetch_live_data():
    from modules.data_pipeline import IndianMarketFeatureEngine
    engine = IndianMarketFeatureEngine(start_date="2024-01-01", end_date=datetime.today().strftime('%Y-%m-%d'))
    return engine.get_final_dataset()

@st.cache_data
def fetch_latest_news():
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/financial_news.db'))
        
        if not os.path.exists(db_path):
            st.sidebar.error(f"Database not found at: {db_path}")
            return pd.DataFrame()
            
        conn = sqlite3.connect(db_path)
        
        # 1. Automatically find the table name
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if tables.empty:
            st.sidebar.error("Database exists but contains no tables.")
            return pd.DataFrame()
            
        table_name = tables['name'].iloc[0]
        
        # 2. Fetch the data
        news_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 10", conn)
        conn.close()
        return news_df
        
    except Exception as e:
        st.sidebar.error(f"SQLite Error: {e}")
        return pd.DataFrame()
    
@st.cache_data
def fetch_historical_predictions():
    try:
        pred_path = os.path.join(os.path.dirname(__file__), '..', 'data/final_predictions.csv')
        df = pd.read_csv(pred_path, index_col='Date', parse_dates=True)
        
        # Dynamically fetch Nifty to get the 'Close' prices for the chart
        nifty = yf.download('^NSEI', start=df.index[0], end=df.index[-1], progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
            
        df['Close'] = nifty['Close']
        return df.tail(200) 
    except Exception as e:
        return pd.DataFrame()

# ==========================================
#           INFERENCE & UI
# ==========================================

# LIVE TICKING DIGITAL CLOCK (BROWSER-SIDE)
clock_html = """
<div id="clock" style="text-align: right; color: #a0a0a0; font-family: sans-serif; font-size: 1.1em; font-weight: 500; padding-top: 10px;"></div>
<script>
function updateClock() {
    let now = new Date();
    let options = { timeZone: 'Asia/Kolkata', weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' };
    document.getElementById('clock').innerText = now.toLocaleString('en-US', options) + ' IST';
}
setInterval(updateClock, 1000);
updateClock();
</script>
"""
components.html(clock_html, height=45)
st.markdown("---")

col_title, col_time = st.columns([3, 1])
with col_title:
    st.title("🚨 Live Market Regime & Tail-Risk Monitor")
st.markdown("---")

try:
    model = load_engine()
    live_df = fetch_live_data()
    news_df = fetch_latest_news()
    hist_df = fetch_historical_predictions()
    
    DYNAMIC_THRESHOLD = float(model.optimal_threshold) * 100 
    latest_features = live_df.drop('Target_Regime', axis=1, errors='ignore').iloc[-1:].copy()
    
    # NLP Injection
    try:
        nlp_path = os.path.join(os.path.dirname(__file__), '..', 'macro_stress_signals.csv')
        live_nlp_df = pd.read_csv(nlp_path)
        today_stress = float(live_nlp_df['Stress_Score'].iloc[-1])
    except:
        today_stress = 0.0
        
    latest_features['Stress_Score'] = today_stress
    
    # Probability Extraction
    try:
        raw_prob = model.predict_proba(latest_features)
        prob = float(np.array(raw_prob).flatten()[-1]) * 100
    except:
        prob = 0.0
    
    # ==================================================
    # TOP ROW: GAUGE & METRICS
    # ==================================================
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # 1. Native Streamlit Header with the Question Mark Tooltip!
        st.subheader("Systemic Crash Probability (T+1)", help="**T+1 (Trading Day + 1):** The model predicts the probability of a systemic regime shift (crash) occurring in the very next active trading session.")
        st.markdown(f"<span style='color:gray; font-size: 0.9em;'>Dynamic Threshold: {DYNAMIC_THRESHOLD:.1f}%</span>", unsafe_allow_html=True)
        
        # 2. The Plotly Gauge (with the internal title removed)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            # Notice the 'title' parameter has been completely deleted from here!
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, max(0, DYNAMIC_THRESHOLD - 20)], 'color': "lightgreen"},
                    {'range': [max(0, DYNAMIC_THRESHOLD - 20), DYNAMIC_THRESHOLD], 'color': "orange"}, 
                    {'range': [DYNAMIC_THRESHOLD, 100], 'color': "red"}],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': DYNAMIC_THRESHOLD}}))
        
        # Squeeze the margins so the gauge sits tight against our new header
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
        st.plotly_chart(fig_gauge, width="stretch")
        
    with col2:
        st.subheader("Microstructure")
        vix_val = latest_features['India_VIX'].iloc[0] if 'India_VIX' in latest_features.columns and not latest_features.empty else 0.0
        st.metric(label="India VIX", value=f"{vix_val:.2f}", help="Volatility Index: Measures the market's expectation of 30-day forward volatility. Values > 25 indicate severe systemic fear.")
        
        st.metric(label="NLP Stress", value=f"{today_stress:.2f}", help="Proprietary FinBERT sentiment index aggregating real-time financial headlines. A rising score indicates compounding macroeconomic fear and negative news flow.")
        
    with col3:
        st.subheader("Macro-Tangents")
        crude_val = latest_features['Crude_Oil'].iloc[0] if 'Crude_Oil' in latest_features.columns and not latest_features.empty else 0.0
        st.metric(label="Crude Oil", value=f"${crude_val:.2f}", help="Global benchmark price. As a heavy net importer, a spike in crude oil acts as a direct tax on the Indian economy, driving inflation and tightening corporate margins.")
        
        inr_val = latest_features['INR_USD'].iloc[0] if 'INR_USD' in latest_features.columns and not latest_features.empty else 0.0
        st.metric(label="USD/INR", value=f"₹{inr_val:.2f}", help="US Dollar to Indian Rupee exchange rate. A surging US Dollar typically triggers Foreign Institutional Investor (FII) outflows, draining liquidity from emerging markets.")

    st.markdown("---")
    
    # ==================================================
    # MIDDLE ROW: HISTORICAL RISK CHART
    # ==================================================
    if not hist_df.empty and 'Close' in hist_df.columns and 'Crash_Probability' in hist_df.columns:
        st.subheader("📉 Historical Risk vs. Nifty 50")
        
        # Create dual-axis plot
        fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Nifty Price Line
        fig_hist.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], name="Nifty 50", line=dict(color='blue')), secondary_y=False)
        
        # Crash Probability Area
        fig_hist.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Crash_Probability'] * 100, name="Crash Probability %", fill='tozeroy', line=dict(color='rgba(255, 0, 0, 0.3)')), secondary_y=True)
        
        # Add Threshold Line
        fig_hist.add_hline(y=DYNAMIC_THRESHOLD, line_dash="dot", line_color="red", annotation_text="Danger Threshold", secondary_y=True)
        
        fig_hist.update_layout(title_text="Recent Regime Shifts", template="plotly_white", height=400)
        fig_hist.update_yaxes(title_text="Nifty 50 Index", secondary_y=False)
        fig_hist.update_yaxes(title_text="Probability (%)", range=[0, 100], secondary_y=True)
        
        st.plotly_chart(fig_hist, width="stretch")
        st.markdown("---")

    # ==================================================
    # BOTTOM ROW: NLP FEED
    # ==================================================
    st.subheader("📰 Live FinBERT Sentiment Feed.")
    if not news_df.empty:
        for idx, row in news_df.head(10).iterrows():
            st.markdown(f"**[{row['headline']}]({row['url']})** - *{row['source']}*")
    else:
        st.warning("No live news data found. Please run scripts/news_scraper.py to populate the database.")

except Exception as e:
    st.error(f"System Offline. Error: {e}")

# ===================================================================================
#                             GLOBAL NAVIGATION FOOTER
# ===================================================================================
st.markdown("---")

nav1, nav2, nav3, nav4, nav5 = st.columns(5)
with nav1:
    st.page_link("Home.py", label="Home", icon="🏠")
with nav2:
    st.page_link("pages/2_Systemic_Context.py", label="Macro View", icon="🌐")
with nav3:
    st.page_link("pages/3_Live_Markets.py", label="Live Markets", icon="📈")
with nav4:
    st.page_link("pages/4_Backtest_Results.py", label="Backtests", icon="📊")
with nav5:
    st.page_link("pages/5_XAI_Engine.py", label="XAI Engine", icon="🧠")    