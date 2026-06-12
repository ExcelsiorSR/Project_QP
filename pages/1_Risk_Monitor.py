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
        csv_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'latest_stress_headlines.csv'
        )

        if not os.path.exists(csv_path):
            return pd.DataFrame()

        return pd.read_csv(csv_path)

    except Exception as e:
        st.sidebar.error(f"News Feed Error: {e}")
        return pd.DataFrame()
    
@st.cache_data(ttl=1800)
def fetch_historical_predictions():
    # Strict OS-agnostic pathing for Linux cloud servers
    pred_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/final_predictions.csv'))
    
    if not os.path.exists(pred_path):
        st.sidebar.error(f"System Error: {pred_path} not found on cloud server.")
        return pd.DataFrame()
        
    try:
        # 1. Load actual AI predictions safely
        df = pd.read_csv(pred_path, index_col='Date', parse_dates=True)
        if df.empty:
            df['Close'] = []
            return df
            
        # 2. Attempt the Yahoo Finance API call separately
        try:
            start_str = df.index[0].strftime('%Y-%m-%d')
            end_str = df.index[-1].strftime('%Y-%m-%d')
            
            nifty = yf.download('^NSEI', start=start_str, end=end_str, progress=False)
            
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
                
            if not nifty.empty and 'Close' in nifty.columns:
                df['Close'] = nifty['Close']
            else:
                df['Close'] = np.nan
                
        except Exception as api_err:
            st.sidebar.warning(f"Yahoo API Rate Limited. Plotting probabilities only.")
            df['Close'] = np.nan # Graceful fallback if cloud IP is blocked
            
        return df.tail(200)
        
    except Exception as e:
        st.sidebar.error(f"Data ingestion failed: {e}")
        return pd.DataFrame()
    
@st.cache_data
def load_macro_history():

    path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'macro_stress_signals.csv'
    )

    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['Date'])

    return df.tail(30)

@st.cache_data
def load_live_risk_history():

    path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'live_risk_history.csv'
    )

    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['Date'])

    return df.tail(30)



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
        nlp_path = os.path.join(os.path.dirname(__file__), '..', 'data/macro_stress_signals.csv')
        live_nlp_df = pd.read_csv(nlp_path)
        today_stress = float(live_nlp_df['Stress_Score'].dropna().iloc[-1])
    except:
        today_stress = 0.0
        
    latest_features['Stress_Score'] = today_stress
    
    # Probability Extraction
    try:
        if not hist_df.empty and 'Crash_Probability' in hist_df.columns:
            # Pull the last row that actually contains a valid probability score
            prob = float(hist_df['Crash_Probability'].dropna().iloc[-1]) * 100
        else:
            prob = 0.0
    except Exception as e:
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
        
        # Pull live India VIX directly from Yahoo Finance ticker India VIX ("^INDIAVIX")
        try:
            vix_data = yf.download('^INDIAVIX', period='1d', progress=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = vix_data.columns.get_level_values(0)
            vix_val = float(vix_data['Close'].iloc[-1])
        except Exception as e:
            # Fallback to feature engine proxies if API fails
            vix_val = float(live_df['VIX_MA_5'].dropna().iloc[-1]) if 'VIX_MA_5' in live_df.columns else 0.0
            
        st.metric(label="India VIX", value=f"{vix_val:.2f}", help="Volatility Index: Measures the market's expectation of 30-day forward volatility. Values > 25 indicate severe systemic fear.")
        st.metric(label="Sentiment Stress", value=f"{today_stress:.2f}", help="""
                                                                            Proprietary FinBERT sentiment index aggregating real-time financial headlines.

                                                                            0.00 - 0.40 → Normal
                                                                            0.40 - 0.65 → Elevated
                                                                            0.65 - 0.80 → High Stress
                                                                            0.80 - 1.00 → Crisis Conditions

                                                                            Values above 0.80 historically correspond to periods of severe macroeconomic uncertainty,
                                                                            liquidity stress, geopolitical escalation, or broad risk-off sentiment.
                                                                            """)
        
    with col3:
        st.subheader("Macro-Tangents")
        
        # Pull live Brent Crude Oil Spot price directly ("BZ=F")
        try:
            crude_data = yf.download('BZ=F', period='1d', progress=False)
            if isinstance(crude_data.columns, pd.MultiIndex):
                crude_data.columns = crude_data.columns.get_level_values(0)
            crude_val = float(crude_data['Close'].iloc[-1])
        except Exception as e:
            crude_val = 0.0
            
        st.metric(label="Crude Oil", value=f"${crude_val:.2f}", help="Global benchmark price for Brent Crude. As a heavy net importer, a spike in crude oil acts as a direct tax on the Indian economy, driving inflation and tightening corporate margins.")
        
        # Pull live USD/INR exchange rate spot directly ("INR=X")
        try:
            usd_inr_data = yf.download('INR=X', period='1d', progress=False)
            if isinstance(usd_inr_data.columns, pd.MultiIndex):
                usd_inr_data.columns = usd_inr_data.columns.get_level_values(0)
            inr_val = float(usd_inr_data['Close'].iloc[-1])
        except Exception as e:
            inr_val = 0.0
            
        st.metric(label="USD/INR", value=f"₹{inr_val:.2f}", help="US Dollar to Indian Rupee exchange rate. A surging US Dollar typically triggers Foreign Institutional Investor (FII) outflows, draining liquidity from emerging markets.")

    if st.button("🔄 Refresh Live Feeds", width='stretch'):
        st.rerun()
          
    st.markdown("---")
    
    # ==================================================
    # MIDDLE ROW: HISTORICAL RISK CHART
    # ==================================================
    if not hist_df.empty and 'Close' in hist_df.columns and 'Crash_Probability' in hist_df.columns:
        st.subheader("📉 Historical Risk vs. Nifty 50")
        
        # Create dual-axis plot
        fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Nifty Price Line
        fig_hist.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], name="Nifty 50", 
                                      line=dict(color='blue')), secondary_y=False)
        
        # Crash Probability Area
        fig_hist.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Crash_Probability'] * 100, 
                                      name="Crash Probability %", fill='tozeroy', 
                                      line=dict(color='rgba(255, 0, 0, 0.3)')), secondary_y=True)
        
        # Add Threshold Line
        fig_hist.add_hline(y=DYNAMIC_THRESHOLD, line_dash="dot", line_color="red", annotation_text="Danger Threshold", secondary_y=True)
        
        fig_hist.update_layout(title_text="Recent Regime Shifts", template="plotly_white", height=400)
        fig_hist.update_yaxes(title_text="Nifty 50 Index", secondary_y=False)
        fig_hist.update_yaxes(title_text="Probability (%)", range=[0, 100], secondary_y=True)
        
        st.plotly_chart(fig_hist, width="stretch")
        st.markdown("---")

    # ==================================================
    # 30-DAY HISTORICAL DASHBOARD
    # ==================================================

    st.subheader("📊 30-Day Risk Dashboard")

    risk_df = load_live_risk_history()

    risk_df = load_live_risk_history()

    col_left, col_right = st.columns(2)

    # --------------------------------------------------
    # SENTIMENT STRESS
    # --------------------------------------------------

    with col_left:

        fig_stress = go.Figure()

        fig_stress.add_trace(
            go.Scatter(x=risk_df['Date'],
                       y=risk_df['Stress_Score'],
                       mode='lines+markers',
                       name='Stress Score'
                       )
                       )

        fig_stress.add_hline(
            y=0.40,
            line_dash="dot",
            annotation_text="Normal"
        )

        fig_stress.add_hline(
            y=0.65,
            line_dash="dot",
            annotation_text="Elevated"
        )

        fig_stress.add_hline(
            y=0.80,
            line_dash="dot",
            annotation_text="Crisis"
        )

        fig_stress.update_layout(
            title="FinBERT Sentiment Stress",
            height=350,
            yaxis_title="Stress Score",
            template="plotly_dark"
        )

        st.plotly_chart(fig_stress, width="stretch")

    # --------------------------------------------------
    # CRASH PROBABILITY
    # --------------------------------------------------

    with col_right:

        fig_prob = go.Figure()

        fig_prob.add_trace(
            go.Scatter(
                x=risk_df['Date'],
                y=risk_df['Crash_Probability'] * 100,
                mode='lines+markers',
                name='Crash Probability'
            )
        )

        fig_prob.add_hline(
            y=DYNAMIC_THRESHOLD,
            line_dash="dot",
            annotation_text="Danger Threshold"
        )

        fig_prob.update_layout(
            title="Crash Probability Trend",
            height=350,
            yaxis_title="Probability (%)",
            template="plotly_dark"
        )

        st.plotly_chart(fig_prob, width="stretch")

    st.markdown("---")

    # ==================================================
    # BOTTOM ROW: NLP FEED
    # ==================================================
    st.subheader("📰 Live FinBERT Sentiment Feed.")
    if not news_df.empty:
        for _, row in news_df.head(10).iterrows():
            st.markdown(
                f"• [{row['headline']}]({row['url']}) "
                f"<span style='color:gray'>({row['source']})</span>",
                unsafe_allow_html=True
            )
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