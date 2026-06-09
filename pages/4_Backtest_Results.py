# ======================================================
#                  MODEL IMPORTS
# ======================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Backtest Analytics", layout="wide")

# ==========================================
#              INFERENCE & UI
# ==========================================
st.title("📊 Historical Backtest Performance")
st.markdown("Comparing the NLP-Enhanced Tail Risk Model against a Nifty 50 Buy & Hold baseline.")

try:
    # 1. Load your predictions
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data/final_predictions.csv')
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    # 2. Dynamically fetch Nifty data for the exact same dates
    with st.spinner("Fetching historical Nifty 50 prices..."):
        nifty = yf.download('^NSEI', start=df.index[0], end=df.index[-1], progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
            
    # 3. Merge and calculate Returns
    df['Close'] = nifty['Close']
    df['Nifty_Return'] = df['Close'].ffill().pct_change()
    
    # 4. Simulate the Strategy (Avoiding the crash threshold)
    # Using 0.739 or whatever your dynamic threshold roughly was
    df['Strategy_Return'] = np.where(df['Crash_Probability'] >= 0.70, 0, df['Nifty_Return'])
    
    # Calculate Cumulative Growth
    df['Nifty_Cumulative'] = (1 + df['Nifty_Return']).cumprod()
    df['Strategy_Cumulative'] = (1 + df['Strategy_Return']).cumprod()
    
    # 5. Plot the Equity Curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Nifty_Cumulative'], mode='lines', name='Nifty 50 (Buy & Hold)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Strategy_Cumulative'], mode='lines', name='Project QP Portfolio', line=dict(color='green')))
    
    fig.update_layout(title="Cumulative Portfolio Returns", xaxis_title="Date", yaxis_title="Growth Factor", template="plotly_white")
    st.plotly_chart(fig, width="stretch")
    
    st.info("Note: Maximum Drawdown calculations and holding period dynamics are mathematically validated via the Backtest Simulator engine.")
    
except Exception as e:
    st.error(f"Could not load backtest data. Error: {e}")

# ===================================================================================
#                             GLOBAL NAVIGATION FOOTER
# ===================================================================================
st.markdown("---")

nav1, nav2, nav3, nav4, nav5 = st.columns(5)
with nav1:
    st.page_link("Home.py", label="Home", icon="🏠")
with nav2:
    st.page_link("pages/1_Risk_Monitor.py", label="Risk Monitor", icon="🚨")
with nav3:
    st.page_link("pages/2_Systemic_Context.py", label="Macro View", icon="🌐")
with nav4:
    st.page_link("pages/3_Live_Markets.py", label="Live Markets", icon="📈")
with nav5:
    st.page_link("pages/5_XAI_Engine.py", label="XAI Engine", icon="🧠")