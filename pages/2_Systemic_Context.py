# ======================================================
#                  MODEL IMPORTS
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

st.set_page_config(page_title="Systemic Context", layout="wide")

# BROWSER-SIDE CLOCK INJECTION
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


# ==========================================
#              INFERENCE & UI
# ==========================================
st.title("🌐 Macro-Regime Heatmap")
st.markdown("Visualizing the invisible macroeconomic forces and sectoral rotations driving the Nifty 50.")

@st.cache_data(ttl=3600)
def fetch_macro_data():
    global_tickers = ['^GSPC', '^N225', '^FTSE', '^NSEI']
    sector_tickers = ['^NSEBANK', '^CNXIT', '^CNXFMCG', '^CNXAUTO']
    data = yf.download(global_tickers + sector_tickers, period="60d", progress=False)['Close']
    returns = data.pct_change().dropna()
    return data, returns

try:
    with st.spinner("Compiling Global Contagion and Sector Flow data..."):
        raw_prices, daily_returns = fetch_macro_data()
        
    # ==========================================
    # ROW 1: CONTAGION & RADAR
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Global Contagion Matrix", help="Measures the 60-day rolling correlation between major global indices. A value near +1 indicates markets are moving in lockstep (high contagion risk).")
        
        # Fetch the columns and immediately forward-fill missing global holidays, then drop remaining NaNs
        global_ret = daily_returns[['^GSPC', '^N225', '^FTSE', '^NSEI']].ffill().dropna()
        global_ret.columns = ['S&P 500 (US)', 'Nikkei (JP)', 'FTSE (UK)', 'Nifty 50 (IN)']
        corr_matrix = global_ret.corr().round(2)
        
        fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_heat.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, width="stretch")

    with col2:
        st.subheader("Sectoral Rotation Radar", help="Visualizes capital flow across Indian sectors over the last 30 days. Defensive sector outperformance (FMCG) often precedes broader market stress.")
        
        sectors = ['^NSEBANK', '^CNXIT', '^CNXFMCG', '^CNXAUTO']
        labels = ['BankNifty', 'IT', 'FMCG', 'Auto']
        
        sector_returns = []
        for tick in sectors:
            ret = (raw_prices[tick].iloc[-1] - raw_prices[tick].iloc[-30]) / raw_prices[tick].iloc[-30]
            sector_returns.append(ret * 100) 
            
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=sector_returns,
            theta=labels,
            fill='toself',
            fillcolor='rgba(0, 255, 255, 0.2)', # Transparent Cyan Fill
            line=dict(color='cyan', width=2),   # Bright Cyan Border
            marker=dict(color='cyan', size=8)
        ))
        
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',  # <-- THIS FIXES THE WHITE CIRCLE
                radialaxis=dict(visible=True, gridcolor='rgba(255, 255, 255, 0.2)', color='white'),
                angularaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)', color='white')
            ),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=10, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_radar, width="stretch")

    st.markdown("---")

    # ==========================================
    # ROW 2: INSTITUTIONAL FLOW TRACKER
    # ==========================================
    st.subheader("Net Institutional Cash Flow (FII vs. DII)", help="Tracks net buying/selling by Foreign (FII) and Domestic (DII) institutions. Severe market crashes are historically triggered by massive, sustained FII capitulation.")
    
    # Mathematical Simulation of FII/DII Flow (Since NSE API is restricted)
    # Note: Replace this block with your own CSV read if you have hard NSE data!
    np.random.seed(42) # Keeps the chart stable on refresh
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    # Simulate FIIs selling in a stressed regime, and DIIs attempting to buy the dip
    fii_flow = np.random.normal(loc=-1500, scale=2000, size=30) 
    dii_flow = np.random.normal(loc=1200, scale=1500, size=30)
    
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(x=dates, y=fii_flow, name="FII Net Flow (Cr)", marker_color='#ff4b4b')) # Red
    fig_flow.add_trace(go.Bar(x=dates, y=dii_flow, name="DII Net Flow (Cr)", marker_color='#00ff00')) # Green
    
    fig_flow.update_layout(
        barmode='group',
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig_flow.update_yaxes(title_text="Net Flow (₹ Crores)", gridcolor='rgba(255, 255, 255, 0.1)')
    st.plotly_chart(fig_flow, width="stretch")

except Exception as e:
    st.error(f"Market Data Feed Offline: {e}")


st.markdown("---")
st.subheader("The Dollar Liquidity Drain")
st.markdown("""
Emerging markets bleed capital when the US Dollar surges. By overlaying the US Dollar Index (DXY) against the Nifty 50, we can visualize macroeconomic capital flight. 
**Both assets have been time-synchronized and normalized to a base of 100**. This mathematically aligns their starting points, making the inverse correlation and relative performance instantly visible.
""")

def fetch_normalized_liquidity():
    try:
        # 1. Fetch 1 year of data
        nifty = yf.download("^NSEI", period="1y")['Close']
        dxy = yf.download("DX-Y.NYB", period="1y")['Close']
        
        # 2. Combine into a single matrix to sync dates (handling holiday mismatches)
        df = pd.concat([nifty, dxy], axis=1)
        df.columns = ['Nifty', 'DXY']
        
        # 3. Forward-fill the gaps to prevent chart breaking, then drop initial NaNs
        df = df.fillna(method='ffill').dropna() 
        
        # 4. The Normalization Engine (Base 100)
        df['Nifty_Norm'] = (df['Nifty'] / df['Nifty'].iloc[0]) * 100
        df['DXY_Norm'] = (df['DXY'] / df['DXY'].iloc[0]) * 100
        
        return df
    except Exception as e:
        st.error(f"Failed to fetch liquidity data: {e}")
        return None

with st.spinner("Synchronizing global liquidity metrics..."):
    df_liquidity = fetch_normalized_liquidity()

if df_liquidity is not None:
    # Build a standard single-axis Plotly chart (since data is now scaled together)
    fig = go.Figure()

    # Nifty 50 Trace
    fig.add_trace(go.Scatter(
        x=df_liquidity.index, 
        y=df_liquidity['Nifty_Norm'], 
        name="Nifty 50", 
        line=dict(color="#00FFAA", width=2.5)
    ))

    # DXY Trace
    fig.add_trace(go.Scatter(
        x=df_liquidity.index, 
        y=df_liquidity['DXY_Norm'], 
        name="US Dollar Index (DXY)", 
        line=dict(color="#FF4B4B", width=2.5, dash='dot')
    ))

    # Dark Mode Aesthetic Formatting
    fig.update_layout(
        title="Macro Liquidity Overlay: Nifty 50 vs DXY (Base 100)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
        yaxis_title="Normalized Level (Base 100)",
        xaxis_title="Date"
    )

    # Clean axes
    fig.update_xaxes(showgrid=False, color="white")
    fig.update_yaxes(showgrid=False, color="white")

    # Render without use_container_width per your deprecation updates
    st.plotly_chart(fig)
else:
    st.warning("Liquidity overlay currently offline.")

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
    st.page_link("pages/3_Live_Markets.py", label="Live Markets", icon="📈")
with nav4:
    st.page_link("pages/4_Backtest_Results.py", label="Backtests", icon="📊")
with nav5:
    st.page_link("pages/5_XAI_Engine.py", label="XAI Engine", icon="🧠")