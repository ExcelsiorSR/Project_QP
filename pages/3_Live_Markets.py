# ======================================================
#                  MODEL IMPORTS
# ======================================================

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

st.set_page_config(page_title="Live Markets", layout="wide")

# ==========================================
# 1. BROWSER-SIDE CLOCK INJECTION
# ==========================================
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

st.title("📈 Live Markets & Global Anchors")
st.markdown("Real-time ticker tape and price action for systemic macro variables.")

# ==========================================
# 2. BULLETPROOF DATA ENGINE (BULK FETCH)
# ==========================================

@st.cache_data(ttl=300) 
def fetch_market_pulse():
    tickers_dict = {
        "^NSEI": "Nifty 50",
        "^BSESN": "BSE Sensex",
        "^NSEBANK": "Nifty Bank",
        "^GSPC": "S&P 500 (US)",
        "^TNX": "US 10-Yr Yield",
        "CL=F": "Brent Crude",
        "GC=F": "Gold (Comex)",
        "INR=X": "USD/INR"
    }
    
    results = {}
    try:
        # BULK DOWNLOAD: 1 request instead of 8 to prevent IP bans
        ticker_list = list(tickers_dict.keys())
        df = yf.download(ticker_list, period="1mo", progress=False)['Close']
        
        for tick, name in tickers_dict.items():
            if tick in df.columns:
                series = df[tick].dropna()
                if len(series) >= 2:
                    current = float(series.iloc[-1])
                    prev = float(series.iloc[-2])
                    pct_change = ((current - prev) / prev) * 100
                    results[name] = {
                        'prices': series.values, 
                        'current': current, 
                        'change': pct_change
                    }
    except Exception as e:
        st.sidebar.error(f"Sparkline API Error: {e}")
        
    return results


@st.cache_data(ttl=3600)
def fetch_yield_energy_axis():
    try:
        # Fetch 6 months of data specifically for the dual-axis chart
        df = yf.download(['^TNX', 'CL=F'], period="6mo", progress=False)['Close']
        return df.dropna(how='all')
    except Exception as e:
        st.sidebar.error(f"Yield/Energy API Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. SPARKLINE UI GRID
# ==========================================
with st.spinner("Pinging Global Exchanges via Bulk Request..."):
    market_data = fetch_market_pulse()

if not market_data:
    st.warning("Market APIs are currently unreachable. Please check your connection or try again in a few minutes.")
else:
    items = list(market_data.items())
    cols = st.columns(4)
    
    for idx, (name, data) in enumerate(items):
        col = cols[idx % 4] 
        with col:
            prefix = ""
            suffix = ""
            if "Yield" in name: suffix = "%"
            elif "Crude" in name or "Gold" in name: prefix = "$"
            elif "INR" in name: prefix = "₹"
            
            st.metric(
                label=name, 
                value=f"{prefix}{data['current']:,.2f}{suffix}", 
                delta=f"{data['change']:.2f}%"
            )
            
            color = "green" if data['change'] >= 0 else "red"
            fig = go.Figure(go.Scatter(y=data['prices'], mode='lines', line=dict(color=color, width=2)))
            
            fig.update_layout(
                height=100, 
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False, showgrid=False),
                yaxis=dict(visible=False, showgrid=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig, width="stretch")
            st.write("") 

st.markdown("---")

# ==========================================
# 4. THE YIELD & ENERGY AXIS
# ==========================================
st.subheader("The Yield & Energy Axis", help="Tracks the US 10-Year Treasury Yield against Brent Crude. Simultaneous spikes in both variables historically trigger massive foreign capital outflows from emerging markets like India.")

with st.spinner("Loading Macro Chokepoint Data..."):
    ye_df = fetch_yield_energy_axis()

if not ye_df.empty and '^TNX' in ye_df.columns and 'CL=F' in ye_df.columns:
    fig_ye = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add US 10-Yr Yield (Left Y-Axis)
    fig_ye.add_trace(go.Scatter(x=ye_df.index, y=ye_df['^TNX'], name="US 10-Yr Yield (%)", line=dict(color='#ff4b4b', width=2)), secondary_y=False)
    
    # Add Brent Crude (Right Y-Axis)
    fig_ye.add_trace(go.Scatter(x=ye_df.index, y=ye_df['CL=F'], name="Brent Crude ($)", line=dict(color='#f6c23e', width=2)), secondary_y=True)
    
    fig_ye.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig_ye.update_yaxes(title_text="Yield (%)", gridcolor='rgba(255, 255, 255, 0.1)', secondary_y=False)
    fig_ye.update_yaxes(title_text="Price (USD)", showgrid=False, secondary_y=True)
    
    st.plotly_chart(fig_ye, width="stretch")
else:
    st.warning("Historical yield and energy data is currently unavailable.")

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
    st.page_link("pages/4_Backtest_Results.py", label="Backtests", icon="📊")
with nav5:
    st.page_link("pages/5_XAI_Engine.py", label="XAI Engine", icon="🧠")