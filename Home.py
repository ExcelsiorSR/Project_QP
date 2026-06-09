# ======================================================
#                  MODEL IMPORTS
# ======================================================

import streamlit as st
import streamlit.components.v1 as components

# ==========================================
#              INFERENCE & UI
# ==========================================

st.set_page_config(
    page_title="Predicting Crash",
    page_icon="⚡",
    layout="wide"
)

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
# 2. ENGAGING HERO SECTION
# ==========================================
st.title("⚡ Predicting Crash: Evaluating Market Panic")
st.markdown("### *An AI-Driven Approach to Detecting Systemic Tail-Risk*")
st.write("")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Welcome to the **Behavioral Regime Shift Classifier**. 
    
    Traditional financial models assume market returns follow a normal distribution, severely underestimating the probability of catastrophic crashes. This architecture abandons traditional assumptions, fusing raw market microstructure with **Live NLP Sentiment Analysis** to detect the invisible onset of macroeconomic panic.
    """)
    
    st.info("**The Quant Engine:** A Cost-Sensitive Stacking Ensemble utilizing XGBoost, Random Forest, and Gradient Boosting, explicitly mathematically penalized to prioritize drawdown protection over false alarms.")

with col2:
    # ------------------------------------------
    # The Black Swan Fact Box
    # ------------------------------------------
    black_swan_html = """
    <div style="background-color: #161326; padding: 20px; border-radius: 10px; color: #F0F2F6; border: 1px solid #8B5CF6; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <h4 style="margin-top: 0; color: #A78BFA; font-weight: 800;">🦢 The Black Swan Event</h4>
        <p style="font-size: 0.95em; line-height: 1.5;">
        During the 2008 Global Financial Crisis, standard volatility metrics lagged. However, retrospective Natural Language Processing (NLP) of financial news showed severe systemic stress compounding <em>weeks</em> before the actual market top.
        </p>
        <p style="font-size: 0.95em; font-weight: bold; margin-bottom: 0; color: #C4B5FD;">
        Panic always leaves a digital footprint.
        </p>
    </div>
    """
    st.markdown(black_swan_html, unsafe_allow_html=True)

st.write("")
st.write("")

# ------------------------------------------
# The Yellow Educational Disclaimer
# ------------------------------------------
st.warning("**Educational Disclaimer:** This system is strictly for academic and educational purposes. The regime shift probabilities, macroeconomic stress scores, and aggregated news sentiment provided by this platform do not constitute financial advice. Algorithmic trading carries significant risk.", icon="⚠️")
st.write("")

# ==========================================
# 3. GLOBAL FOOTER NAVIGATION
# ==========================================
st.markdown("---")

nav1, nav2, nav3, nav4, nav5 = st.columns(5)
with nav1:
    st.page_link("pages/1_Risk_Monitor.py", label="Risk Monitor", icon="🚨")
with nav2:
    st.page_link("pages/2_Systemic_Context.py", label="Macro View", icon="🌐")
with nav3:
    st.page_link("pages/3_Live_Markets.py", label="Live Markets", icon="📈")
with nav4:
    st.page_link("pages/4_Backtest_Results.py", label="Backtests", icon="📊")
with nav5:
    st.page_link("pages/5_XAI_Engine.py", label="XAI Engine", icon="🧠")