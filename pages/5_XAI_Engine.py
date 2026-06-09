# ======================================================
#                  MODEL IMPORTS
# ======================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import shap
import joblib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="XAI Engine", layout="wide")

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
#             INFERENCE & UI
# ==========================================
st.title("🧠 Explainable AI (XAI) & System Architecture")
st.markdown("Deconstructing the Cost-Sensitive Stacking Ensemble and real-time feature attributions.")

# ==========================================
# 2. DATA & MODEL INGESTION WITH BULLETPROOF CLEANING
# ==========================================
@st.cache_resource(show_spinner=False, ttl=600)
def load_ensemble():
    """Load the trained ensemble model"""
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'crash_predictor.pkl'))
    try:
        model = joblib.load(model_path)
        # st.success("✅ Ensemble model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load ensemble: {e}")
        return None

@st.cache_data
def fetch_background_data():
    """
    Fetch and BULLETPROOF clean background data for SHAP
    CRITICAL: Convert ALL columns to native float64
    """
    try:
        from modules.data_pipeline import IndianMarketFeatureEngine
        
        print("\n🔧 Fetching background data for SHAP...")
        engine = IndianMarketFeatureEngine(
            start_date="2024-01-01", 
            end_date=datetime.today().strftime('%Y-%m-%d')
        )
        df = engine.get_final_dataset()
        print(f"✅ Fetched {len(df)} samples from data engine")
        
        # ==================================================
        # BULLETPROOF DATA CLEANING - THREE LAYERS
        # ==================================================
        
        # Layer 1: Drop the target label
        if 'Target_Regime' in df.columns:
            df = df.drop('Target_Regime', axis=1, errors='ignore')
            print("✅ Dropped target label")
        
        # Layer 2: Convert entire dataframe to float64
        # This handles mixed types, strings, scientific notation, etc.
        df = df.astype(str)  # First convert everything to string
        print("✅ Converted to strings")
        
        # Layer 3: Clean Stress_Score specifically (most problematic column)
        if 'Stress_Score' in df.columns:
            print("🔧 Deep-cleaning Stress_Score column...")
            
            # Step 3a: Remove ALL brackets, quotes, and whitespace
            df['Stress_Score'] = df['Stress_Score'].astype(str).str.replace(r'[\[\]\(\)\{\}]', '', regex=True)
            df['Stress_Score'] = df['Stress_Score'].str.replace(r'[\'\"]', '', regex=True)
            df['Stress_Score'] = df['Stress_Score'].str.strip()
            
            # Step 3b: Replace any 'nan' strings with '0.0'
            df['Stress_Score'] = df['Stress_Score'].replace(['nan', 'NaN', 'None', '', ' '], '0.0')
            
            # Step 3c: Handle scientific notation explicitly
            # Convert '6.2245935E-1' → 0.62245935
            df['Stress_Score'] = df['Stress_Score'].apply(
                lambda x: str(float(x)) if x and x != '0.0' else '0.0'
            )
            
            # Step 3d: Force numeric conversion
            df['Stress_Score'] = pd.to_numeric(df['Stress_Score'], errors='coerce').fillna(0.0)
            print("✅ Stress_Score column cleaned")
        
        # Layer 4: Convert ALL columns to float64 (critical for SHAP)
        print("🔧 Converting all columns to float64...")
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                df[col] = 0.0  # Fallback for any remaining garbage
        
        # Layer 5: Fill any NaN values with 0
        df = df.fillna(0.0)
        print("✅ Filled NaN values")
        
        # Layer 6: Isolate recent data (last 100 days) for faster SHAP computation
        background_df = df.tail(100)
        print(f"✅ Using last {len(background_df)} samples as background")
        
        # Layer 7: Validate all values are numeric
        assert background_df.applymap(lambda x: isinstance(x, (int, float, np.number))).all().all(), \
            "❌ Non-numeric values still present!"
        
        print("✅ Data validation passed - all values are numeric\n")
        return background_df.astype(np.float64)  # Final type guarantee
        
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# 3. TABBED INTERFACE
# ==========================================
tab1, tab2, tab3 = st.tabs(["🏛️ Ensemble Architecture", "🔍 SHAP Feature Analysis", "🔧 Debug Info"])

with tab1:
    st.header("The Cost-Sensitive Meta-Learner")
    st.markdown("""
    Systemic market crashes are fundamentally non-linear and incredibly rare. Relying on a single machine learning algorithm introduces severe structural bias. Project QP utilizes a **Stacking Ensemble Architecture** to balance variance, bias, and localized market noise.
    """)
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("1. The Base Learners (Non-Linearity & Variance)")
        st.markdown("""
        * **Extreme Gradient Boosting (XGBoost):** The primary alpha-generator. Highly efficient at mapping multi-dimensional, non-linear interactions (e.g., tracking simultaneous spikes in the India VIX and Brent Crude).
        * **Random Forest (RF):** The variance dampener. Utilizes bootstrap aggregating to prevent the XGBoost engine from memorizing isolated historical anomalies, smoothing the predictive baseline.
        * **Gradient Boosting Machine (GBM):** The structural bridge. Applies a different regularization penalty during sequential learning, providing a tertiary perspective on macroeconomic drift.
        """)
        
    with colB:
        st.subheader("2. The Meta-Learner (Logistic Regression)")
        st.markdown("""
        Instead of simply averaging the predictions of the three trees, a Logistic Regression model acts as the mathematical judge. It takes the raw outputs of the Base Learners and applies a calibrated linear equation to output the final $T+1$ crash probability.
        
        This layer is mathematically forced to become **Cost-Sensitive**, heavily penalizing false negatives (missing a crash) while balancing false positives to prevent excessive portfolio turnover.
        """)
        
    st.info("By decoupling the non-linear pattern recognition (Trees) from the final probability calibration (Regression), the model achieves institutional-grade precision during regime shifts.")

with tab2:
    st.header("Global Feature Importance (Random Forest Core)")
    st.markdown("Visualizing the marginal contribution of each macroeconomic feature using SHapley Additive exPlanations (SHAP).")
    
    try:
        with st.spinner("⏳ Loading ensemble and background data (this requires heavy matrix math)..."):
            ensemble = load_ensemble()
            background_df = fetch_background_data()
            
            if ensemble is None or background_df is None:
                st.error("❌ Could not load ensemble or background data")
            else:
                # Verify data types before SHAP
                # st.info(f"✅ Background data shape: {background_df.shape}")
                # st.info(f"✅ Data types: {background_df.dtypes.unique()}")
                
                with st.spinner("⏳ Calculating Shapley Values (Heavy computation, please wait 30-60 seconds)..."):
                   # 1. Pierce the ensemble to extract the raw XGBoost Booster
                    rf_core = ensemble.models['rf']
                    
                    # Initialize SHAP explainer on the RF model
                    explainer = shap.TreeExplainer(rf_core)
                    
                    # Calculate SHAP values (check_additivity=False prevents floating point math errors)
                    shap_values = explainer.shap_values(background_df, check_additivity=False)
                    
                    # Scikit-Learn RF returns SHAP values for both classes [No Crash, Crash]. 
                    # Isolate index 1 to plot the drivers of the Crash probability.
                    if isinstance(shap_values, list):
                        plot_values = shap_values[1]
                    elif len(np.shape(shap_values)) == 3:
                        if np.shape(shap_values)[2] == 2:
                            plot_values = shap_values[:, :, 1] # Shape: (samples, features, classes)
                        else:
                            plot_values = shap_values[1]       # Shape: (classes, samples, features)
                    else:
                        plot_values = shap_values
                        
                    # Render SHAP Summary Plot
                    plt.style.use('dark_background')

                    # ==================================================
                    # PRODUCTION UI: DARK MODE & COMPACT SIZING
                    # ==================================================
                    # 1. Force the standard Beeswarm (dot) plot
                    shap.summary_plot(plot_values, background_df, show=False, plot_type="dot")
                    
                    # 2. Grab the figure and force the exact dimensions
                    fig = plt.gcf()
                    ax = plt.gca()
                    fig.set_size_inches(10, 6)
                    
                    # 3. Aggressively strip ALL Matplotlib backgrounds
                    fig.patch.set_facecolor('none')
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor('none')
                    ax.patch.set_alpha(0.0)
                    
                    # 4. Turn text white and completely REMOVE distracting borders
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.tick_params(colors='white')
                    
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                        
                    # 5. OVERWRITE THE TRUNCATED X-AXIS LABEL
                    plt.xlabel("SHAP Value (Impact on Crash Probability)", color="white", fontsize=12)
                        
                    # 6. Render cleanly in Streamlit
                    st.pyplot(fig, transparent=True, width="stretch", bbox_inches='tight')
                    plt.clf()
                    
                # st.success("✅ SHAP analysis complete!")
                
                st.markdown("""
                ### How to Read This Chart:
                * **Horizontal Location:** Features pushed further to the right **increase** the model's fear of a crash.
                * **Color:** 
                    - 🔴 **Red dots** = Historically high value for that metric (e.g., high volatility)
                    - 🔵 **Blue dots** = Historically low value for that metric
                * **Case Study:** Notice **`Return_Skewness`**. A highly negative skew (Blue dots) pushes the SHAP value far to the right, confirming that left-tail distribution risk correctly increases the crash probability.
                
                ### Top Crash Predictors:
                1. **Composite_Stress** - Multi-factor market stress index
                2. **Composite_Stress_MA** - Smoothed stress indicator
                3. **Nifty_Drawdown** - Distance from 50-day high
                4. **Lower Partial Moments** - Tail risk metrics
                5. **Return_Skewness** - Distribution shape (crashes have negative skew)
                
                ### 🧠 Dominant Macro Variables:
                By analyzing the Random Forest core, we observe that the architecture heavily prioritizes custom NLP stress indicators (`Composite_Stress`) and structural market weakness (`Nifty_Drawdown`) over traditional, lagging volatility metrics like the India VIX.
                """)
                        
    except Exception as e:
        st.error(f"❌ SHAP Engine Error: {str(e)}")
        st.error("⚠️ This is likely due to data type issues. Check the Debug Info tab.")
        import traceback
        st.error(traceback.format_exc())

with tab3:
    st.header("🔧 Debug Information")
    st.markdown("Diagnostic information to help troubleshoot SHAP engine issues.")
    
    try:
        if st.button("Run Data Diagnostics"):
            st.info("Running diagnostics...")
            
            background_df = fetch_background_data()
            
            if background_df is not None:
                st.subheader("Data Shape & Types")
                st.write(f"Shape: {background_df.shape}")
                st.write(f"Columns: {list(background_df.columns)}")
                st.write(f"Data types:\n{background_df.dtypes}")
                
                st.subheader("Sample Values (First 5 Rows)")
                st.dataframe(background_df.head())
                
                st.subheader("Statistical Summary")
                st.dataframe(background_df.describe())
                
                st.subheader("Missing Values")
                missing = background_df.isnull().sum()
                if missing.sum() > 0:
                    st.warning(f"⚠️ Found {missing.sum()} missing values:")
                    st.write(missing[missing > 0])
                else:
                    st.success("✅ No missing values")
                
                st.subheader("Type Validation")
                all_numeric = background_df.applymap(lambda x: isinstance(x, (int, float, np.number))).all().all()
                if all_numeric:
                    st.success("✅ All values are numeric")
                else:
                    st.error("❌ Non-numeric values found!")
                    
                st.subheader("Value Ranges")
                st.write(background_df.describe())
                
            else:
                st.error("❌ Could not load data for diagnostics")
                
    except Exception as e:
        st.error(f"Diagnostic error: {e}")
        import traceback
        st.error(traceback.format_exc())

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
    st.page_link("pages/4_Backtest_Results.py", label="Backtests", icon="📊")
