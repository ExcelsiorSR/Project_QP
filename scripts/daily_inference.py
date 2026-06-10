# ======================================================
#                  MODEL IMPORTS
# ======================================================

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_pipeline import IndianMarketFeatureEngine

# ======================================================
#                  UPDATION MODULE
# ======================================================

def run_production_inference():
    print("🔄 Starting Live Inference Pipeline...\n")

    # GET LAST PREDICTION DATE
    try:
        preds_df = pd.read_csv("data/final_predictions.csv", index_col=0, parse_dates=True)
        last_date = preds_df.index[-1]
        print(f"✅ Last Prediction Date -> {last_date.date()}")
    except Exception as e:
        print(f"❌ Could not read final_predictions.csv. Error: {e}")
        return

    # FETCH DATA (75-DAY LOOKBACK)
    lookback_date = last_date - timedelta(days=75)
    lookback_str = lookback_date.strftime('%Y-%m-%d')
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📡 Initializing Market Engine ({lookback_str} to {today_str})")
    try:
        feature_engine = IndianMarketFeatureEngine(start_date=lookback_str, end_date=today_str)
        feature_engine.fetch_all_market_data()
        print(f"✅ Market Data Extracted.")
    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        return

    # ENGINEER FEATURES & INTEGRATE NLP STRESS
    print("\n⚙️ Engineering Features & Merging NLP...")
    try:
        master_df = feature_engine.engineer_comprehensive_features()
        
        # Merge the NLP Stress Score from the scraper's CSV output
        if os.path.exists('macro_stress_signals.csv'):
            stress_df = pd.read_csv('macro_stress_signals.csv')
            stress_df['Date'] = pd.to_datetime(stress_df['Date'])
            stress_df.set_index('Date', inplace=True)
            
            master_df = master_df.merge(stress_df[['Stress_Score']], left_index=True, right_index=True, how='left')
            master_df['Stress_Score'] = master_df['Stress_Score'].ffill().fillna(0.0)
            print("✅ NLP Stress Score integrated.")
        else:
            print("⚠️ macro_stress_signals.csv not found. Defaulting Stress_Score to 0.0")
            master_df['Stress_Score'] = 0.0

        # Apply exact exclusion logic
        exclude_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
            'Panic_Intensity', 'Crash_Signal_Count', 'Target_Regime',
            'Crude_Oil', 'Gold_Price', 'India_VIX', 'US_VIX', 'INR_USD', 'SPY_Return'
        ]
        feature_cols = [col for col in master_df.columns if col not in exclude_cols]
        engineered_df = master_df[feature_cols]
        
    except Exception as e:
        print(f"❌ Feature Engineering crashed. Error: {e}")
        return

    # STEP 5: LOAD MODEL & FORCE EXACT FEATURE ALIGNMENT
    print("\n🔀 Enforcing Institutional Feature Alignment...")
    try:
        ensemble = joblib.load('crash_predictor.pkl')
        
        # Extract the exact feature list from the Scaler (which remembers the string names)
        expected_features = list(ensemble.scaler.feature_names_in_)
        
        # BULLETPROOFING: If Yahoo Finance failed to fetch a sector,
        # this safely fills those missing columns with 0.0 so the model doesn't crash.
        for col in expected_features:
            if col not in engineered_df.columns:
                print(f"⚠️ API Glitch Warning: Missing feature '{col}'. Auto-filling to bypass network error.")
                engineered_df[col] = 0.0
                
        # Force the dataframe to have the exact columns in the exact order
        engineered_df = engineered_df[expected_features]
        print("✅ Matrix perfectly aligned with Meta-Learner.")
        
    except Exception as e:
        print(f"❌ Model Loading/Alignment Error: {e}")
        return

    # STEP 6: SLICE THE DELTA AND RUN INFERENCE
    delta_matrix = engineered_df[engineered_df.index > last_date]
    delta_matrix = delta_matrix.ffill().fillna(0.0)
    
    if len(delta_matrix) == 0:
        print("\n✅ System is already up to date. No new trading days found.")
        return

    print(f"\n🧠 Running Model Inference on {len(delta_matrix)} days...")
    try:
        # Predict the probability of Class 1 (Crash)
        probabilities, individual_model_probs = ensemble.predict_proba(delta_matrix)
        
        new_preds = pd.DataFrame({'Crash_Probability': probabilities}, index=delta_matrix.index)
        print(f"✅ Max Crash Risk in new window: {probabilities.max()*100:.2f}%")

    except Exception as e:
        print(f"❌ Inference error: {e}")
        return

    # STEP 7: UPDATE THE DATABASE
    print("\n💾 Updating the predictions database...")
    try:
        updated_preds = pd.concat([preds_df, new_preds])
        updated_preds = updated_preds[~updated_preds.index.duplicated(keep='last')]
        updated_preds = updated_preds.sort_index()
        
        updated_preds.to_csv("final_predictions.csv")
        
        print(f"✅ Database extended to {updated_preds.index[-1].date()}")
        print("\n🎉 LIVE INFERENCE PIPELINE FULLY OPERATIONAL!")
        
    except Exception as e:
        print(f"❌ Could not save CSV: {e}")

# =============================================================
#                           EXECUTION
# =============================================================

if __name__ == "__main__":
    run_production_inference()