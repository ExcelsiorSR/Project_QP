# ======================================================
#                  MODEL IMPORTS
# ======================================================

import pandas as pd
import numpy as np
import os
import sys
import sqlite3
import joblib
import yfinance as yf
from datetime import datetime
import time
import joblib
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from modules.data_pipeline import IndianMarketFeatureEngine
from modules.db_manager import NewsDatabase
from modules.nlp_engine import NewsAPIClient, StressScoreCalculator



# ==================================================
# DAILY LIVE CRASH PROBABILITY
# ==================================================

today = datetime.now().strftime('%Y-%m-%d')

conn = sqlite3.connect("data/financial_news.db")

query = """
SELECT stress_score
FROM stress_scores
ORDER BY date DESC
LIMIT 1
"""

stress_score = pd.read_sql_query(
    query,
    conn
).iloc[0]["stress_score"]

conn.close()

try:

    model = joblib.load("crash_predictor.pkl")

    engine = IndianMarketFeatureEngine(
            start_date="2024-01-01",
            end_date=today
        )

    latest_df = engine.get_final_dataset()

    latest_features = latest_df.drop('Target_Regime',axis=1,errors='ignore').iloc[-1:].copy()

    latest_features['Stress_Score'] = stress_score

    latest_features = (latest_features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0))
    nan_cols = latest_features.columns[
    latest_features.isna().any()
        ]

    ensemble_probs, _ = model.predict_proba(latest_features)

    crash_prob = float(ensemble_probs[0])

except Exception as e:

    print(f"Crash probability generation failed: {e}")

    crash_prob = np.nan

# ==================================================
# DAILY LIVE RISK SNAPSHOT
# ==================================================

try:

        # India VIX
    vix = yf.download(
            '^INDIAVIX',
            period='1d',
            progress=False
        )

    if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

    vix_value = float(vix['Close'].iloc[-1])

except:
        vix_value = np.nan

try:

    crude = yf.download(
            'BZ=F',
            period='1d',
            progress=False
        )

    if isinstance(crude.columns, pd.MultiIndex):
            crude.columns = crude.columns.get_level_values(0)

    crude_value = float(crude['Close'].iloc[-1])

except:
    crude_value = np.nan

try:

    usd = yf.download(
            'INR=X',
            period='1d',
            progress=False
        )

    if isinstance(usd.columns, pd.MultiIndex):
            usd.columns = usd.columns.get_level_values(0)

    usd_value = float(usd['Close'].iloc[-1])

except:
    usd_value = np.nan

# ==================================================
# UPDATE LIVE RISK HISTORY
# ==================================================

history_path = "data/live_risk_history.csv"

new_row = pd.DataFrame([{
        "Date": today,
        "Crash_Probability": crash_prob,
        "Stress_Score": stress_score,
        "India_VIX": vix_value,
        "Crude_Oil": crude_value,
        "USDINR": usd_value
    }])

if os.path.exists(history_path):

    history_df = pd.read_csv(history_path)

    history_df = pd.concat(
            [history_df, new_row],
            ignore_index=True
        )

    history_df.drop_duplicates(
            subset=["Date"],
            keep="last",
            inplace=True
        )

else:

    history_df = new_row

    history_df.to_csv(
        history_path,
        index=False
        )