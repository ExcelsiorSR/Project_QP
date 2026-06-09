# ======================================================
#                  MODEL IMPORTS
# ======================================================

import pandas as pd
import yfinance as yf
import numpy as np

import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from modules.model_engine import CostSensitiveEnsemble

# ======================================================
#                  BACKTESTER MODULE
# ======================================================

def run_backtest():
    # Load model's outputs
    preds_df = pd.read_csv('data/final_predictions.csv', index_col='Date', parse_dates=True)

    # Fetch the actual Nifty data for the exact same date range
    nifty = yf.download('^NSEI', start=preds_df.index[0], end=preds_df.index[-1])
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    # Merge them cleanly
    df = pd.DataFrame(index=preds_df.index)
    df['Close'] = nifty['Close']
    df['Crash_Probability'] = preds_df['Crash_Probability']
    df['Nifty_Return'] = df['Close'].ffill().pct_change()
    
    # ==========================================
    # 4. THE DYNAMIC THRESHOLD EXTRACTION
    # ==========================================
    # Look one folder up to find the .pkl file
    model_path = os.path.join(os.path.dirname(__file__), '..', 'crash_predictor.pkl')
    engine = joblib.load(model_path)
    
    THRESHOLD = engine.optimal_threshold
    print(f"\n🎯 Dynamic Threshold Automatically Loaded: {THRESHOLD:.3f}\n")
    
    # 5. Shift predictions by 1 day to prevent look-ahead bias
    df['Signal'] = (df['Crash_Probability'] >= THRESHOLD).shift(1)

    # 1. Define the cooling-off period (e.g., stay in cash for 5 days)
    HOLDING_PERIOD = 20

    # 2. Stretch the signal forward using a rolling max
    df['Sell_Window'] = df['Signal'].rolling(window=HOLDING_PERIOD, min_periods=1).max()

    # 3. Position is 0 (Cash) if we are inside the Sell_Window, else 1 (Invested)
    df['Position'] = np.where(df['Sell_Window'] == 1, 0, 1)

    # Calculate strategy returns
    df['Strategy_Return'] = df['Position'] * df['Nifty_Return']

    # Drop NaN from the first row's pct_change
    df = df.dropna()

    # Cumulative compounded returns (starts at 1.0)
    df['Nifty_Cumulative'] = (1 + df['Nifty_Return']).cumprod()
    df['Strategy_Cumulative'] = (1 + df['Strategy_Return']).cumprod()

    print(f"Buy & Hold Final Return: {(df['Nifty_Cumulative'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Project QP Final Return: {(df['Strategy_Cumulative'].iloc[-1] - 1) * 100:.2f}%")

    # Calculate Drawdowns
    nifty_roll_max = df['Nifty_Cumulative'].cummax()
    nifty_drawdown = (df['Nifty_Cumulative'] / nifty_roll_max) - 1
    strat_roll_max = df['Strategy_Cumulative'].cummax()
    strat_drawdown = (df['Strategy_Cumulative'] / strat_roll_max) - 1

    print(f"Buy & Hold Max Drawdown: {nifty_drawdown.min() * 100:.2f}%")
    print(f"Project QP Max Drawdown: {strat_drawdown.min() * 100:.2f}%")

# =============================================================
#                           EXECUTION
# =============================================================

if __name__ == "__main__":
    run_backtest()