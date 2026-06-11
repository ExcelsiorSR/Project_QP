# ======================================================
#                  MODEL IMPORTS
# ======================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import subprocess
import sys
import os
from datetime import datetime, timedelta
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from modules.data_pipeline import IndianMarketFeatureEngine
from modules.model_engine import CostSensitiveEnsemble

# =============================================================
#                    NEWS SCRAPER MODULE
# =============================================================

def run_scraper_safely():
    print("🌐 [ORCHESTRATOR] Step 1: Initiating Live NLP Data Extraction...")
    try:
        # This virtually types "python scripts/news_scraper.py" into the terminal
        subprocess.run([sys.executable, "scripts/news_scraper.py"], check=True)
        print("✅ [ORCHESTRATOR] NLP Extraction Complete. Database Updated.")
    except Exception as e:
        print(f"⚠️ [ORCHESTRATOR] Warning: Scraper encountered network resistance. Falling back to existing historical database. Error: {e}")

run_scraper_safely()


# =============================================================
#                    MAIN TRAINING PIPELINE
# =============================================================

def train_model():
    """Complete training pipeline with features"""
    
    print("\n" + "="*70)
    print("🚀 MARKET CRASH PREDICTION - TRAINING PIPELINE")
    print("="*70 + "\n")


    today = datetime.now()

    # Monday=0 ... Sunday=6
    days_since_friday = (today.weekday() - 4) % 7

    if days_since_friday == 0:
        days_since_friday = 7

    last_friday = today - timedelta(days=days_since_friday)

    end_date = last_friday.strftime("%Y-%m-%d")
    
    
    # 1. Create dataset
    print("Step 1: Creating dataset...")
    engine = IndianMarketFeatureEngine(
        start_date="2008-01-01",
        end_date=end_date
    )
    master_df = engine.get_final_dataset()
    
    # 2. Add NLP stress scores if available
    try:
        print("Step 2b: Integrating historical NLP datasets...")
        
        # Load Kaggle (Broad News)
        kaggle_nlp = pd.read_csv('data/historical_nlp_stress.csv')
        kaggle_nlp['Date'] = pd.to_datetime(kaggle_nlp['Date'])
        kaggle_nlp.set_index('Date', inplace=True)
        
        # Load ResearchGate (Financial News)
        rg_nlp = pd.read_csv('data/historical_nlp_stress_rg.csv')
        rg_nlp['Date'] = pd.to_datetime(rg_nlp['Date'])
        rg_nlp.set_index('Date', inplace=True)
        
        # Merge the two NLP dataframes side-by-side
        combined_nlp = kaggle_nlp.join(rg_nlp, how='outer', lsuffix='_kaggle', rsuffix='_rg')
        
        # Calculate the mathematical average to create the Composite Sentiment
        # We name it 'Stress_Score' to perfectly match your live scraper's output!
        combined_nlp['Stress_Score'] = combined_nlp[['Stress_Score_kaggle', 'Stress_Score_rg']].mean(axis=1)
        
        # Join the averaged score into the main market matrix
        master_df = master_df.merge(combined_nlp[['Stress_Score']], left_index=True, right_index=True, how='left')
        
        # Forward fill the sentiment for weekends/holidays
        master_df['Stress_Score'] = master_df['Stress_Score'].ffill().fillna(0.0)
        print("✅ Composite averaged NLP stress signals integrated successfully")
        
    except Exception as e:
        print(f"⚠️  NLP integration failed: {e}. Continuing without NLP.")
        master_df['Stress_Score'] = 0.0
     
    master_df = master_df.dropna()
    
    # 3. Split data
    train_size = int(len(master_df) * 0.7)
    val_size = int(len(master_df) * 0.15)
    
    train_df = master_df.iloc[:train_size]
    val_df = master_df.iloc[train_size:train_size+val_size]
    test_df = master_df.iloc[train_size+val_size:]
    
    X_train = train_df.drop('Target_Regime', axis=1)
    y_train = train_df['Target_Regime']
    
    X_val = val_df.drop('Target_Regime', axis=1)
    y_val = val_df['Target_Regime']
    
    X_test = test_df.drop('Target_Regime', axis=1)
    y_test = test_df['Target_Regime']
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples\n")
    
    # 4. Apply SMOTE to training data
    print("Step 2: Applying SMOTE-Tomek resampling...")
    sampler = SMOTETomek(
        smote=SMOTE(sampling_strategy=0.6, k_neighbors=3, random_state=42),
        random_state=42
    )
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
    print(f"Original: {np.bincount(y_train.astype(int))}")
    print(f"Resampled: {np.bincount(y_train_resampled.astype(int))}\n")
    
    # 5. Train cost-sensitive ensemble
    print("Step 3: Training cost-sensitive ensemble...")
    ensemble = CostSensitiveEnsemble(fn_cost=5, fp_cost=3)
    ensemble.train(X_train_resampled, y_train_resampled)
    
    # 6. Find optimal threshold
    print("\nStep 4: Optimizing decision threshold...")
    optimal_threshold = ensemble.find_cost_sensitive_threshold(X_val, y_val)
    
    # 7. Evaluate on test set
    print("Step 5: Evaluating on test set...")
    predictions, probabilities = ensemble.evaluate(X_test, y_test, threshold=optimal_threshold)
    
    # 8. Generate plots
    ensemble.plot_threshold_analysis(X_val, y_val)
    ensemble.plot_feature_importance(X_train.columns.tolist())
    
    # 9. Save results
    results_df = pd.DataFrame({
        'Date': test_df.index,
        'True_Label': y_test.values,
        'Predicted_Label': predictions,
        'Crash_Probability': probabilities
    })
    results_df.to_csv('data/model_validation_results.csv', index=False)
    
    # 10. Save model
    joblib.dump(ensemble, 'crash_predictor.pkl')
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("📁 Saved files:")
    print("   • crash_predictor.pkl (trained model)")
    print("   • data/model_validation_results.csv (test results)")
    print("   • feature_importance.png")
    print("   • threshold_analysis.png")
    
    return ensemble, results_df


# =============================================================
#                           EXECUTION
# =============================================================

if __name__ == "__main__":
    # Explicitly point the trainer to live, NLP database
    ensemble, results = train_model()

    # ==========================================
    #       MLOPS STEP: AUTO-BACKTEST
    # ==========================================
    def run_backtest_safely():
        print("\n📊 [ORCHESTRATOR] Step 5: Initiating Historical Backtest Simulation...")
        try:
            subprocess.run([sys.executable, "scripts/backtest_simulator.py"], check=True)
            print("✅ [ORCHESTRATOR] Backtest Complete. Performance metrics updated.")
        except Exception as e:
            print(f"⚠️ [ORCHESTRATOR] Warning: Backtest Simulator failed. Error: {e}")

    # Trigger the backtest immediately after training finishes
    run_backtest_safely()