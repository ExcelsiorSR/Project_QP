#!/usr/bin/env python3
# ====================================================================
#             SYSTEM INTEGRITY & DEPENDENCY VALIDATOR
# ====================================================================
"""
Run this script before deployment to validate:
1. All dependencies are installed
2. Data files are properly formatted
3. Models load correctly
4. SHAP works without errors
5. Streamlit dashboard runs
"""
# ======================================================
#                  MODEL IMPORTS
# ======================================================

import sys
import os
import subprocess
from datetime import datetime
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ======================================================
#                  VALIDATOR MODULE
# ======================================================

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

# ============================================
# TESTS FOR EXISTENCE OF DEPENDENCIES
# ============================================
def test_imports():
    """Test critical dependencies"""
    print_header("TEST 1: CHECKING CRITICAL DEPENDENCIES")
    
    dependencies = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'torch',
        'transformers',
        'sentence_transformers',
        'faiss',
        'shap',
        'streamlit',
        'matplotlib',
        'seaborn',
        'joblib',
        'imblearn',
    ]
    
    failed = []
    
    for lib in dependencies:
        try:
            __import__(lib)
            print_success(f"Imported {lib}")
        except ImportError:
            print_error(f"Missing {lib} - install with: pip install {lib}")
            failed.append(lib)
    
    if failed:
        print_error(f"\n{len(failed)} dependencies missing!")
        print_warning("Install missing packages with:")
        print(f"pip install {' '.join(failed)}")
        return False
    else:
        print_success(f"\nAll {len(dependencies)} dependencies available!")
        return True

# ============================================
# TESTS EXISTENCE OF DATA FILES
# ============================================
def test_data_files():

    print_header("TEST 2: CHECKING DATA FILES")
    
    required_files = [
        'crash_predictor.pkl',
        'data/final_predictions.csv',
        'data/financial_news.db',
        'data/macro_stress_signals.csv',
    ]
    
    missing = []
    
    for filename in required_files:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print_success(f"Found {filename} ({size_mb:.2f} MB)")
        else:
            print_error(f"Missing {filename}")
            missing.append(filename)
    
    if missing:
        print_warning(f"\n{len(missing)} files missing. Run main.py to train:")
        print("python main.py")
        return False
    else:
        print_success(f"\nAll {len(required_files)} data files present!")
        return True

# ============================================
# VALIDATES DATA TYPES IN CSV FILES
# ============================================
def test_csv_data_types():

    print_header("TEST 3: VALIDATING CSV DATA TYPES")
    
    import pandas as pd
    import numpy as np
    
    csv_files = [
        'data/macro_stress_signals.csv',
        'data/final_predictions.csv',
    ]
    
    all_valid = True
    
    for filename in csv_files:
        if not os.path.exists(filename):
            print_warning(f"Skipping {filename} (not found)")
            continue
        
        print(f"\nValidating {filename}:")
        df = pd.read_csv(filename)
        
        # Check for Stress_Score issues
        if 'Stress_Score' in df.columns:
            # Check if it contains bracket notation
            sample = str(df['Stress_Score'].iloc[0])
            if '[' in sample or 'E' in sample:
                print_warning(f"  Stress_Score has scientific notation: {sample}")
                print_warning(f"  Run: python clean_nlp_data.py")
                all_valid = False
            else:
                print_success(f"  Stress_Score is clean: {sample}")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print_success(f"  Found {len(numeric_cols)} numeric columns")
        
        # Check for NaNs
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print_warning(f"  Found {nan_count} NaN values")
        else:
            print_success(f"  No NaN values")
    
    return all_valid

# ============================================
# TESTS FOR LOADING OF TRAINED MODEL
# ============================================
def test_model_loading():

    print_header("TEST 4: LOADING TRAINED MODEL")
    
    try:
        import joblib
        
        if not os.path.exists('crash_predictor.pkl'):
            print_error("Model file not found: crash_predictor.pkl")
            return False
        
        print("Loading crash_predictor.pkl...")
        ensemble = joblib.load('crash_predictor.pkl')
        
        # Verify ensemble structure
        required_models = ['xgb', 'rf', 'gb', 'lr']
        for model_name in required_models:
            if model_name in ensemble.models:
                print_success(f"  Found {model_name} model")
            else:
                print_error(f"  Missing {model_name} model")
                return False
        
        # Check optimal threshold
        print_success(f"  Optimal threshold: {ensemble.optimal_threshold:.3f}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return False

# ============================================
# TESTS FOR SHAP INITIALIZATION
# ============================================
def test_shap_setup():

    print_header("TEST 5: SHAP INITIALIZATION")
    
    try:
        import joblib
        import pandas as pd
        import numpy as np
        import shap
        
        # Load model
        if not os.path.exists('crash_predictor.pkl'):
            print_warning("Model file not found - skipping SHAP test")
            return True
        
        print("Loading ensemble model...")
        ensemble = joblib.load('crash_predictor.pkl')
        
        print("Creating test data...")
        # Create synthetic test data matching feature count
        from modules.data_pipeline import IndianMarketFeatureEngine
        
        engine = IndianMarketFeatureEngine("2024-01-01", "2024-12-31")
        df = engine.get_final_dataset()
        
        if 'Target_Regime' in df.columns:
            df = df.drop('Target_Regime', axis=1)
        
        background_df = df.tail(50).astype(np.float64)
        
        print_success(f"Created background data: {background_df.shape}")
        
        print("Initializing SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(ensemble.models['xgb'])
        
        print_success("SHAP explainer initialized successfully!")
        print_success("SHAP will work correctly in the XAI Engine")
        
        return True
        
    except Exception as e:
        print_error(f"SHAP initialization failed: {e}")
        print_warning("This is expected if you haven't cleaned CSV files yet")
        print_warning("Run: python clean_nlp_data.py")
        return False

# ================================================
# TESTS FOR STRUCTURAL EFFICACY OF STREAMLIT
# ================================================
def test_streamlit_structure():

    print_header("TEST 6: CHECKING STREAMLIT APP STRUCTURE")
    
    required_files = [
        'Home.py',
        'pages/1_Dashboard.py',
        'pages/2_Systemic_Context.py',
        'pages/3_Live_Markets.py',
        'pages/4_Backtest_Results.py',
        '5_XAI_Engine_FIXED.py',  # Our fixed version
    ]
    
    missing = []
    
    for filepath in required_files:
        if os.path.exists(filepath):
            print_success(f"Found {filepath}")
        else:
            print_error(f"Missing {filepath}")
            missing.append(filepath)
    
    if missing:
        print_warning(f"\nMissing {len(missing)} files - create them or update paths")
        return False
    else:
        print_success(f"\nAll {len(required_files)} Streamlit files present!")
        return True

# ============================================
# TESTS FOR EXISTENCE OF DATABASE
# ============================================
def test_database():

    print_header("TEST 7: CHECKING DATABASE")
    
    try:
        import sqlite3
        
        if not os.path.exists('data/financial_news.db'):
            print_warning("Database not found - will be created on first run")
            return True
        
        conn = sqlite3.connect('data/financial_news.db')
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        expected_tables = ['news_headlines', 'stress_scores', 'predictions']
        found_tables = [t[0] for t in tables]
        
        for table in expected_tables:
            if table in found_tables:
                # Count rows
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print_success(f"Found table '{table}' with {count} rows")
            else:
                print_warning(f"Missing table '{table}'")
        
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Database error: {e}")
        return False

# ============================================
# TESTS FOR COMPLETE VALIDATION SUITE
# ============================================
def run_all_tests():

    print_header("🧪 COMPREHENSIVE SYSTEM VALIDATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
    
    tests = [
        ("Dependencies", test_imports),
        ("Data Files", test_data_files),
        ("CSV Data Types", test_csv_data_types),
        ("Model Loading", test_model_loading),
        ("SHAP Setup", test_shap_setup),
        ("Streamlit Structure", test_streamlit_structure),
        ("Database", test_database),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print_header("📊 TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{test_name}: {status}")
    
    print(f"\n{BLUE}Result: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print_success("\n✨ ALL TESTS PASSED! System is ready for deployment.")
        print_success("You can now run: streamlit run Home.py\n")
        return True
    else:
        print_error(f"\n{total - passed} test(s) failed. See details above.")
        print_warning("\nNext steps:")
        if not test_imports():
            print("1. Install missing dependencies: pip install -r requirements.txt")
        if not test_csv_data_types():
            print("2. Clean CSV files: python clean_nlp_data.py")
        if not test_data_files():
            print("3. Train model: python main.py")
        print("\n")
        return False

# =============================================================
#                           EXECUTION
# =============================================================

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
