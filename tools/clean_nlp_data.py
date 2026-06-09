# ======================================================
#     DATA CLEANING UTILITY FOR XAIENGINE COMPATIBILITY
# ======================================================
"""
This script fixes the Stress_Score columns in your historical CSV files
to be SHAP-compatible. Run this once at startup.
"""

# ======================================================
#                  MODEL IMPORTS
# ======================================================

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ======================================================
#                  CLEANER MODULE
# ======================================================

# ============================================
# CLEANS STRESS SCORES
# ============================================
def clean_stress_score_column(value):

    if pd.isna(value) or value is None:
        return 0.0
    
    # Convert to string
    s = str(value).strip()
    
    # Handle empty strings
    if s in ['', 'nan', 'NaN', 'None', 'null', 'NULL']:
        return 0.0
    
    # Remove all brackets, quotes, parentheses
    s = s.replace('[', '').replace(']', '')
    s = s.replace('(', '').replace(')', '')
    s = s.replace('{', '').replace('}', '')
    s = s.replace('"', '').replace("'", '')
    s = s.strip()
    
    # Try to convert to float
    try:
        return float(s)
    except ValueError:
        return 0.0

# ============================================
# CLEANS CSV FILE STRESS SCORES
# ============================================
def clean_csv_stress_scores(csv_path, output_path=None):
    """
    Clean a CSV file's Stress_Score column
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file to clean
    output_path : str, optional
        Path to save the cleaned CSV. If None, overwrites original.
    
    Returns:
    --------
    pd.DataFrame
        The cleaned dataframe
    """
    
    print(f"\n📖 Reading {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    # Check if Stress_Score column exists
    if 'Stress_Score' not in df.columns:
        print("⚠️  No Stress_Score column found. Skipping.")
        return df
    
    print("🔧 Cleaning Stress_Score column...")
    
    # Store original for comparison
    original_count = df['Stress_Score'].isna().sum()
    
    # Apply cleaning
    df['Stress_Score'] = df['Stress_Score'].apply(clean_stress_score_column)
    
    # Verify cleaning
    new_count = df['Stress_Score'].isna().sum()
    print(f"   Original NaN count: {original_count}")
    print(f"   After cleaning NaN count: {new_count}")
    
    # Statistics
    print(f"   Value range: {df['Stress_Score'].min():.6f} to {df['Stress_Score'].max():.6f}")
    print(f"   Mean: {df['Stress_Score'].mean():.6f}")
    
    # Ensure all values are valid floats
    assert df['Stress_Score'].dtype in [np.float32, np.float64], \
        f"Expected float, got {df['Stress_Score'].dtype}"
    
    # Save cleaned version
    if output_path is None:
        output_path = csv_path
    
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}\n")
    
    return df

# ============================================
# CLEANS NLP FILES
# ============================================
def clean_all_nlp_files():

    
    print("\n" + "="*60)
    print("🚀 CLEANING ALL NLP STRESS CSV FILES")
    print("="*60 + "\n")
    
    nlp_files = [
        'macro_stress_signals.csv',
        'macro_stress_signals.csv',
        'data/historical_nlp_stress.csv',
        'data/historical_nlp_stress_rg.csv',
    ]
    
    cleaned_count = 0
    
    for filename in nlp_files:
        if os.path.exists(filename):
            print(f"Processing: {filename}")
            try:
                clean_csv_stress_scores(filename)
                cleaned_count += 1
            except Exception as e:
                print(f"❌ Error cleaning {filename}: {e}\n")
        else:
            print(f"⏭️  Skipping {filename} (not found)\n")
    
    print("="*60)
    print(f"✅ Successfully cleaned {cleaned_count} files")
    print("="*60 + "\n")

# ============================================
# VALIDATES DATA TYPES OF ALL CSV FILES
# ============================================
def validate_all_data_types():

    print("\n" + "="*60)
    print("🔍 VALIDATING DATA TYPES")
    print("="*60 + "\n")
    
    nlp_files = [
        'macro_stress_signals.csv',
        'macro_stress_signals.csv',
        'data/historical_nlp_stress.csv',
        'data/historical_nlp_stress_rg.csv',
    ]
    
    for filename in nlp_files:
        if os.path.exists(filename):
            print(f"Validating: {filename}")
            df = pd.read_csv(filename)
            
            # Check Stress_Score
            if 'Stress_Score' in df.columns:
                dtype = df['Stress_Score'].dtype
                nan_count = df['Stress_Score'].isna().sum()
                min_val = df['Stress_Score'].min()
                max_val = df['Stress_Score'].max()
                
                status = "✅" if dtype in [np.float32, np.float64] else "❌"
                print(f"  {status} Stress_Score dtype: {dtype}")
                print(f"      NaN count: {nan_count} / {len(df)}")
                print(f"      Range: [{min_val:.6f}, {max_val:.6f}]")
            else:
                print(f"  ⏭️  No Stress_Score column")
            
            print()

# =============================================================
#                           EXECUTION
# =============================================================

if __name__ == "__main__":
    # Run the cleaning
    clean_all_nlp_files()
    
    # Validate results
    validate_all_data_types()
    
    print("\n✅ Data cleaning complete! You can now run the XAI Engine without errors.\n")
