# ======================================================
#                  MODEL IMPORTS
# ======================================================

import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from modules.db_manager import NewsDatabase
from modules.nlp_engine import NewsAPIClient, StressScoreCalculator
from dotenv import load_dotenv

# ============================================================================
#                    DAILY UPDATE MODULE (NEWS)
# ============================================================================

def daily_update_pipeline(api_key, provider='newsapi', 
                          db_path='data/financial_news.db'):
    """
    Main function to run daily:
    1. Fetch today's news
    2. Calculate stress score
    3. Store in database
    4. Generate predictions
    """
    
    print("\n" + "="*60)
    print("📰 DAILY NEWS UPDATE PIPELINE")
    print("="*60 + "\n")
    
    # Initialize components
    db = NewsDatabase(db_path)
    news_client = NewsAPIClient(api_key=api_key, provider=provider)
    stress_calculator = StressScoreCalculator()
    
    # Fetch today's news
    print("Fetching today's headlines...")
    headlines_df = news_client.fetch_today_headlines()
    
    if headlines_df.empty:
        print("⚠️  No headlines fetched. Check API key or rate limits.")
        return None
    
    print(f"✅ Fetched {len(headlines_df)} headlines")
    
    # Store in database
    db.insert_headlines(headlines_df)
    
    # Calculate stress score
    today = datetime.now().strftime('%Y-%m-%d')
    headlines_list = headlines_df['headline'].tolist()
    stress_score, anomalies = stress_calculator.calculate_stress(headlines_list)
    
    print(f"\n🔥 Stress Score for {today}: {stress_score:.3f}")
    if anomalies:
        print(f"⚠️  Detected {len(anomalies)} stress-related headlines:")
        for i, headline in enumerate(anomalies[:5], 1):
            print(f"   {i}. {headline[:80]}...")
    
    # Store stress score
    db.insert_stress_score(today, stress_score, len(headlines_list))
    
    # Export stress signals CSV
    stress_history = db.get_stress_history(days=5000)  # All history
    stress_history = stress_history.sort_values('date')
    stress_history.columns = ['Date', 'Stress_Score']
    stress_history.to_csv('data/macro_stress_signals.csv', index=False)
    
    print(f"\n✅ Stress signals exported")
    print(f"📊 Database now contains {len(stress_history)} days of data")
    
    db.close()
    return stress_score

# =============================================================
#                           EXECUTION
# =============================================================

if __name__ == "__main__":
    
    db_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/financial_news.db'))

    load_dotenv()
    
    # Security: Use environment variables to prevent hardcoding keys in public repositories
    api_key = os.environ.get('NEWS_API_KEY', 'YOUR_API_KEY_HERE')

    stress_score = daily_update_pipeline(
        api_key=api_key,
        provider='newsapi',
        db_path=db_absolute_path
    )
    print(f"Daily scrape complete. Final Stress Score: {stress_score}")