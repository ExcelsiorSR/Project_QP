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

# ==================================================================
#                          DATABASE MODULE
# ==================================================================

class NewsDatabase:
    """
    SQLite database for storing financial news headlines
    Enables historical tracking and model improvement over time
    """
    
    def __init__(self, db_path='data/financial_news.db'):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    # ==========================================
    # CREATES TABLES
    # ==========================================
    def initialize_database(self):

        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # News headlines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_headlines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                headline TEXT NOT NULL,
                source TEXT,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, headline)
            )
        ''')
        
        # Stress scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stress_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                stress_score REAL NOT NULL,
                num_headlines INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model predictions table (for tracking performance)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                crash_probability REAL NOT NULL,
                predicted_regime INTEGER,
                actual_regime INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        print("✅ Database initialized successfully")
    
    # ==========================================
    # INSERTS NEWS HEADLINES INTO DATABASE
    # ==========================================
    def insert_headlines(self, headlines_df):

        cursor = self.conn.cursor()
        
        for _, row in headlines_df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_headlines (date, headline, source, url)
                    VALUES (?, ?, ?, ?)
                ''', (row['date'], row['headline'], row.get('source', ''), row.get('url', '')))
            except Exception as e:
                print(f"Error inserting headline: {e}")
                continue
        
        self.conn.commit()
        print(f"✅ Inserted {len(headlines_df)} headlines")
    
    # ==========================================
    # INSERTS DAILY STRESS SCORE
    # ==========================================
    def insert_stress_score(self, date, stress_score, num_headlines):

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO stress_scores (date, stress_score, num_headlines)
            VALUES (?, ?, ?)
        ''', (date, stress_score, num_headlines))
        self.conn.commit()
    
    # ==========================================
    # RETRIEVES HEADLINES FOR A SPECIFIC DATE
    # ==========================================
    def get_headlines_by_date(self, date):
        
        query = "SELECT headline FROM news_headlines WHERE date = ?"
        df = pd.read_sql_query(query, self.conn, params=(date,))
        return df['headline'].tolist()
    
    def get_stress_history(self, days=30):
        """Get stress score history"""
        query = f"SELECT date, stress_score FROM stress_scores ORDER BY date DESC LIMIT {days}"
        return pd.read_sql_query(query, self.conn)
    
    def close(self):
        if self.conn:
            self.conn.close()