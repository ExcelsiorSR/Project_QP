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

from modules.db_manager import NewsDatabase


# ========================================================
#               ALTERNATIVE DATA MODULE
# ========================================================

class NewsAPIClient:

    '''
    Fetches financial news from NewsAPI.org
    '''
    
    def __init__(self, api_key=None, provider='newsapi'):
        
        self.api_key = api_key
        self.provider = provider
        
        # API endpoints
        self.endpoints = {
            'newsapi': 'https://newsapi.org/v2/everything',
        }
    
    # ============================================
    # FETCHES TODAY'S FINANCIAL NEWS HEADLINES
    # ============================================  
    def fetch_today_headlines(self):
        
        
        if self.provider == 'newsapi':
            return self._fetch_newsapi()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    # ============================================
    # CALLS NEWSAPI
    # ============================================ 
    def _fetch_newsapi(self):
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        params = {
            'apiKey': self.api_key,
            'q': '(Nifty OR Sensex OR RBI OR "Indian economy" OR "Dalal Street" OR inflation OR recession OR "Federal Reserve" OR Fed OR "Bank of Japan" OR oil OR crude OR banking OR liquidity)',
            'language': 'en',
            'from': yesterday,
            'to': today,
            'sortBy': 'relevancy',
            'pageSize': 100
        }
        
        try:
            response = requests.get(self.endpoints['newsapi'], params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                articles = data['articles']
                headlines = [{
                    'date': today,
                    'headline': article['title'],
                    'source': article['source']['name'],
                    'url': article['url']
                } for article in articles]
                
                return pd.DataFrame(headlines)
            else:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return pd.DataFrame()
    
# ========================================================
#               STRESS ENGINE MODULE
# ========================================================
class StressScoreCalculator:
    """
    Calculates daily market stress score using FinBERT
    """
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            batch_size=32,
            truncation = True,
            max_length = 128,
        )
        
        # Anchor texts for different types of market stress
        self.anchor_texts = [
            "Severe liquidity crisis, cascading margin calls, structural default, market panic",
            "Banking system collapse, credit freeze, financial contagion",
            "Market crash, circuit breakers triggered, unprecedented selloff",
            "Sovereign debt default, currency crisis, economic collapse",

            "Federal Reserve emergency rate hike, inflation shock, recession fears",
            "RBI policy tightening, liquidity squeeze, banking stress",
            "BoJ intervention, yen collapse, bond market stress",
            "Global financial crisis, credit downgrade, sovereign risk",
            "Oil shock, energy crisis, stagflation concerns",
            "Trade war escalation, sanctions, geopolitical market uncertainty"
        ]
    
    # ============================================
    # CALCULATES STESS SCORES
    # ============================================
    def calculate_stress(self, headlines_list):
        """
        Calculate stress score for a list of headlines
        Returns: stress_score (0-1), anomalies_detected
        """
        
        if not headlines_list or len(headlines_list) == 0:
            return 0.0, []
        
        # Encode headlines
        headline_embeddings = self.embedder.encode(headlines_list)
        
        # Create FAISS index
        dimension = headline_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(headline_embeddings)
        
        # Search for stress-related headlines using multiple anchors
        all_anomaly_indices = set()
        
        for anchor_text in self.anchor_texts:
            anchor_vector = self.embedder.encode([anchor_text])
            distances, indices = index.search(anchor_vector, k=min(5, len(headlines_list)))
            
            # Only consider headlines with similarity above threshold
            for dist, idx in zip(distances[0], indices[0]):
                if dist < 2.0:  # Similarity threshold (lower distance = higher similarity)
                    all_anomaly_indices.add(idx)
        
        if len(all_anomaly_indices) == 0:
            return 0.0, []
        
        # Get unique anomalies
        anomalies = [headlines_list[idx] for idx in sorted(all_anomaly_indices)]
        
        # Run FinBERT on anomalies
        stress_results = self.finbert(
            anomalies[:50],
            batch_size = 32)  # Limit to avoid API issues
        
        # Calculate aggregate stress
        financial_keywords = [
            'market',
            'stock',
            'stocks',
            'equity',
            'equities',
            'bank',
            'banking',
            'rbi',
            'reserve bank',
            'federal reserve',
            'fed',
            'boj',
            'bank of japan',
            'ecb',
            'inflation',
            'interest rate',
            'rate hike',
            'recession',
            'economy',
            'economic',
            'nifty',
            'sensex',
            'bond',
            'yield',
            'currency',
            'forex',
            'rupee',
            'dollar',
            'oil',
            'crude',
            'debt',
            'default',
            'liquidity',
            'trade',
            'tariff',
            'sanctions',
            'gdp',
            'unemployment'
        ]

        negative_scores = [
            result['score']
            for headline, result in zip(anomalies, stress_results)
            if result['label'] == 'negative'
            and any(
                keyword in headline.lower()
                for keyword in financial_keywords
            )
        ]
        
        if negative_scores:
            # Use mean of negative scores as stress indicator
            stress_score = np.mean(negative_scores)
        else:
            stress_score = 0.0
        
        return stress_score, anomalies
    
    # ============================================
    # RECALCULATES HITORICAL STRESS SCORES
    # ============================================
    def batch_calculate_historical(self, db: NewsDatabase, start_date, end_date):
        """
        Recalculate stress scores for historical data in database
        Useful for improving the model with new data
        """
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            headlines = db.get_headlines_by_date(date_str)
            
            if headlines:
                stress_score, _ = self.calculate_stress(headlines)
                db.insert_stress_score(date_str, stress_score, len(headlines))
                print(f"Processed {date_str}: {len(headlines)} headlines, stress={stress_score:.3f}")
                time.sleep(0.1)  # Rate limiting