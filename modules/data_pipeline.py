# ======================================================
#                  MODEL IMPORTS
# ======================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import yfinance as yf
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
#                   DATA ENGINE - INDIAN MARKET FEATURES
# ================================================================================

class IndianMarketFeatureEngine:
    """
    Feature engineering specifically for Indian markets
    Includes VIX, FII/DII data, sector rotation and global correlations.
    """
    
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.market_data = {}

    # ========================================    
    # FETCHES COMPREHENSIVE MARKET DATA 
    # ========================================    
    def fetch_all_market_data(self):
        
        print("\n" + "="*60)
        print("📊 FETCHING COMPREHENSIVE INDIAN MARKET DATA")
        print("="*60 + "\n")
        
        # Dictionary of tickers to fetch
        tickers = {
            'nifty': '^NSEI', #Nifty
            'bank_nifty': '^NSEBANK', #BankNifty
            'nifty_it': '^CNXIT',  # IT sector
            'nifty_pharma': '^CNXPHARMA',  # Pharma sector
            'india_vix': '^INDIAVIX',  # India VIX
            'us_spy': 'SPY',  # S&P 500 (global correlation)
            'us_vix': '^VIX',  # US VIX (global fear gauge)
            'dxy': 'DX-Y.NYB',  # Dollar Index (affects FII flows)
            'crude_oil': 'CL=F',  # Crude oil (India imports)
            'gold': 'GC=F',  # Gold (safe haven indicator)
            'inr_usd': 'INR=X',  # INR/USD exchange rate
        }
        
        for name, ticker in tickers.items():
            try:
                print(f"Fetching {name} ({ticker})...")
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                
                # To fix MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                self.market_data[name] = data
                print(f"  ✅ {name}: {len(data)} days")
            except Exception as e:
                print(f"  ⚠️  Failed to fetch {name}: {e}")
                self.market_data[name] = pd.DataFrame()
        
        return self
    
    # ========================================
    # CREATES COMPREHENSIVE FEATURE SET
    # ========================================
    def engineer_comprehensive_features(self):
        """Create a comprehensive feature set"""
        
        print("\n🔧 Engineering comprehensive features...")
        
        # Start with Nifty as the base
        base_df = self.market_data['nifty'].copy()
        

        base_df = self._add_basic_features(base_df, 'Nifty') # SECTION 1: BASIC MARKET FEATURES
        
        
        base_df = self._add_tail_risk_features(base_df) # SECTION 2: ADVANCED TAIL RISK FEATURES
        

        # SECTION 3: INDIA VIX FEATURES
        
        if not self.market_data['india_vix'].empty:
            vix_df = self.market_data['india_vix'][['Close']].rename(columns={'Close': 'India_VIX'})
            base_df = base_df.join(vix_df, how='left')
            base_df['India_VIX'] = base_df['India_VIX'].ffill()
            
            # VIX-based features
            base_df['VIX_Change'] = base_df['India_VIX'].pct_change()
            base_df['VIX_MA_5'] = base_df['India_VIX'].rolling(5).mean()
            base_df['VIX_MA_20'] = base_df['India_VIX'].rolling(20).mean()
            base_df['VIX_Spike'] = base_df['India_VIX'] / base_df['VIX_MA_20']
            base_df['VIX_Acceleration'] = base_df['VIX_Change'].diff()
            
            print("  ✅ India VIX features added")
        

        # SECTION 4: SECTOR ROTATION FEATURES
        
        # Bank Nifty
        if not self.market_data['bank_nifty'].empty:
            bank_features = self._create_sector_features(
                self.market_data['bank_nifty'], 
                'Bank'
            )
            base_df = base_df.join(bank_features, how='left')
            
            # Bank-Nifty divergence (early warning signal)
            base_df['Bank_Nifty_Divergence'] = (
                base_df['Bank_Return'] - base_df['Nifty_Return']
            )
            print("  ✅ Bank Nifty features added")
        
        # IT Sector
        if not self.market_data['nifty_it'].empty:
            it_features = self._create_sector_features(
                self.market_data['nifty_it'], 
                'IT'
            )
            base_df = base_df.join(it_features, how='left')
            print("  ✅ IT Sector features added")
        
        # Pharma Sector (defensive)
        if not self.market_data['nifty_pharma'].empty:
            pharma_features = self._create_sector_features(
                self.market_data['nifty_pharma'], 
                'Pharma'
            )
            base_df = base_df.join(pharma_features, how='left')
            
            # Defensive rotation indicator
            base_df['Defensive_Rotation'] = (
                base_df['Pharma_Return'] - base_df['Nifty_Return']
            )
            print("  ✅ Pharma Sector features added")
        

        # SECTION 5: GLOBAL MARKET CORRELATION
        
        # US Market correlation
        if not self.market_data['us_spy'].empty:
            spy_returns = self.market_data['us_spy']['Close'].pct_change()
            base_df['SPY_Return'] = spy_returns
            
            # Rolling correlation
            base_df['US_India_Correlation'] = (
                base_df['Nifty_Return']
                .rolling(20)
                .corr(base_df['SPY_Return'])
            )
            print("  ✅ US market correlation added")
        
        # US VIX (global fear)
        if not self.market_data['us_vix'].empty:
            us_vix = self.market_data['us_vix'][['Close']].rename(columns={'Close': 'US_VIX'})
            base_df = base_df.join(us_vix, how='left')
            base_df['US_VIX'] = base_df['US_VIX'].ffill()
            base_df['US_VIX_Change'] = base_df['US_VIX'].pct_change()
            print("  ✅ US VIX features added")
        

        # SECTION 6: CURRENCY AND COMMODITIES
        
        # INR/USD (FII flow indicator)
        if not self.market_data['inr_usd'].empty:
            inr_data = self.market_data['inr_usd'][['Close']].rename(columns={'Close': 'INR_USD'})
            base_df = base_df.join(inr_data, how='left')
            base_df['INR_USD'] = base_df['INR_USD'].ffill()
            base_df['INR_Depreciation'] = base_df['INR_USD'].pct_change()
            base_df['INR_Stress'] = (
                (base_df['INR_Depreciation'] > 0.01).rolling(5).sum()
            )
            print("  ✅ Currency features added")
        
        # Crude Oil (India imports 80%+)
        if not self.market_data['crude_oil'].empty:
            oil_data = self.market_data['crude_oil'][['Close']].rename(columns={'Close': 'Crude_Oil'})
            base_df = base_df.join(oil_data, how='left')
            base_df['Crude_Oil'] = base_df['Crude_Oil'].ffill()
            base_df['Oil_Change'] = base_df['Crude_Oil'].pct_change()
            base_df['Oil_Shock'] = (abs(base_df['Oil_Change']) > 0.05).astype(int)
            print("  ✅ Oil features added")
        
        # Gold (safe haven)
        if not self.market_data['gold'].empty:
            gold_data = self.market_data['gold'][['Close']].rename(columns={'Close': 'Gold_Price'})
            base_df = base_df.join(gold_data, how='left')
            base_df['Gold_Price'] = base_df['Gold_Price'].ffill()
            base_df['Gold_Return'] = base_df['Gold_Price'].pct_change()
            
            # Gold-Equity divergence (flight to safety)
            base_df['Flight_To_Safety'] = (
                base_df['Gold_Return'] - base_df['Nifty_Return']
            )
            print("  ✅ Gold features added")

        
        base_df = self._add_composite_indicators(base_df) # SECTION 7: COMPOSITE STRESS INDICATORS
        
        
        base_df = self._create_target(base_df)  # SECTION 8: TARGET VARIABLE 
        
        return base_df
    
       
    # ======================================
    # ADDS PRICE & VOLUME FEATURES
    # ======================================
    def _add_basic_features(self, df, prefix):
        
        df[f'{prefix}_Return'] = df['Close'].pct_change()
        df[f'{prefix}_Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multiple volatility windows
        df[f'{prefix}_Vol_5D'] = df[f'{prefix}_Return'].rolling(5).std()
        df[f'{prefix}_Vol_20D'] = df[f'{prefix}_Return'].rolling(20).std()
        df[f'{prefix}_Vol_60D'] = df[f'{prefix}_Return'].rolling(60).std()
        
        # Z-scores
        df[f'{prefix}_ZScore_20D'] = (
            (df[f'{prefix}_Return'] - df[f'{prefix}_Return'].rolling(20).mean()) 
            / df[f'{prefix}_Vol_20D']
        )
        
        # Volume features
        df[f'{prefix}_RelVol'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df[f'{prefix}_VolSpike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
        
        # Drawdown
        rolling_max = df['Close'].rolling(50).max()
        df[f'{prefix}_Drawdown'] = (df['Close'] / rolling_max) - 1
        
        return df
    
    # ======================================
    # ADDS TAIL RISK & DISTRIBUTION FEATURES
    # ======================================
    def _add_tail_risk_features(self, df):
        
        # Downside deviation
        df['Downside_Deviation'] = df['Nifty_Return'].rolling(20).apply(
            lambda x: np.sqrt(np.mean(np.minimum(x, 0)**2))
        )
        
        # Skewness and Kurtosis
        df['Return_Skewness'] = df['Nifty_Return'].rolling(20).apply(lambda x: skew(x))
        df['Return_Kurtosis'] = df['Nifty_Return'].rolling(20).apply(lambda x: kurtosis(x))
        
        # Lower partial moments (crash risk)
        df['LPM_1'] = df['Nifty_Return'].rolling(20).apply(
            lambda x: np.mean(np.minimum(x, 0))
        )
        df['LPM_2'] = df['Nifty_Return'].rolling(20).apply(
            lambda x: np.mean(np.minimum(x, 0)**2)
        )
        
        # Maximum drawdown in window
        df['Max_Drawdown_20D'] = df['Nifty_Drawdown'].rolling(20).min()
        
        return df
    
    # ==========================================
    # CREATES FEATURES FOR SECTORIAL INDICES
    # ==========================================
    def _create_sector_features(self, sector_df, sector_name):
        
        features = pd.DataFrame(index=sector_df.index)
        
        features[f'{sector_name}_Return'] = sector_df['Close'].pct_change()
        features[f'{sector_name}_Vol'] = features[f'{sector_name}_Return'].rolling(20).std()
        features[f'{sector_name}_RelVol'] = (
            sector_df['Volume'] / sector_df['Volume'].rolling(20).mean()
        )
        
        return features
    
    # ==========================================
    # ADDS COMPOSITE STRESS INDICATORS
    # ==========================================
    def _add_composite_indicators(self, df):
        
        # Multi-factor stress score
        stress_components = []
        
        # Component 1: Volatility stress
        if 'Nifty_Vol_20D' in df.columns:
            vol_stress = df['Nifty_Vol_20D'] / df['Nifty_Vol_60D']
            stress_components.append(vol_stress)
        
        # Component 2: VIX stress
        if 'VIX_Spike' in df.columns:
            stress_components.append(df['VIX_Spike'])
        
        # Component 3: Drawdown stress
        if 'Nifty_Drawdown' in df.columns:
            drawdown_stress = abs(df['Nifty_Drawdown']) * 2  # Scale to 0-1 range
            stress_components.append(drawdown_stress)
        
        # Component 4: Volume stress
        if 'Nifty_RelVol' in df.columns:
            vol_stress = np.clip(df['Nifty_RelVol'] - 1, 0, 2) / 2
            stress_components.append(vol_stress)
        
        # Combine into composite stress
        if stress_components:
            df['Composite_Stress'] = np.mean(stress_components, axis=0)
            df['Composite_Stress_MA'] = df['Composite_Stress'].rolling(5).mean()
        
        return df
    
    # ================================================
    # CREATES TARGET VARIABLE USING CRASH INDICATORS
    # ================================================
    def _create_target(self, df):
        
        # Original panic intensity
        df['Panic_Intensity'] = np.where(
            df['Nifty_ZScore_20D'] < 0,
            abs(df['Nifty_ZScore_20D']) * df['Nifty_RelVol'],
            0
        )
        
        # Multi-factor crash detection
        crash_signals = []
        
        # Signal 1: Extreme Z-score with volume
        signal_1 = (df['Nifty_ZScore_20D'] <= -2.0) & (df['Nifty_RelVol'] >= 1.3)
        crash_signals.append(signal_1.astype(int))
        
        # Signal 2: VIX spike
        if 'VIX_Spike' in df.columns:
            signal_2 = (df['VIX_Spike'] > 1.5)
            crash_signals.append(signal_2.astype(int))
        
        # Signal 3: Rapid drawdown
        if 'Nifty_Drawdown' in df.columns:
            signal_3 = (df['Nifty_Drawdown'] < -0.05)  # 5% drawdown
            crash_signals.append(signal_3.astype(int))
        
        # Signal 4: Composite stress
        if 'Composite_Stress' in df.columns:
            signal_4 = (df['Composite_Stress'] > df['Composite_Stress'].quantile(0.90))
            crash_signals.append(signal_4.astype(int))
        
        # Combine signals (at least 2 out of N must trigger)
        if len(crash_signals) >= 2:
            df['Crash_Signal_Count'] = sum(crash_signals)
            df['Target_Regime'] = (df['Crash_Signal_Count'] >= 2).astype(int)
        else:
            # Fallback to original method
            threshold95 = df['Panic_Intensity'].quantile(0.93)  # Slightly lower threshold
            df['Target_Regime'] = (df['Panic_Intensity'] > threshold95).astype(int)
        
        # Look ahead 1 day
        df['Target_Regime'] = df['Target_Regime'].shift(-1)
        
        return df
    
    # ==========================================
    # CREATES CONSOLIDATED & CLEANED DATASET
    # ==========================================
    def get_final_dataset(self):
        
        # Fetch all data
        self.fetch_all_market_data()
        
        # Engineer features
        master_df = self.engineer_comprehensive_features()
        
        # Select feature columns (exclude OHLCV and intermediate calculations)
        exclude_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
            'Panic_Intensity', 'Crash_Signal_Count', 'Target_Regime',
            'Crude_Oil', 'Gold_Price', 'India_VIX', 'US_VIX', 'INR_USD', 'SPY_Return'
        ]
        
        feature_cols = [col for col in master_df.columns if col not in exclude_cols]
        
        # Create final dataframe
        X = master_df[feature_cols]
        y = master_df['Target_Regime']
        
        # Combine
        final_df = X.copy()
        final_df['Target_Regime'] = y
        
        # Clean
        final_df = final_df.dropna()
        final_df.columns = final_df.columns.astype(str)
        
        # Stats
        crash_count = (final_df['Target_Regime'] == 1).sum()
        normal_count = (final_df['Target_Regime'] == 0).sum()
        
        print("\n" + "="*60)
        print("📊 FINAL DATASET STATISTICS")
        print("="*60)
        print(f"Total samples: {len(final_df)}")
        print(f"Crash days: {crash_count} ({crash_count/len(final_df)*100:.2f}%)")
        print(f"Normal days: {normal_count} ({normal_count/len(final_df)*100:.2f}%)")
        print(f"Features: {len(feature_cols)}")
        print(f"Date range: {final_df.index[0]} to {final_df.index[-1]}")
        print("="*60 + "\n")
        
        return final_df


# =============================================================
#                           EXECUTION
# =============================================================

if __name__ == "__main__":
    
    # Creates dataset
    engine = IndianMarketFeatureEngine(
        start_date="2008-01-01",
        end_date="2026-05-01"
    )
    
    master_df = engine.get_final_dataset()
    
    print("✅ Dataset created successfully!")
    print(f"\nSample features:\n{master_df.columns.tolist()[:10]}")
