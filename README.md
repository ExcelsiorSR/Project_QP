# Project_QP
An AI-driven quantitative finance architecture that fuses macroeconomic indicators with FinBERT NLP sentiment to predict systemic market crashes and behavioral regime shifts.
# Predicting Crash: Evaluating Market Panic 📉
**Core Architecture: Behavioral Regime Shift Classifier**

An AI-driven quantitative finance architecture that fuses macroeconomic indicators with FinBERT NLP sentiment to predict systemic market crashes and behavioral regime shifts.


---

## 🏛 System Architecture
Traditional risk models are structurally reactive, failing to detect compounding behavioral panic until the price has already collapsed. This project introduces a **Dual-Engine Architecture** to detect the invisible onset of macroeconomic panic before the sell-off.

1. **Alternative Data NLP Engine:** Scrapes unstructured daily financial headlines via NewsAPI and utilizes a localized HuggingFace **FinBERT** pipeline to calculate a daily "Systemic Stress Score."
2. **Market Microstructure Engine:** Extracts structural tail-risk indicators (Volatility Skew, Lower Partial Moments, Drawdowns, Dollar Index) using 18 years of historical OHLCV data.
3. **Cost-Sensitive Meta-Learner:** A highly optimized ensemble (XGBoost, Random Forest, GBM) mathematically penalized (5x) for False Negatives, ensuring capital preservation over yield chasing.

---

## 📊 Evaluation Metrics & Financial Impact
The system was validated on unseen test data (2023-2026) focusing on severe drawdown events. Standard ML accuracy was discarded in favor of Recall and Capital Preservation.

* **Crash Detection Recall:** `88.9%` (Successfully captured 16 out of 18 crash regimes)
* **ROC-AUC Score:** `0.973`
* **Weighted Cost Function:** `25.00` (Optimized for False Negative minimization)
* **Financial Impact (Backtest):** Reduced the maximum portfolio drawdown from `-14.78%` (Buy & Hold) to `-13.20%` (Strategy).

---

## 📂 Repository Structure

```text
├── data/                           # SQLite database and static historical CSVs
├── modules/                        # Core Quantitative Classes
│   ├── data_pipeline.py            # IndianMarketFeatureEngine
│   ├── db_manager.py               # Database Architecture
│   ├── model_engine.py             # CostSensitiveEnsemble
│   └── nlp_engine.py               # FinBERT Systemic Stress Engine
├── pages/                          # Streamlit Multi-Page UI files
│   ├── 1_Risk_Monitor.py           
│   ├── 2_Systemic_Context.py       
│   ├── 3_Live_Markets.py           
│   ├── 4_Backtest_Results.py       
│   └── 5_XAI_Engine.py             
├── scripts/                        # Execution & Automation Scripts
│   ├── backtest_simulator.py       # Historical performance evaluation
│   ├── daily_inference.py          # Live dynamic delta-pipeline
│   ├── main.py                     # Core Historical Training Pipeline
│   └── news_scraper.py             # Automated NewsAPI ingestion
├── tools/                          # Research, batch processing, and CI/CD validation
│   ├── clean_nlp_data.py           
│   ├── colab_nlp_finbert.ipynb     
│   └── validate_system.py          
├── Home.py                         # Streamlit Dashboard Entry Point
├── crash_predictor.pkl             # Serialized Cost-Sensitive Ensemble
├── requirements.txt                # System dependencies
└── *.png / *.csv                   # Output visuals and root data signals
