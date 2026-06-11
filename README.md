# Project_QP
An AI-driven quantitative finance architecture that fuses macroeconomic indicators with FinBERT NLP sentiment to predict systemic market crashes and behavioral regime shifts.
# Predicting Crash: Evaluating Market Panic 📉
**Core Architecture: Behavioral Regime Shift Classifier**

An AI-driven quantitative finance architecture that fuses macroeconomic indicators with FinBERT NLP sentiment to predict systemic market crashes and behavioral regime shifts.


---

## 🏛 System Architecture
Three-Layer Institutional Architecture

Traditional risk models are structurally reactive, failing to detect compounding behavioral panic until prices have already collapsed. Project QP introduces a Three-Layer Architecture designed to identify the invisible onset of systemic stress before the broader market reprices risk.

Layer 1: Alternative Data Intelligence Engine
• Scrapes real-time financial headlines through NewsAPI
• Processes unstructured text using a localized FinBERT pipeline
• Generates a daily Systemic Stress Score
• Produces macro_stress_signals.csv and latest_stress_headlines.csv

Layer 2: Market Microstructure Engine
• Extracts structural tail-risk indicators from global market data
• Engineers volatility, drawdown, and cross-asset stress features
• Tracks Flight-to-Safety behaviour through Gold, Oil, Currency and VIX dynamics
• Calculates Composite Stress and sectoral rotation metrics

Layer 3: Cost-Sensitive Ensemble
• Ensemble of XGBoost, Random Forest, and Gradient Boosting models
• Optimized to heavily penalize False Negatives (5x cost weighting)
• Produces dynamic crash probabilities and market regime classifications

Feature Count: 41 engineered quantitative and alternative-data features.

---

## 📊 Evaluation Metrics & Financial Impact
The system was validated on unseen test data using a cost-sensitive framework focused on crash detection rather than conventional classification accuracy.

Crash Detection Recall: 88.9%
Successfully captured 16 out of 18 crash regimes.

ROC-AUC Score: 0.968

Weighted Cost Function: 25.00
Optimized specifically to minimize False Negatives.

Financial Impact (Backtest)

Buy & Hold Maximum Drawdown:
-14.78%

Project QP Maximum Drawdown:
-10.85%

The model demonstrates a meaningful reduction in portfolio drawdown while maintaining exposure during normal market conditions.

## ⚙️ Automated MLOps Pipeline

The entire production architecture is automated using GitHub Actions.

Live Market & Sentiment Sync
Frequency: Every 30 minutes during NSE trading hours

Tasks:
• News ingestion via NewsAPI
• FinBERT sentiment extraction
• Systemic Stress Score generation
• Daily feature engineering
• Crash probability inference
• Prediction database updates

Weekly Ensemble Retraining
Frequency: Every Monday before market open

Tasks:
• Historical dataset refresh
• Model retraining
• Threshold recalibration
• Validation generation
• Automated backtesting
• Artifact deployment

---

## 📂 Repository Structure

```text
├── data/                           # SQLite database and static historical CSVs
│   ├── final_predictions.csv       
│   ├── financial_news.db
│   ├── historical_nlp_stress_rg.csv 
│   ├── historical_nlp_stress.csv 
│   ├── latest_stress_headlines.csv
│   ├── live_risk_history.csv
│   ├── macro_stress_signals.csv
│   └── model_validation_results.csv                       
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
│   ├── news_scraper.py             # Automated NewsAPI ingestion
│   └── risk_snapshot.py            # Daily live risk history generation
├── tools/                          # Research, batch processing, and CI/CD validation
│   ├── clean_nlp_data.py           
│   ├── colab_nlp_finbert.ipynb     
│   └── validate_system.py          
├── Home.py                         # Streamlit Dashboard Entry Point
├── crash_predictor.pkl             # Serialized Cost-Sensitive Ensemble
├── requirements.txt                # System dependencies
└── *.png / *.csv                   # Output visuals and root data signals
```
## 🚀 How to Run Locally

**Clone the Repository**
```bash
git clone [https://github.com/ExcelsiorSR/Project_QP.git](https://github.com/ExcelsiorSR/Project_QP.git)
cd Project_QP
```
**Setup Environment:**

```Bash
# Create and activate a conda environment
conda create -n ProjectQP python=3.10
conda activate ProjectQP

# Install all required packages
pip install -r requirements.txt
```
**Unstructured NLP Processing (Colab)**
Due to the computational intensity of running HuggingFace Transformers, the historical financial news dataset was processed via Google Colab. To bypass GitHub's 100MB file limit, the raw dataset is fragmented.

1. Clone the repository and upload `tools/colab_nlp_finbert.ipynb` to Google Colab.
2. Upload the entire `raw_data/` folder (containing the fragmented `.csv` parts) to your Colab session storage.
3. Run the **Dataset Reassembly** cell at the top of the notebook to stitch the fragments back into the master dataset.
4. Execute the rest of the pipeline using a T4 GPU runtime to generate the daily Systemic Stress Scores.
5. Download the resulting `historical_nlp_stress.csv` and place it in your local `/data` directory before running `main.py`.
   
**Secure API Keys:**
Before running, you must provide a NewsAPI key for the NLP engine.
Create a file named exactly ```.env``` in the root folder and add:
```NEWS_API_KEY=your_actual_api_key_here```

**Run the Pre-Flight Diagnostic:**
Verify system integrity, directory structures, and dependencies before execution.
```python tools/validate_system.py```

**Run the AI Engine (Historical Training):**
This script trains the Cost-Sensitive Ensemble on 18 years of historical data (2008-2026) and generates the .pkl model file.

```Bash
python scripts/main.py
```

**Run the Live Updater:**
Fetch today's live market data, scrape the latest news, calculate the FinBERT stress score, and update the database without retraining the whole model.

```Bash
python scripts/daily_inference.py
```

**Launch the Web App:**
This runs the Streamlit production UI.

```Bash
streamlit run Home.py
```
## ⚙️ Key Components

```scripts/main.py``` **(The "Historical Engine")**: This is the core training script. It processes the raw historical data, applies the SMOTETomek imbalanced learning techniques, and trains the Cost-Sensitive Ensemble (XGBoost, Random Forest, GBM), saving it as ```crash_predictor.pkl```.

```scripts/daily_inference.py``` **(The "Live Engine")**: The dynamic delta-pipeline. It runs the localized HuggingFace FinBERT pipeline and fetches ```yfinance``` data to update the live system without touching the historical training data.

```Home.py``` & ```pages/``` **(The "App")**: This is the Streamlit web application that visually renders the historical backtests, SHAP explainability, and live market regimes.

## 🗄️ Data Files
Input Files

• financial_news.db
  Historical financial news archive.

• historical_nlp_stress.csv
  FinBERT stress scores generated from broad news datasets.

• historical_nlp_stress_rg.csv
  FinBERT stress scores generated from financial news datasets.

• market_data.csv and supporting historical datasets
  Raw market and macroeconomic inputs.

Output Files

• crash_predictor.pkl
  Serialized production ensemble model.

• final_predictions.csv
  Historical model predictions and crash probabilities.

• live_risk_history.csv
  Daily risk-monitoring dataset used by the dashboard.

• macro_stress_signals.csv
  Daily systemic sentiment signals generated from live news.

• latest_stress_headlines.csv
  Most recent headlines contributing to sentiment scores.

• model_validation_results.csv
  Weekly validation outputs generated during retraining.

## ✨ Features / App Overview
🏠 Home (Home.py)
Introduces the Three-Layer Architecture and explains the rationale behind behavioral regime shift detection.

🚨 Risk Monitor (1_Risk_Monitor.py)
Provides live crash probabilities, historical risk tracking, and institutional-grade risk monitoring metrics.

🌐 Systemic Context (2_Systemic_Context.py)
Displays macroeconomic drivers including global contagion, liquidity conditions, sector rotation, and sentiment dynamics.

📈 Live Markets (3_Live_Markets.py)
Tracks live market conditions, volatility structures, and key macroeconomic indicators.

📊 Backtest Results (4_Backtest_Results.py)
Evaluates historical strategy performance, drawdown reduction, and crash detection effectiveness.

🧠 XAI Engine (5_XAI_Engine.py)
Uses SHAP explainability to identify the primary factors driving every model prediction.

## 🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, want to add tick-level data granularity, or find any bugs, please feel free to:

Fork the repository.

Create a new branch (```git checkout -b feature/YourAmazingFeature```).

Commit your changes (```git commit -m 'Add some AmazingFeature'```).

Push to the branch (```git push origin feature/YourAmazingFeature```).

Open a Pull Request.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

*Author: Utthan Singh Roy* | [LinkedIn](https://www.linkedin.com/in/srexcelsior/) | [Portfolio](https://excelsiorsr.github.io/)
