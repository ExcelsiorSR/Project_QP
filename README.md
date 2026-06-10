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
**Input**: Historical ```.csv``` files stored in the /data folder, alongside the unstructured NLP data dynamically fetched via NewsAPI.

**Output**: ```crash_predictor.pkl``` (the model) and ```financial_news.db``` (the SQLite database). The Streamlit app relies on these generated files to function.

## ✨ Features / App Overview
This application provides an institutional-grade toolset for tracking compounding behavioral panic and macroeconomic tail-risk. Here is a breakdown of each module:

🏠 **Home**(```Home.py```): The main landing page outlining the Dual-Engine Architecture, the lagging indicator trap, and the overall thesis of the regime shift classifier.

🛡️ **Risk Monitor** (```1_Risk_Monitor.py```): The live dashboard displaying the current market regime. It calculates the live probability of an impending crash based on the latest daily inference.

🌍 **Systemic Context** (```2_Systemic_Context.py```): Visualizes the Alternative Data (NLP) Engine's outputs, tracking the daily Systemic Stress Score extracted from unstructured financial headlines using FinBERT.

📈 **Live Markets** (```3_Live_Markets.py```): Displays the structural market microstructure, including the global contagion correlation matrix and volatility skews across global indices.

📊 **Backtest Results** (```4_Backtest_Results.py```): Allows users to evaluate the historical performance of the model (2023-2026), specifically showcasing the reduction in Maximum Drawdown from -14.78% (Buy & Hold) to -13.20% (Strategy) and an 88.9% Crash Detection Recall.

🧠 **XAI Engine** (```5_XAI_Engine.py```): Proves the system is not a "Black Box" by deploying SHAP (SHapley Additive exPlanations) to dynamically map exactly which features (e.g., VIX spikes, negative news sentiment) are driving the crash probability on any given day.

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
