

💼📈 AI-Powered Financial Portfolio Manager

Welcome to the AI-Powered Financial Portfolio Manager, an advanced, modular platform that integrates cutting-edge Deep Reinforcement Learning (DRL), technical indicators, and sentiment analysis to build, simulate, and optimize financial trading strategies — all within an interactive and user-friendly Streamlit app.

⸻

🚀 Overview

This project aims to empower traders and financial engineers to design intelligent, self-adaptive trading agents capable of:

✅ Learning optimal trading strategies using DRL (PPO, A2C, SAC)
✅ Incorporating real-world technical indicators and advanced feature engineering
✅ Integrating FinBERT sentiment analysis for news-based signals
✅ Visualizing backtests, agent performance, and scenario analyses interactively

⸻

🧰 Features

✨ Streamlit Dashboard
	•	Intuitive, modern interface to configure date ranges, tickers, agent type, and parameters
	•	Real-time plots using Plotly (account value, benchmark comparison, scenario analysis)
	•	Interactive controls for risk and sentiment shock adjustments

🤖 DRL Agents
	•	Supports PPO, A2C, and SAC from Stable-Baselines3
	•	Modular design allows easy integration of new algorithms
	•	Automated training and prediction loops

📊 Advanced Feature Engineering
	•	Rolling volatility, momentum, RSI scaling
	•	Custom technical indicators via indicators.py
	•	Turbulence and VIX signals for robust strategy design

💬 Sentiment Analysis
	•	Integrates FinBERT (ProsusAI/finbert) for analyzing financial news sentiment
	•	Adjustable scenario shock based on sentiment

⚔️ Strategy Comparison
	•	Visual performance against simulated benchmark strategies
	•	Scenario-adjusted performance visualization

🛡️ Risk Metrics
	•	Sharpe Ratio
	•	Sortino Ratio
	•	Maximum Drawdown
	•	Scenario stress testing

⸻

📁 Project Structure

├── actorcritic.py                # Actor-Critic RL example
├── BTCUSD_raw.csv                # Sample BTC historical data
├── data/                         # Folder for raw and processed data
├── data_pipeline.ipynb          # Data processing & preparation notebook
├── featurengg.py                # Feature engineering utility
├── frozenlake-examples/        # Gym environment experiments
├── gymwithneuralnetwork.py     # Custom gym experiments with NN
├── indicators.py               # Technical indicators definitions
├── openaigymbasic.py          # Basic OpenAI Gym examples
├── portfoliomanager.py        # Main Streamlit app (AI Portfolio Manager)
├── README.md                  # Project description (this file)
├── requirements.txt          # Python dependencies
└── venv310/                  # Virtual environment (ignored in git)


⸻

🗂️ Data Sources
	•	Yahoo Finance (via yfinance API)
	•	Technical indicators generated in indicators.py
	•	Optional: BTC dataset for crypto experiments

⸻

⚙️ Setup & Installation

# Clone this repository
git clone https://github.com/anmolsureka30/soc-finance.git
cd soc-finance

# Create a virtual environment (if not already)
python3 -m venv venv310
source venv310/bin/activate

# Install dependencies
pip install -r requirements.txt


⸻

▶️ Running the Streamlit App

streamlit run portfoliomanager.py

Once running, open http://localhost:8501 in your browser to interact with the app.

⸻

📊 Example Usage

1️⃣ Select start/end dates and tickers (e.g., AAPL, MSFT, GOOG)
2️⃣ Choose a DRL agent (PPO, A2C, or SAC)
3️⃣ Train and visualize agent performance
4️⃣ Analyze risk metrics and scenario-adjusted curves
5️⃣ Tune sentiment shock and risk aversion sliders to simulate extreme market conditions

⸻

🤝 Contributing

We welcome contributions! You can:
	•	Add new indicators or alternative data sources
	•	Integrate new DRL algorithms or custom strategies
	•	Improve dashboard UX/UI
	•	Write unit tests and improve documentation

⸻

💬 Contact

Anmol Sureka
⸻

⭐️ Acknowledgements
	•	FinRL Library
	•	Stable Baselines3
	•	Yahoo Finance API
	•	HuggingFace Transformers

⸻

⚖️ License

This project is licensed under the MIT License — see the LICENSE file for details.

⸻

💥 Future Extensions
	•	✅ Real-time data streaming integration
	•	✅ Automated model hyperparameter optimization
	•	✅ Live paper trading or Alpaca integration
	•	✅ Multi-agent collaboration and ensemble strategies
	•	✅ Portfolio optimization with options, futures, and crypto

