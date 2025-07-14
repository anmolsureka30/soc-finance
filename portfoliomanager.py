# -------------------------------------------
# 🚀 Libraries
# -------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.data_processor import DataProcessor
from finrl.agents.stablebaselines3.models import DRLAgent
import datetime
import warnings

# -------------------------------------------
# 💡 Suppress Warnings
# -------------------------------------------
warnings.filterwarnings("ignore")

# -------------------------------------------
# 🎯 Streamlit Setup
# -------------------------------------------
st.set_page_config(page_title="AI Financial Portfolio Manager", layout="wide")
st.title("💼📈 AI-powered Financial Portfolio Manager (DRL + Sentiment)")

st.sidebar.header("Configuration")
start_date = st.sidebar.date_input("Start date", datetime.date(2022, 1, 3))  # Weekday start to avoid YF error
end_date = st.sidebar.date_input("End date", datetime.date(2023, 12, 31))
tickers = st.sidebar.multiselect(
    "Select tickers", ["AAPL", "MSFT", "GOOG", "META", "TSLA"], default=["AAPL", "MSFT", "GOOG"]
)

# -------------------------------------------
# 📥 Data Download & Processing
# -------------------------------------------
dp = DataProcessor(
    data_source="yahoofinance",
    start_date=start_date,
    end_date=end_date,
    time_interval="1D",
    technical_indicator_list=config.INDICATORS,
    if_vix=True
)

try:
    dp.download_data(ticker_list=tickers, start_date=start_date, end_date=end_date, time_interval="1D")
    dp.clean_data()
    dp.add_technical_indicator()
    dp.add_vix()
    dp.add_turbulence()
    df = dp.dataframe

    if df.empty:
        st.error("❌ No data downloaded. Please adjust your date range or tickers.")
        st.stop()
    else:
        st.success("✅ Data successfully downloaded!")
except Exception as e:
    st.error(f"❌ Data download failed: {e}")
    st.stop()

st.subheader("📄 Raw Historical Data")
st.dataframe(df.head())

# -------------------------------------------
# 💬 FinBERT Sentiment Analysis
# -------------------------------------------
st.subheader("💬 Sentiment Feature (FinBERT)")

sample_news = [
    "Apple launches new AI-powered chip, stock surges",
    "Google faces antitrust lawsuit, investors worry",
    "Tesla delivers record number of vehicles this quarter",
]
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
sentiment_results = sentiment_pipeline(sample_news)

sentiment_df = pd.DataFrame(sentiment_results)
sentiment_df["headline"] = sample_news
st.table(sentiment_df)

# -------------------------------------------
# 🔧 Feature Engineering
# -------------------------------------------
st.subheader("⚙️ Feature Engineering & Indicators")

scaler = StandardScaler()
df["volatility"] = df.groupby("tic")["close"].rolling(window=20).std().reset_index(0, drop=True)
df["momentum"] = df.groupby("tic")["close"].pct_change(periods=10).reset_index(0, drop=True)
df["rsi_scaled"] = scaler.fit_transform(df[["rsi"]].fillna(0))

fig_feat = px.line(df, x="date", y=["rsi", "volatility", "momentum"], color="tic", title="Indicators over time")
st.plotly_chart(fig_feat, use_container_width=True)

# -------------------------------------------
# 🤖 DRL Agent Training
# -------------------------------------------
st.subheader("🤖 DRL Agent Training & Simulation")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1e6,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4,
    "state_space": df.shape[1],
    "action_space": len(tickers),
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 0,
}

e_train_gym = StockTradingEnv(df=df, turbulence_threshold=250, **env_kwargs)
agent = DRLAgent(env=e_train_gym)

agent_choice = st.selectbox("Choose DRL Agent", ["PPO", "A2C", "SAC"])
if agent_choice == "PPO":
    model = agent.get_model("ppo")
elif agent_choice == "A2C":
    model = agent.get_model("a2c")
else:
    model = agent.get_model("sac")

st.info("Training the selected DRL agent... ⏳")
trained_model = agent.train_model(model=model, tb_log_name=agent_choice, total_timesteps=5000)

df_account_value, df_actions = agent.DRL_prediction(model=trained_model)

st.subheader("📈 Portfolio Value Curve")
fig_acc = px.line(df_account_value, x="date", y="account_value", title="Account Value Over Time")
st.plotly_chart(fig_acc, use_container_width=True)

# -------------------------------------------
# ⚔️ Strategy Comparison
# -------------------------------------------
st.subheader("⚔️ Strategy Comparison")

benchmark = df_account_value.copy()
benchmark["benchmark"] = benchmark["account_value"] * np.random.uniform(0.95, 1.05, len(benchmark))

fig_comp = go.Figure()
fig_comp.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["account_value"], name=f"{agent_choice} Strategy"))
fig_comp.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["benchmark"], name="Simulated Benchmark"))
fig_comp.update_layout(title="Strategy vs Benchmark", xaxis_title="Date", yaxis_title="Value")
st.plotly_chart(fig_comp, use_container_width=True)

# -------------------------------------------
# 🔮 Scenario Analysis
# -------------------------------------------
st.subheader("🔮 Scenario Analysis")

sentiment_shock = st.slider("Sentiment Shock Factor (Simulated)", 0.5, 1.5, 1.0)
risk_aversion = st.slider("Risk Aversion Penalty", 0.1, 1.0, 0.5)

st.write(f"Applying sentiment shock: {sentiment_shock} and risk aversion: {risk_aversion}")

df_account_value["adjusted_value"] = df_account_value["account_value"] * sentiment_shock * (1 - risk_aversion * 0.1)

fig_scenario = px.line(df_account_value, x="date", y="adjusted_value", title="Scenario-Adjusted Account Value")
st.plotly_chart(fig_scenario, use_container_width=True)

# -------------------------------------------
# 🛡️ Risk Metrics
# -------------------------------------------
st.subheader("📊 Risk & Performance Metrics")

returns = df_account_value["account_value"].pct_change().dropna()
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
sortino_ratio = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252)
max_drawdown = (df_account_value["account_value"] / df_account_value["account_value"].cummax() - 1).min()

st.markdown(f"""
- **Sharpe Ratio**: `{sharpe_ratio:.2f}`
- **Sortino Ratio**: `{sortino_ratio:.2f}`
- **Max Drawdown**: `{max_drawdown:.2%}`
""")

st.success("🎉 Analysis Complete! Explore different agents and tune scenarios!")