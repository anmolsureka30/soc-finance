# -------------------------------------------
# 🚀 Libraries
# -------------------------------------------
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

# -------------------------------------------
# 🎯 Download Data
# -------------------------------------------
ticker = "AAPL"

# Download data
df = yf.download(ticker, start="2022-01-01", end="2023-12-31")
df.reset_index(inplace=True)

# -------------------------------------------
# 🔧 Handle MultiIndex Columns (Important!)
# -------------------------------------------
# Flatten columns if MultiIndex appears (usually when using group_by='ticker' or default yfinance behavior)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

# Rename Date_ if present
if "Date_" in df.columns:
    df.rename(columns={"Date_": "Date"}, inplace=True)

# Rename AAPL columns to standard names
if "Close_AAPL" in df.columns:
    df["Close"] = df["Close_AAPL"]
    df["High"] = df["High_AAPL"]
    df["Low"] = df["Low_AAPL"]
    df["Open"] = df["Open_AAPL"]
    df["Volume"] = df["Volume_AAPL"]

print(df.columns)
print(df.head())
# -------------------------------------------
# 📈 Log Returns & Winsorization
# -------------------------------------------
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# Clip extreme outliers (Winsorization)
lower = df["log_return"].quantile(0.01)
upper = df["log_return"].quantile(0.99)
df["log_return_winsor"] = np.clip(df["log_return"], lower, upper)

# -------------------------------------------
# 📊 Rolling Average Detrending
# -------------------------------------------
window = 20
df["ma_20"] = df["Close"].rolling(window=window).mean()
df["detrended"] = df["Close"] - df["ma_20"]

# -------------------------------------------
# 🌍 Simulated Macro & Sentiment Features
# -------------------------------------------
np.random.seed(42)
df["cpi"] = np.random.normal(2.0, 0.2, len(df))      # Simulated CPI values
df["cpi_lag1"] = df["cpi"].shift(1)

df["sentiment_score"] = np.random.uniform(0, 1, len(df))   # Simulated sentiment score
df["sentiment_lag1"] = df["sentiment_score"].shift(1)

# -------------------------------------------
# ⚡ Technical Indicators using pandas-ta
# -------------------------------------------
df["macd"] = ta.macd(df["Close"])["MACD_12_26_9"]
df["rsi"] = ta.rsi(df["Close"], length=14)
df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)
df["adx"] = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"]

# Turbulence proxy: rolling volatility
df["turbulence"] = df["log_return"].rolling(window=20).std()

# Drop rows with NaNs from indicators and rolling features
df.dropna(inplace=True)

# -------------------------------------------
# 🔥 Feature Engineering: Scale RSI
# -------------------------------------------
scaler = StandardScaler()
df["rsi_scaled"] = scaler.fit_transform(df[["rsi"]])

# -------------------------------------------
# 🎯 Feature Matrix
# -------------------------------------------
feature_cols = [
    "log_return_winsor", "detrended", "macd", "rsi", "cci", 
    "adx", "turbulence", "cpi_lag1", "sentiment_lag1"
]

feature_matrix = df[feature_cols].dropna()
print("✅ Feature matrix shape:", feature_matrix.shape)

# -------------------------------------------
# 🟢 Visualization: Price & Moving Average
# -------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(df["Date"], df["Close"], label="Close Price", alpha=0.8, color="blue")
plt.plot(df["Date"], df["ma_20"], label="20-day MA", alpha=0.7, color="orange")
plt.title("AAPL Close Price with 20-day Moving Average", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------------------
# 🔵 Log Returns
# -------------------------------------------
plt.figure(figsize=(14, 4))
plt.plot(df["Date"], df["log_return"], label="Log Return", alpha=0.7, color="green")
plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
plt.title("AAPL Log Returns Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------------------
# ⚡ Technical Indicators (MACD & RSI)
# -------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["macd"], name="MACD", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["rsi"], name="RSI", line=dict(color="orange")))
fig.update_layout(title="AAPL MACD & RSI Over Time",
                  xaxis_title="Date",
                  yaxis_title="Value",
                  template="plotly_white")
fig.show()

# -------------------------------------------
# 🔥 Feature Correlation Heatmap
# -------------------------------------------
corr_matrix = feature_matrix.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------
# 🟣 Feature Distributions
# -------------------------------------------
feature_matrix.hist(figsize=(14, 10), bins=50, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# -------------------------------------------
# 💜 Rolling Volatility
# -------------------------------------------
df["rolling_vol"] = df["log_return"].rolling(window=20).std()

plt.figure(figsize=(14, 5))
plt.plot(df["Date"], df["rolling_vol"], label="20-day Rolling Volatility", color="purple")
plt.title("AAPL 20-day Rolling Volatility", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------------------
# 💠 Pairplot of Selected Features
# -------------------------------------------
sample_df = feature_matrix[["log_return_winsor", "macd", "rsi", "cci"]].dropna()

sns.pairplot(sample_df)
plt.suptitle("Pairwise Relationships of Selected Features", fontsize=16, y=1.02)
plt.show()

# -------------------------------------------
# ✅ Done!
# -------------------------------------------
print("🎉 All visualizations and feature engineering complete!")