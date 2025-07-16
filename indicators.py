#Given daily stock price data, generate major technical indicators to analyze whether a stock is overbought, oversold, trending, or consolidating. 
# Then visualize these indicators to understand their signals.
# we will be using FinRL for data loading 

# Import
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np

## Downloading Price Data
# Example: Apple stock (AAPL)
ticker = "AAPL"
df = yf.download(ticker, start="2022-01-01", end="2023-12-31", auto_adjust=True)

# If columns are multi-index, flatten them
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

# If columns are like 'Close_AAPL', rename to 'Close', etc.
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if f"{col}_{ticker}" in df.columns:
        df[col] = df[f"{col}_{ticker}"]

# Display first few rows
print(df.head())

## Add log returns 
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

### Technical Indicators
# RSI (momentum)
df["rsi"] = ta.rsi(df["Close"], length=14)

# MACD (momentum/trend)
macd = ta.macd(df["Close"])
if macd is not None and "MACD_12_26_9" in macd and "MACDs_12_26_9" in macd:
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
else:
    print("MACD calculation failed or columns missing.")
    df["macd"] = np.nan
    df["macd_signal"] = np.nan

# SMA & EMA (trend)
df["sma_20"] = ta.sma(df["Close"], length=20)
df["ema_50"] = ta.ema(df["Close"], length=50)

# Bollinger Bands (volatility)
bb = ta.bbands(df["Close"], length=20)
if bb is not None and "BBU_20_2.0" in bb and "BBL_20_2.0" in bb:
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]
else:
    df["bb_upper"] = np.nan
    df["bb_lower"] = np.nan

# ATR (volatility)
df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

# OBV (volume confirmation)
df["obv"] = ta.obv(df["Close"], df["Volume"])

df.dropna(inplace=True)

# --- Plotly chart for price, SMA, EMA, BB ---
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index,
                open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], line=dict(color='blue', width=1), name='SMA 20'))
fig.add_trace(go.Scatter(x=df.index, y=df["ema_50"], line=dict(color='orange', width=1), name='EMA 50'))
fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], line=dict(color='green', width=1), name='BB Upper'))
fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], line=dict(color='red', width=1), name='BB Lower'))
fig.update_layout(title=f"{ticker} Price with SMA, EMA, Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
fig.show()

# --- Matplotlib subplots for all indicators ---
import matplotlib.dates as mdates
fig, axs = plt.subplots(5, 1, figsize=(16, 18), sharex=True)

# 1. Price with SMA, EMA, BB
axs[0].plot(df.index, df["Close"], label="Close", color="black", linewidth=1)
axs[0].plot(df.index, df["sma_20"], label="SMA 20", color="blue", linewidth=1)
axs[0].plot(df.index, df["ema_50"], label="EMA 50", color="orange", linewidth=1)
axs[0].fill_between(df.index, df["bb_upper"], df["bb_lower"], color='gray', alpha=0.2, label='Bollinger Bands')
axs[0].set_title(f"{ticker} Price, SMA, EMA, Bollinger Bands")
axs[0].set_ylabel("Price")
axs[0].legend()
axs[0].grid(True)

# 2. RSI
axs[1].plot(df.index, df["rsi"], label='RSI', color='purple')
axs[1].axhline(70, color='red', linestyle='--', label='Overbought (70)')
axs[1].axhline(30, color='green', linestyle='--', label='Oversold (30)')
axs[1].set_title("RSI (Overbought >70, Oversold <30)")
axs[1].set_ylabel("RSI")
axs[1].legend()
axs[1].grid(True)

# 3. MACD
axs[2].plot(df.index, df["macd"], label='MACD', color='blue')
axs[2].plot(df.index, df["macd_signal"], label='Signal Line', color='orange')
axs[2].axhline(0, color='grey', linestyle='--')
axs[2].set_title("MACD & Signal Line")
axs[2].set_ylabel("MACD")
axs[2].legend()
axs[2].grid(True)

# 4. ATR
axs[3].plot(df.index, df["atr"], label='ATR', color='brown')
axs[3].set_title("Average True Range (ATR)")
axs[3].set_ylabel("ATR")
axs[3].legend()
axs[3].grid(True)

# 5. OBV
axs[4].plot(df.index, df["obv"], label='On-Balance Volume (OBV)', color='teal')
axs[4].set_title("On-Balance Volume (OBV)")
axs[4].set_ylabel("OBV")
axs[4].legend()
axs[4].grid(True)

# Format x-axis as dates
axs[4].xaxis.set_major_locator(mdates.MonthLocator())
axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()