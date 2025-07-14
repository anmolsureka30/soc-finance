import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np

# --------------------------
# Downloading Data
# --------------------------
ticker = "AAPL"

df = yf.download(ticker, start="2022-01-01", end="2023-12-31", auto_adjust=True)

# If MultiIndex columns (common with yfinance), flatten them
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

# Check columns
print("Columns after flattening:", df.columns)

# Rename columns if needed
col_rename = {}
for col in df.columns:
    if "Close" in col:
        col_rename[col] = "Close"
    if "Open" in col:
        col_rename[col] = "Open"
    if "High" in col:
        col_rename[col] = "High"
    if "Low" in col:
        col_rename[col] = "Low"
    if "Volume" in col:
        col_rename[col] = "Volume"

df.rename(columns=col_rename, inplace=True)

# Reset index to get "Date" as a column
df.reset_index(inplace=True)

# --------------------------
# Add log returns
# --------------------------
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# --------------------------
# Technical Indicators
# --------------------------
df["rsi"] = ta.rsi(df["Close"], length=14)

macd = ta.macd(df["Close"])
if macd is not None:
    df = pd.concat([df, macd], axis=1)
else:
    print("MACD could not be calculated.")

df["sma_20"] = ta.sma(df["Close"], length=20)
df["ema_50"] = ta.ema(df["Close"], length=50)

bb = ta.bbands(df["Close"], length=20)
if bb is not None:
    df = pd.concat([df, bb], axis=1)
else:
    print("Bollinger Bands could not be calculated.")

df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
df["obv"] = ta.obv(df["Close"], df["Volume"])

df.dropna(inplace=True)

# --------------------------
# Plotly Candlestick Chart
# --------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Candlestick"
))

fig.add_trace(go.Scatter(x=df["Date"], y=df["sma_20"], line=dict(color='blue', width=1), name="SMA 20"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["ema_50"], line=dict(color='orange', width=1), name="EMA 50"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["BBU_20_2.0"], line=dict(color='green', width=1), name="BB Upper"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["BBL_20_2.0"], line=dict(color='red', width=1), name="BB Lower"))

fig.update_layout(title=f"{ticker} Price with SMA, EMA & Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
fig.show()

# --------------------------
# RSI
# --------------------------
plt.figure(figsize=(14, 4))
plt.plot(df["Date"], df["rsi"], label="RSI", color='purple')
plt.axhline(70, color='red', linestyle='--', label="Overbought")
plt.axhline(30, color='green', linestyle='--', label="Oversold")
plt.title("RSI (Relative Strength Index)")
plt.legend()
plt.show()

# --------------------------
# MACD
# --------------------------
plt.figure(figsize=(14, 4))
plt.plot(df["Date"], df["MACD_12_26_9"], label="MACD", color='blue')
plt.plot(df["Date"], df["MACDs_12_26_9"], label="Signal Line", color='orange')
plt.axhline(0, color='grey', linestyle='--')
plt.title("MACD & Signal Line")
plt.legend()
plt.show()

# --------------------------
# ATR (Volatility)
# --------------------------
plt.figure(figsize=(14, 4))
plt.plot(df["Date"], df["atr"], label="ATR", color='brown')
plt.title("Average True Range (Volatility)")
plt.legend()
plt.show()

# --------------------------
# OBV (On-Balance Volume)
# --------------------------
plt.figure(figsize=(14, 4))
plt.plot(df["Date"], df["obv"], label="OBV", color='darkcyan')
plt.title("On-Balance Volume")
plt.legend()
plt.show()

# --------------------------
# Log Returns
# --------------------------
plt.figure(figsize=(14, 4))
plt.plot(df["Date"], df["log_return"], label="Log Return", color='grey')
plt.axhline(0, color='black', linestyle='--')
plt.title("Log Returns Over Time")
plt.legend()
plt.show()