import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = "trained_model.pkl"


# ===============================
# 1️⃣ Load NIFTY Data
# ===============================
def load_data(ticker="^NSEI", period="3mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df


# ===============================
# 2️⃣ Feature Engineering
# ===============================
def feature_engineer(df):
    df = df.copy()

    # Ensure 'Close' is a clean numeric Series
    close = df["Close"].astype(float)

    # Technical Indicators
    df["SMA_5"] = SMAIndicator(close, window=5).sma_indicator()
    df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()

    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    rsi = RSIIndicator(close, window=14)
    df["RSI"] = rsi.rsi()

    bb = BollingerBands(close)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    df.dropna(inplace=True)
    return df


# ===============================
# 3️⃣ Train Model
# ===============================
def train_and_save():
    df = load_data()
    df = feature_engineer(df)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    X = df[["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]]
    y = df["Target"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


# ===============================
# 4️⃣ Load Model
# ===============================
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save()


# ===============================
# 5️⃣ Predict Next Day
# ===============================
def predict_next_day():
    df = load_data()
    df = feature_engineer(df)
    model = load_model()

    latest = df[["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]].iloc[-1:].values
    y_pred = model.predict(latest)[0]
    conf = model.predict_proba(latest)[0].max()

    trend = "📈 Uptrend Expected Tomorrow" if y_pred == 1 else "📉 Downtrend Expected Tomorrow"
    return trend, round(conf * 100, 2)
