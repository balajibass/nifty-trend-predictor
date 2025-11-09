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
# Load NIFTY Data
# ===============================
def load_data(ticker="^NSEI", period="3mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df


# ===============================
# Feature Engineering
# ===============================
def feature_engineer(df):
    df = df.copy()

    # Ensure Close column is numeric 1D
    if "Close" not in df.columns:
        raise ValueError("❌ DataFrame missing 'Close' column")
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    close_series = df["Close"].astype(float)

    df["SMA_5"] = SMAIndicator(close_series, window=5).sma_indicator()
    df["SMA_20"] = SMAIndicator(close_series, window=20).sma_indicator()

    macd = MACD(close_series)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    rsi = RSIIndicator(close_series, window=14)
    df["RSI"] = rsi.rsi()

    bb = BollingerBands(close_series)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    df.dropna(inplace=True)
    return df


# ===============================
# Train & Save Model
# ===============================
def train_and_save():
    df = load_data()
    df = feature_engineer(df)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    X = df[["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]]
    y = df["Target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


# ===============================
# Load or Train Model
# ===============================
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save()


# ===============================
# Predict Next Day
# ===============================
def predict_next_day():
    df = load_data()
    df = feature_engineer(df)
    model = load_model()

    X_latest = df[["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]].tail(1)
    y_pred = model.predict(X_latest)[0]
    conf = max(model.predict_proba(X_latest)[0])

    trend = "📈 Uptrend Expected Tomorrow" if y_pred == 1 else "📉 Downtrend Expected Tomorrow"
    return trend, round(conf * 100, 2)
