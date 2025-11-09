import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import yfinance as yf
import joblib
import os

MODEL_PATH = "app/trained_model.pkl"

# ========== 1️⃣ Feature Engineering ==========
def feature_engineer(df):
    df = df.copy()

    # Ensure Close column exists and is 1D
    if "Close" not in df.columns:
        raise ValueError("❌ DataFrame does not contain 'Close' column")

    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)

    close_series = df["Close"].astype(float)

    # Technical Indicators
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


# ========== 2️⃣ Load Market Data ==========
def load_data(ticker="^NSEI", period="3mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df


# ========== 3️⃣ Train Model ==========
def train_and_save():
    from sklearn.ensemble import RandomForestClassifier
    df = load_data()
    df = feature_engineer(df)

    # Create Target (simple up/down)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    features = ["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]
    X = df[features]
    y = df["Target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


# ========== 4️⃣ Load or Re-train ==========
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save()


# ========== 5️⃣ Predict Next Day ==========
def predict_next_day():
    df = load_data()
    df = feature_engineer(df)
    model = load_model()

    features = ["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]
    X = df[features].tail(1)

    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    trend = "📈 UP" if prediction == 1 else "📉 DOWN"
    return trend, round(confidence * 100, 2)
