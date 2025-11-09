import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import joblib
import os

MODEL_PATH = "app/models/nifty_model.pkl"

# -----------------------------
# Load NIFTY data
# -----------------------------
def fetch_data(symbol="^NSEI", period="3mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    return df


# -----------------------------
# Feature Engineering
# -----------------------------
def feature_engineer(df):
    df = df.copy()

    # Ensure Close column is numeric and 1D
    if "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Close"] = np.array(df["Close"]).reshape(-1)   # flatten to 1D
    else:
        raise ValueError("❌ DataFrame does not contain 'Close' column")

    # Technical Indicators (with safe casting)
    close_series = pd.Series(df["Close"].astype(float))

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

# -----------------------------
# Model Trainer (optional retraining)
# -----------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save():
    df = fetch_data()
    df = feature_engineer(df)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    X = df[["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


# -----------------------------
# Load or Train Model
# -----------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        model = train_and_save()
    else:
        model = joblib.load(MODEL_PATH)
    return model


# -----------------------------
# Predict Next Day Trend
# -----------------------------
def predict_next_day(symbol="^NSEI"):
    df = fetch_data(symbol)
    df = feature_engineer(df)
    model = load_model()

    latest = df.iloc[-1][["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]].values.reshape(1, -1)
    pred = model.predict(latest)[0]

    if pred == 1:
        return "📈 Uptrend expected tomorrow"
    else:
        return "📉 Downtrend expected tomorrow"
