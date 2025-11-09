import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
import joblib, os


MODEL_PATH = "trained_model.pkl"


# ==================================================
# 1️⃣ Load latest Nifty data safely
# ==================================================
def load_data(ticker="^NSEI", period="3mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    # Flatten multi-index columns (yfinance does this sometimes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Ensure proper numeric Close column
    if "Close" not in df.columns:
        raise ValueError("Close column not found in downloaded data")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Drop null rows and reset index
    df = df.dropna().reset_index()

    return df


# ==================================================
# 2️⃣ Feature Engineering (robust version)
# ==================================================
def feature_engineer(df):
    df = df.copy()

    # Convert Close to clean float Series (strictly 1D)
    close = pd.Series(df["Close"].astype(float).values, name="Close")

    # Technical indicators
    df["SMA_5"] = SMAIndicator(close, window=5, fillna=True).sma_indicator()
    df["SMA_20"] = SMAIndicator(close, window=20, fillna=True).sma_indicator()

    macd = MACD(close, fillna=True)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    rsi = RSIIndicator(close, window=14, fillna=True)
    df["RSI"] = rsi.rsi()

    bb = BollingerBands(close, window=20, fillna=True)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    df = df.dropna().reset_index(drop=True)
    return df


# ==================================================
# 3️⃣ Train & Save model
# ==================================================
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


# ==================================================
# 4️⃣ Load model
# ==================================================
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save()


# ==================================================
# 5️⃣ Predict Next Day
# ==================================================
def predict_next_day():
    df = load_data()
    df = feature_engineer(df)
    model = load_model()

    latest_features = df[["SMA_5", "SMA_20", "MACD", "MACD_signal", "RSI", "BB_high", "BB_low"]].iloc[-1:].values

    y_pred = model.predict(latest_features)[0]
    conf = model.predict_proba(latest_features)[0].max()

    trend = "📈 Uptrend Expected Tomorrow" if y_pred == 1 else "📉 Downtrend Expected Tomorrow"
    return trend, round(conf * 100, 2)
