# model.py
# Feature engineering, training and prediction utilities

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import xgboost as xgb
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'xgb_model.joblib')

def fetch_data(ticker='^NSEI', period='10y'):
    df = yf.download(ticker, period=period, interval='1d', auto_adjust=True)
    df.dropna(inplace=True)
    return df

def feature_engineer(df):
    df = df.copy()
    df['return1'] = df['Close'].pct_change()
    df['ret_2'] = df['Close'].pct_change(2)
    df['dayofweek'] = df.index.dayofweek
    df['sma5'] = SMAIndicator(df['Close'], window=5).sma_indicator()
    df['sma20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['sma_diff'] = df['sma5'] - df['sma20']
    df['rsi14'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_h'] - df['bb_l']) / df['Close']
    df['atr14'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

def train_and_save(df=None):
    if df is None:
        df = fetch_data()
        df = feature_engineer(df)
    features = ['return1','ret_2','dayofweek','sma5','sma20','sma_diff',
                'rsi14','macd','macd_signal','macd_hist','bb_width','atr14','Volume']
    X = df[features]
    y = df['target']
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric='logloss')
    model.fit(X.iloc[:-1], y.iloc[:-1])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, features), MODEL_PATH)
    return model, features

def load_model():
    if os.path.exists(MODEL_PATH):
        model, features = joblib.load(MODEL_PATH)
        return model, features
    else:
        return train_and_save()

def predict_next_day(latest_df=None):
    model, features = load_model()
    if latest_df is None:
        df = fetch_data()
        df = feature_engineer(df)
        latest = df.iloc[[-1]]
    else:
        latest = latest_df.copy()
    X_latest = latest[features]
    prob = model.predict_proba(X_latest)[0][1]
    label = int(model.predict(X_latest)[0])
    return {'prob_up': float(prob), 'pred': int(label)}
