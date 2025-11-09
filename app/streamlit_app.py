# streamlit_app.py
import streamlit as st
import json, os, time
from app.model import predict_next_day, load_model, fetch_data
from app.kite_client import KiteWrapper
from app.telegram_client import TelegramClient
from app.gsheet_logger import GSheetLogger
from app.trade_manager import record_trade, get_ledger
import pandas as pd
import plotly.express as px

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'utils', 'config.json')
def load_config():
    if not os.path.exists(CONFIG_PATH):
        st.warning('Please copy utils/config.example.json -> utils/config.json and fill credentials.')
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)

st.set_page_config(page_title='Nifty Direction App', layout='wide')
st.title('Nifty Direction Predictor — Full App')

cfg = load_config()
mode = cfg.get('mode', 'paper')
st.sidebar.write('Mode:')
mode = st.sidebar.selectbox('Paper or Real', options=['paper','real'], index=0 if mode=='paper' else 1)
st.sidebar.write('Current mode: ' + mode)

kite = KiteWrapper()
tg = TelegramClient()
gsheet = GSheetLogger()

col1, col2 = st.columns([2,1])
with col1:
    st.header('Market & Prediction')
    if st.button('Fetch Latest Price'):
        price = kite.get_ltp()
        st.success(f'Latest price: {price}')
        st.session_state['latest_price'] = price
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            res = predict_next_day()
            st.session_state['last_prediction'] = res
            st.success(f"Next-day UP probability: {res['prob_up']:.2f}  Pred: {'UP' if res['pred']==1 else 'DOWN'}")
            st.write(res)
    if 'last_prediction' in st.session_state:
        res = st.session_state['last_prediction']
        st.metric('UP probability', f"{res['prob_up']:.2f}", delta=None)
        st.write(res)
    st.markdown('---')
    st.header('Actions')
    colA, colB = st.columns(2)
    with colA:
        if st.button('Send Telegram Alert'):
            if 'last_prediction' in st.session_state:
                res = st.session_state['last_prediction']
                txt = f"Signal: {'UP' if res['pred']==1 else 'DOWN'} | UP_prob: {res['prob_up']:.2f}"
                ok = tg.send(txt)
                st.success('Telegram sent' if ok else 'Telegram failed or not configured')
    with colB:
        if st.button('Execute Trade (Market)'):
            if 'latest_price' not in st.session_state:
                st.warning('Fetch latest price first')
            else:
                price = st.session_state['latest_price']
                pred = st.session_state.get('last_prediction', {'pred':1})
                side = 'BUY' if pred['pred']==1 else 'SELL'
                qty = cfg.get('default_lot_size', 1)
                # if real mode, attempt kite order (example)
                if mode == 'real':
                    try:
                        order = kite.place_order_market('NIFTY 21NOV24000CE', qty, transaction_type=side)
                        note = f'Real order id: {order}'
                        st.success('Real order placed: ' + str(order))
                    except Exception as e:
                        st.error('Real order failed: ' + str(e))
                        note = 'Real order failed: ' + str(e)
                else:
                    note = 'Paper trade recorded'
                record_trade('NIFTY', side, price, qty, mode=mode, note=note)
                st.success('Trade recorded (' + mode + ')')

with col2:
    st.header('Ledger & PnL')
    df = get_ledger()
    st.write(df.tail(10))
    if not df.empty:
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
        daily = df.groupby(df['timestamp'].str[:10])['pnl'].sum().reset_index()
        if not daily.empty:
            fig = px.bar(daily, x='timestamp', y='pnl', title='Daily PnL (paper trades)')
            st.plotly_chart(fig)

st.markdown('---')
st.header('Utilities')
st.write('Model & Data utilities:')
if st.button('Retrain model (this will fetch 10y data and overwrite saved model)'):
    with st.spinner('Training...'):
        model, features = load_model()
        st.success('Model retrained/loaded (see logs in console).')
