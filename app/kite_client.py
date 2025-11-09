# kite_client.py
# Simple Kite Connect wrapper with fallback to yfinance.
import os, json
from kiteconnect import KiteConnect
import yfinance as yf

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'utils', 'config.json')

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Please create utils/config.json from utils/config.example.json')
    with open(CONFIG_PATH) as f:
        return json.load(f)

class KiteWrapper:
    def __init__(self):
        self.cfg = {}
        try:
            self.cfg = load_config()
        except:
            self.cfg = {}
        self.kite = None
        api_key = self.cfg.get('kite_api_key')
        token = self.cfg.get('kite_access_token')
        if api_key and token:
            try:
                self.kite = KiteConnect(api_key=api_key)
                self.kite.set_access_token(token)
            except Exception as e:
                print('Kite init failed:', e)
                self.kite = None
    def get_ltp(self, tradingsymbol='NIFTY 50'):
        # Try Kite LTP first (requires correct instrument token mapping)
        if self.kite:
            try:
                # This is a simplified example; in real use map instrument token via instrument lookup
                ltp = self.kite.ltp('NSE:NIFTY 50')
                return float(list(ltp.values())[0]['last_price'])
            except Exception as e:
                print('Kite LTP failed:', e)
        # fallback to yfinance
        try:
            ticker = self.cfg.get('nifty_ticker','^NSEI')
            df = yf.download(ticker, period='5d', interval='1d', auto_adjust=True)
            return float(df['Close'].iloc[-1])
        except Exception as e:
            print('yfinance fallback failed:', e)
            return None

    def place_order_market(self, tradingsymbol, qty, transaction_type='BUY', exchange='NSE', product='MIS'):
        if not self.kite:
            raise RuntimeError('Kite not configured (paper mode or missing credentials)')
        # Example derivative order; in real usage specify correct exchange/symbol
        return self.kite.order_place(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=qty,
            order_type='MARKET',
            product=product
        )
