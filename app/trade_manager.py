# trade_manager.py
# Simple ledger that records trades to local CSV and computes P&L for paper trades.
import os, csv, datetime, pandas as pd, json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'utils', 'config.json')
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)

LEDGER = os.path.join(os.path.dirname(__file__), '..', 'data', 'trades.csv')
os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
if not os.path.exists(LEDGER):
    with open(LEDGER, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','symbol','side','price','qty','mode','status','close_price','pnl'])

def record_trade(symbol, side, price, qty, mode='paper', status='open', note=''):
    ts = datetime.datetime.utcnow().isoformat()
    with open(LEDGER, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, symbol, side, price, qty, mode, status, '', ''])
    return True

def close_trade(index, close_price):
    df = pd.read_csv(LEDGER)
    if index < 0 or index >= len(df):
        return False
    row = df.iloc[index].to_dict()
    entry_price = float(row['price'])
    qty = float(row['qty'])
    side = row['side']
    if side.upper() == 'BUY':
        pnl = (close_price - entry_price) * qty
    else:
        pnl = (entry_price - close_price) * qty
    df.at[index, 'status'] = 'closed'
    df.at[index, 'close_price'] = close_price
    df.at[index, 'pnl'] = pnl
    df.to_csv(LEDGER, index=False)
    return True

def get_ledger():
    return pd.read_csv(LEDGER)
