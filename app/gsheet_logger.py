# gsheet_logger.py
# Basic Google Sheets logger using gspread and a service account JSON.
import os, json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'utils', 'config.json')

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)

class GSheetLogger:
    def __init__(self):
        self.cfg = load_config()
        self.sa_path = self.cfg.get('google_service_account_json')
        self.sheet_name = self.cfg.get('google_sheet_name')
        self.client = None
        self.sheet = None
        if self.sa_path and os.path.exists(self.sa_path):
            scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.sa_path, scope)
            self.client = gspread.authorize(creds)
            try:
                self.sheet = self.client.open(self.sheet_name).sheet1
            except Exception as e:
                print('Open sheet failed:', e)
                self.sheet = None
    def log_trade(self, trade_dict):
        # trade_dict: {timestamp, type, price, qty, mode, note}
        if self.sheet:
            try:
                self.sheet.append_row([trade_dict.get('timestamp'), trade_dict.get('type'),
                                      trade_dict.get('price'), trade_dict.get('qty'),
                                      trade_dict.get('mode'), trade_dict.get('note')])
                return True
            except Exception as e:
                print('Append row failed:', e)
                return False
        else:
            print('GSheet not configured or sheet unavailable.')
            return False
