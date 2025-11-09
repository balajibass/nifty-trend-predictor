# telegram_client.py
import os, json, requests
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'utils', 'config.json')

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)

class TelegramClient:
    def __init__(self):
        self.cfg = load_config()
        self.bot_token = self.cfg.get('telegram_bot_token')
        self.chat_id = self.cfg.get('telegram_chat_id')
    def send(self, text):
        if not (self.bot_token and self.chat_id):
            print('Telegram not configured in config.json; skipping send.')
            return False
        url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        payload = {'chat_id': self.chat_id, 'text': text}
        try:
            r = requests.post(url, json=payload, timeout=10)
            return r.status_code == 200
        except Exception as e:
            print('Telegram send failed:', e)
            return False
