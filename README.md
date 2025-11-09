# Nifty Direction App — Full Customized (Kite, Telegram, Google Sheets, Paper/Real switch, Streamlit UI)

**What's included**
- `app/streamlit_app.py` — Web UI (Streamlit) with prediction, trade execution (paper/real switch), Telegram alerts, and PnL chart.
- `app/model.py` — Feature engineering + XGBoost baseline (same as prototype) with utility functions to train/load/predict.
- `app/kite_client.py` — Kite Connect wrapper: fetch LTP / historical OHLC. Falls back to yfinance if credentials missing.
- `app/telegram_client.py` — Telegram bot wrapper to send alerts.
- `app/gsheet_logger.py` — Google Sheets logger using service account credentials (sample).
- `app/trade_manager.py` — Paper/Real trade handling, local CSV ledger, P&L computation.
- `utils/config.example.json` — example config with keys & settings.
- `requirements.txt` — Python deps.

## Quick start
1. Unzip the bundle and `cd` into folder.
2. Create a Python virtualenv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `utils/config.example.json` -> `utils/config.json` and fill your keys:
   - kite_api_key, kite_api_secret, kite_access_token (optional)
   - telegram_bot_token, telegram_chat_id
   - google_service_account (path to JSON), google_sheet_name
   - mode: "paper" or "real"
4. (Optional) Place Google service account JSON at `utils/sa_credentials.json` and reference it from config.
5. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
6. In the UI:
   - Click "Fetch Latest" to load market price (Kite if available).
   - Click "Predict" to compute up/down probability.
   - Click "Execute Trade" to create a paper or real trade (based on config).
   - Click "Send Telegram Alert" to push the signal.
   - View PnL chart and trade ledger.

## Important notes
- This is a prototype. Do NOT deploy live without testing in paper mode.
- Kite Connect orders in `kite_client.py` are examples. Always test in sandbox/paper first.
- Google Sheets logging requires service account sharing access to the target sheet.

If you want, I can push this to a GitHub repo and provide CI/deployment steps. Reply "push" to do that.
