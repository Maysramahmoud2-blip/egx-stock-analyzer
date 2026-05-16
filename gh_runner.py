import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if BOT_TOKEN and CHAT_ID:
    with open(Path(__file__).parent / "telegram_config.txt", "w") as f:
        f.write(f"BOT_TOKEN={BOT_TOKEN}\nCHAT_ID={CHAT_ID}\n")

from datetime import datetime
from daily_report import generate_report
from telegram_logger import send_telegram_message
from stock_analyzer import check_historical_bottoms, check_volume_spike

LEADERS_TICKERS = {
    "COMI.CA": "البنك التجاري الدولي",
    "FWRY.CA": "فوري",
    "SKPC.CA": "سيدي كرير للبتروكيماويات",
    "TMGH.CA": "طلعت مصطفى",
    "ABUK.CA": "أبو قير للأسمدة",
    "EAST.CA": "الشرقية - إيسترن كومباني"
}

def check_market_hours(dt=None):
    now = dt or datetime.now()
    if now.weekday() in [4, 5]:
        return False
    return 10 <= now.hour < 15

def hourly_market_monitor():
    if not check_market_hours():
        return
    print(f"[OK] فحص لحظي {datetime.now().strftime('%H:%M')}")
    for ticker, name in LEADERS_TICKERS.items():
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period="1mo", interval="1h")
            if df.empty:
                continue
            bottom_alert = check_historical_bottoms(ticker, df)
            if bottom_alert:
                send_telegram_message(bottom_alert)
                print(f"[OK] تنبيه قاع {ticker}")
            volume_alert = check_volume_spike(ticker, df)
            if volume_alert:
                send_telegram_message(volume_alert)
                print(f"[OK] تنبيه سيولة {ticker}")
        except Exception as e:
            print(f"[FAIL] خطأ {ticker}: {e}")

def daily_report_job():
    if datetime.now().weekday() in [4, 5]:
        return
    print(f"[DATA] تقرير يومي شامل...")
    try:
        generate_report()
    except Exception as e:
        print(f"[FAIL] خطأ في التقرير: {e}")

if __name__ == "__main__":
    now = datetime.now()
    msg = f"[OK] GH Runner {now.strftime('%Y-%m-%d %H:%M')}"

    if now.hour == 16 and now.minute < 30:
        daily_report_job()
        msg += " + تقرير يومي"
    elif check_market_hours():
        hourly_market_monitor()
        msg += " + مراقبة لحظية"
    else:
        print("[!] خارج أوقات السوق")

    print(msg)
