import schedule
import time
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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

def check_market_hours():
    now = datetime.now()
    today = now.weekday()
    if today in [4, 5]:
        return False
    if 10 <= now.hour < 15:
        return True
    return False

def hourly_market_monitor():
    if not check_market_hours():
        return

    print(f"🔍 [فحص لحظي] مراقبة القيعان والسيولة للأسهم القيادية... {datetime.now().strftime('%H:%M')}")

    for ticker, name in LEADERS_TICKERS.items():
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period="1mo", interval="1h")

            if df.empty:
                continue

            bottom_alert = check_historical_bottoms(ticker, df)
            if bottom_alert:
                send_telegram_message(bottom_alert)
                print(f"✅ تنبيه قاع لـ {ticker}")

            volume_alert = check_volume_spike(ticker, df)
            if volume_alert:
                send_telegram_message(volume_alert)
                print(f"✅ تنبيه سيولة لـ {ticker}")

        except Exception as e:
            print(f"❌ خطأ في مراقبة {ticker}: {e}")

def daily_report_job():
    today = datetime.now().weekday()
    if today in [4, 5]:
        return

    print(f"📊 [تقرير ختامي] حان وقت التقرير اليومي الشامل...")
    try:
        generate_report()
    except Exception as e:
        print(f"❌ خطأ في التقرير اليومي: {e}")

schedule.every().hour.do(hourly_market_monitor)
schedule.every().day.at("16:00").do(daily_report_job)

if __name__ == "__main__":
    print("🤖 الروبوت مبرمج بالكامل ويعمل الآن...")
    print("⏱️ يراقب القيعان كل ساعة أثناء الجلسة، ويرسل التقرير الشامل الساعة 4 عصراً.")

    hourly_market_monitor()

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 تم إيقاف النظام المبرمج.")
            sys.exit()
