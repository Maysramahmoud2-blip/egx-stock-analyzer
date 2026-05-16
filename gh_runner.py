import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if BOT_TOKEN and CHAT_ID:
    with open(Path(__file__).parent / "telegram_config.txt", "w") as f:
        f.write(f"BOT_TOKEN={BOT_TOKEN}\nCHAT_ID={CHAT_ID}\n")

from datetime import datetime
from auto_runner import hourly_market_monitor, daily_report_job, check_market_hours

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
