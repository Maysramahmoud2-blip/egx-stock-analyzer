import asyncio
import csv
import io
from datetime import datetime
from pathlib import Path
from telegram import Bot
from telegram.error import TelegramError

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / 'telegram_config.txt'

TOKEN_KEY = "BOT_TOKEN"
CHAT_KEY = "CHAT_ID"

_saved_token = None
_saved_chat = None

def load_config():
    global _saved_token, _saved_chat
    if not CONFIG_PATH.exists():
        return False
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith(TOKEN_KEY + '='):
                _saved_token = line.split('=', 1)[1].strip()
            elif line.startswith(CHAT_KEY + '='):
                _saved_chat = line.split('=', 1)[1].strip()
    return bool(_saved_token and _saved_chat)

def save_config(token, chat_id):
    with open(CONFIG_PATH, 'w') as f:
        f.write(f"{TOKEN_KEY}={token}\n")
        f.write(f"{CHAT_KEY}={chat_id}\n")
    global _saved_token, _saved_chat
    _saved_token = token
    _saved_chat = chat_id

def send_now(message):
    if not load_config():
        return False, "الإعدادات مش موجودة — سوّي البوت أولًا"
    try:
        bot = Bot(token=_saved_token)
        asyncio.run(bot.send_message(chat_id=_saved_chat, text=message))
        return True, "تم الإرسال ✅"
    except Exception as e:
        return False, f"خطأ: {e}"

def send_report():
    reports = sorted(BASE_DIR.glob('report_*.txt'))
    summary = ""
    if reports:
        with open(reports[-1], encoding='utf-8') as f:
            summary = f.read()[:3500]
    else:
        summary = "لا يوجد تقرير اليوم — شغّل التحديث من Dashboard"

    today = datetime.now().strftime('%Y-%m-%d %H:%M')
    msg = f"📊 تقرير البورصة المصرية\n📅 {today}\n\n{summary}"
    return send_now(msg)

def send_report_from_cache(buys, sells, waits, ai_results):
    lines = [f"📊 تقرير البورصة — {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    if buys:
        lines.append(f"\n🟢 شراء ({len(buys)}):")
        for r in buys[:10]:
            ai = ai_results.get(r['symbol'], {})
            ai_txt = f" | AI {ai.get('ai_dir','')}" if ai else ""
            lines.append(f"  {r['symbol']} - {r['price']} | RSI {r.get('rsi','')}{ai_txt}")
    if sells:
        lines.append(f"\n🔴 بيع ({len(sells)}):")
        for r in sells[:5]:
            lines.append(f"  {r['symbol']} - {r['price']} | RSI {r.get('rsi','')}")
    lines.append(f"\n⏳ مراقبة: {len(waits)}")
    return send_now("\n".join(lines))

def send_alert(symbol, name, decision, price, reason=""):
    msg = (
        f"🔔 تنبيه: {symbol}\n"
        f"📌 {name}\n"
        f"القرار: {decision}\n"
        f"السعر: {price}\n"
    )
    if reason:
        msg += f"السبب: {reason}\n"
    msg += f"⏰ {datetime.now().strftime('%H:%M')}"
    return send_now(msg)

def send_portfolio(results):
    if not results:
        return send_now("لا توجد نتائج للمحفظة")
    lines = ["📊 تقييم المحفظة", "=" * 20]
    for r in results:
        if 'error' in r:
            continue
        ai_txt = f"{r.get('ai_dir','')} ({r.get('ai_conf',0):.0%})" if r.get('ai_dir') else "—"
        perf = f"{r['perf_1m']:+.1f}%" if r.get('perf_1m') is not None else "--"
        lines.append(
            f"\n{r['symbol']}: {r.get('trend','')}"
            f"\n  السعر: {r['price']} | شهري: {perf}"
            f"\n  RSI: {r['rsi']:.0f} | AI: {ai_txt}"
        )
    return send_now("\n".join(lines))


def send_bottom_alerts():
    import yfinance as yf
    from stock_analyzer import LEADERS_TICKERS, check_historical_bottoms
    alerts = []
    for sym in LEADERS_TICKERS:
        try:
            df = yf.download(sym, period="2y", progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, type(df.columns)):
                df.columns = [c.lower() for c in df.columns]
            msg = check_historical_bottoms(sym, df)
            if msg:
                alerts.append(msg)
        except Exception as e:
            print(f"خطأ في فحص {sym}: {e}")
    if alerts:
        return send_now("\n\n".join(alerts))
    return send_now("✅ لا توجد أسهم قيادية قريبة من القاع اليوم.")
