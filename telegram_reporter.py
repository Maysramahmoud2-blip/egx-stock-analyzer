import pandas as pd
from datetime import datetime
import telebot
from pathlib import Path

from stock_analyzer import StockAnalyzer, check_historical_bottoms, check_volume_spike

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / 'telegram_config.txt'

def load_config():
    token = None
    chat_id = None
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith('BOT_TOKEN='):
                    token = line.split('=', 1)[1].strip()
                elif line.startswith('CHAT_ID='):
                    chat_id = line.split('=', 1)[1].strip()
    return token, chat_id

TELEGRAM_TOKEN, CHAT_ID = load_config()
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise Exception("لم يتم العثور على إعدادات التليجرام في telegram_config.txt")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

EGX_WATCHLIST = [
    "COMI.CA", "EAST.CA", "TMGH.CA", "ABUK.CA", "MFOT.CA",
    "FWRY.CA", "EKHO.CA", "HRHO.CA", "SWDY.CA", "AMOC.CA"
]


def format_telegram_message(scanned_stocks):
    sorted_stocks = sorted(scanned_stocks, key=lambda x: x['score'], reverse=True)

    today_str = datetime.now().strftime("%Y-%m-%d")

    msg = f"🔔 **التقرير اليومي لتحليل الأسهم (EGX)**\n"
    msg += f"📅 تاريخ اليوم: `{today_str}`\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━\n\n"

    strong_buy = [s for s in sorted_stocks if s['score'] >= 75]
    medium_buy = [s for s in sorted_stocks if 50 <= s['score'] < 75]
    watchlist = [s for s in sorted_stocks if 0 <= s['score'] < 50]

    if strong_buy:
        msg += f"🔥 **الفرص الذهبية الفائقة (شراء قوي):**\n"
        for s in strong_buy:
            msg += f"📌 السهم: *{s['symbol']}*\n"
            msg += f"📈 التقييم الإجمالي: `{s['score']}%`\n"
            msg += f"💰 السعر الحالي: `{s['price']:.2f} ج.م`\n"
            if 'ai' in s:
                msg += f"🤖 توقع الـ AI: *{s['ai']}*\n"
            msg += f"📉 قاعه التاريخي: `{s['hist_low']:.2f} ج.م` (يبعد `% {s['proximity']}+`)\n"
            msg += f"🎯 الهدف الفني: `{s['target']:.2f} ج.م`\n"
            msg += f"🛡️ وقف الخسارة: `{s['stop_loss']:.2f} ج.م`\n"
            msg += f"💡 *أهم الأسباب:*\n"
            for reason in s['reasons'][:3]:
                msg += f"   • {reason}\n"
            msg += f"---------------------\n"
        msg += f"\n"

    if medium_buy:
        msg += f"⚡ **فرص مضاربية (ثقة متوسطة):**\n"
        for s in medium_buy:
            msg += f"🔹 السهم: *{s['symbol']}* | التقييم: `{s['score']}%`\n"
            msg += f"💵 السعر: `{s['price']:.2f} ج.م` | المستهدف: `{s['target']:.2f} ج.م`\n"
            if 'ai' in s:
                msg += f"🤖 الـ AI يتوقع: {s['ai']}\n"
            msg += f"---------------------\n"
        msg += f"\n"

    if watchlist:
        msg += f"⏳ **أسهم تحت المراقبة (انتظار):**\n"
        for s in watchlist:
            msg += f"👀 *{s['symbol']}* (تقييم: `{s['score']}%`)\n"

    msg += f"\n━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🚀 *تم التحليل والترتيب آلياً بواسطة المحرك المطور*"

    return msg


def run_daily_report():
    print("⏳ بدأ محرك الفحص اليومي للأسهم وتجهيز التقرير...")
    scanned_results = []

    for symbol in EGX_WATCHLIST:
        try:
            print(f"🔍 جاري فحص وتحليل سهم: {symbol}")
            analyzer = StockAnalyzer(symbol, period="2y")

            analyzer.fetch_data()
            analyzer.add_indicators()

            volume_alert = check_volume_spike(symbol, analyzer.data)
            if volume_alert:
                bot.send_message(CHAT_ID, volume_alert, parse_mode="Markdown")

            bottom_alert = check_historical_bottoms(symbol, analyzer.data)
            if bottom_alert:
                bot.send_message(CHAT_ID, bottom_alert, parse_mode="Markdown")

            analyzer.run()
            sig = analyzer.signals

            stock_info = {
                'symbol': symbol,
                'score': sig['score'],
                'price': sig['price'],
                'target': sig['target'],
                'stop_loss': sig['stop_loss'],
                'hist_low': sig['historical_low'],
                'proximity': sig['proximity_to_low'],
                'reasons': sig['reasons']
            }
            if 'ai_prediction' in sig:
                stock_info['ai'] = sig['ai_prediction']

            scanned_results.append(stock_info)

        except Exception as e:
            print(f"❌ فشل فحص السهم {symbol}: {e}")
            continue

    if scanned_results:
        final_report = format_telegram_message(scanned_results)

        try:
            bot.send_message(CHAT_ID, final_report, parse_mode="Markdown")
            print("✅ تم إرسال التقرير اليومي المطور إلى تليجرام بنجاح!")
        except Exception as e:
            print(f"❌ فشل إرسال التقرير عبر تليجرام: {e}")
            if "message is too long" in str(e).lower():
                print("⚠️ الرسالة طويلة، جاري إرسالها على أجزاء...")
                for chunk in [final_report[i:i+4000] for i in range(0, len(final_report), 4000)]:
                    bot.send_message(CHAT_ID, chunk, parse_mode="Markdown")
    else:
        print("⚠️ لم يتم تجميع أي بيانات لعدم وجود اتصالات كافية بالسوق.")


if __name__ == "__main__":
    run_daily_report()
