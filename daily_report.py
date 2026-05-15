from stock_analyzer import StockAnalyzer, AIAnalyzer
from egx_stocks import EGX_STOCKS, STOCK_SECTORS
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent

def quick_analysis(symbol, name):
    try:
        a = StockAnalyzer(symbol, period="1y")
        a.fetch_data()
        a.add_indicators()
        a.generate_signals()
        s = a.signals
        return {
            "symbol": symbol, "name": name,
            "sector": STOCK_SECTORS.get(symbol, ""),
            "price": f"{s['price']:.2f}",
            "price_num": round(s['price'], 2),
            "decision": s['decision'],
            "score": s['score'],
            "rsi": f"{s['rsi']:.1f}" if not pd.isna(s['rsi']) else "--",
            "support": f"{s['support']:.2f}" if s['support'] else "--",
            "resistance": f"{s['resistance']:.2f}" if s['resistance'] else "--",
            "stop_loss": round(s.get('stop_loss', 0), 2) if s.get('stop_loss') else None,
            "target": round(s.get('target', 0), 2) if s.get('target') else None,
            "historical_low": s.get('historical_low'),
        }
    except Exception as e:
        return {"symbol": symbol, "name": name, "error": str(e)[:60]}

def ai_analysis(symbol, name):
    try:
        a = StockAnalyzer(symbol)
        a.fetch_data()
        a.add_indicators()
        a.generate_signals()
        ai = AIAnalyzer()
        ai.train_or_load(a.data, symbol)
        ai_dir, ai_conf = ai.get_signal()
        s = a.signals
        decision = s['decision']
        if ai_dir == "صاعد" and s['score'] >= 2:
            decision = "شراء (دخول) ✅✅"
        elif ai_dir == "هابط" and s['score'] <= -2:
            decision = "بيع (خروج) ❌❌"
        return {
            "symbol": symbol, "name": name,
            "sector": STOCK_SECTORS.get(symbol, ""),
            "price": f"{s['price']:.2f}",
            "decision": decision,
            "score": s['score'],
            "rsi": f"{s['rsi']:.1f}" if not pd.isna(s['rsi']) else "--",
            "support": f"{s['support']:.2f}" if s['support'] else "--",
            "resistance": f"{s['resistance']:.2f}" if s['resistance'] else "--",
            "ai_dir": ai_dir or "--",
            "ai_conf": f"{ai_conf:.0%}" if ai_conf else "--",
        }
    except Exception as e:
        print(f"  خطأ AI {symbol}: {e}")
        return None

def generate_report():
    print(f"🚀 بدء المسح الشامل لأسهم البورصة المصرية...\n")

    with ThreadPoolExecutor(max_workers=10) as executor:
        quick_results = list(executor.map(lambda s: quick_analysis(s[0], s[1]), EGX_STOCKS.items()))

    valid_results = [r for r in quick_results if r and "score" in r]

    sorted_buys = sorted(
        [r for r in valid_results if int(r.get('score', 0)) >= 3],
        key=lambda x: int(x.get('score', 0)),
        reverse=True
    )

    telegram_msg = f"🔔 *تقرير النخبة - البورصة المصرية*\n"
    telegram_msg += f"📅 {datetime.now().strftime('%Y-%m-%d')}\n"
    telegram_msg += "---------------------------------------\n\n"

    if sorted_buys:
        telegram_msg += "🏆 *أقوى الفرص المرتبة حسب ثقة الـ AI:*\n\n"
        for i, stock in enumerate(sorted_buys, 1):
            rank_icon = "⭐" if i <= 3 else "▫️"
            sl = stock.get('stop_loss', 'N/A')
            tg = stock.get('target', 'N/A')
            h_low = stock.get('historical_low', 0)
            dist_from_low = ((stock['price_num'] - h_low) / h_low) * 100 if h_low and h_low > 0 else 0
            telegram_msg += f"{rank_icon} *{stock['symbol']}* - {stock['name']}\n"
            telegram_msg += f"    🔥 القوة: `% {stock['score']}`\n"
            telegram_msg += f"    💰 السعر: `{stock['price_num']:.2f} ج.م`\n"
            telegram_msg += f"    📉 عن القاع: `% {dist_from_low:.1f}+` (قاعه: {h_low:.2f})\n"
            telegram_msg += f"    🎯 الهدف: `{tg}`\n"
            telegram_msg += f"    🛡️ الوقف: `{sl}`\n"
            telegram_msg += "-------------------\n"
    else:
        telegram_msg += "⚠️ لم يتم العثور على أسهم تتخطى معايير الجودة اليوم.\n"

    date_str = datetime.now().strftime('%Y%m%d_%H%M')
    with open(BASE_DIR / f"report_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(f"تقرير البورصة المصرية - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"إجمالي: {len(sorted_buys)} فرصة\n\n")
        for r in sorted_buys:
            f.write(f"{r['symbol']} | {r['decision']} | سعر {r['price_num']} | هدف {r.get('target','')} | وقف {r.get('stop_loss','')}\n")

    print(f"✅ تم إنشاء التقرير والحفظ كـ report_{date_str}.txt")
    from telegram_logger import send_telegram_message
    send_telegram_message(telegram_msg)
    print("✅ تم إرسال أقوى الفرص مرتبة إلى تليجرام.")

if __name__ == "__main__":
    generate_report()
