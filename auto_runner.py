import schedule
import time
import sys
import sqlite3
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent))

from daily_report import quick_analysis
from telegram_logger import send_telegram_message
from stock_analyzer import check_historical_bottoms, check_volume_spike
from egx_stocks import EGX_STOCKS, STOCK_SECTORS

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'predictions.db'

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

def save_prediction(sym, direction, confidence, price, score):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS predictions (date TEXT, symbol TEXT, direction TEXT, confidence REAL, price REAL, score INT)")
    conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                 (datetime.now().strftime("%Y-%m-%d"), sym.replace('.CA',''),
                  direction or "--", round(float(confidence or 0), 1),
                  round(float(price or 0), 2), int(score or 0)))
    conn.commit()
    conn.close()

def evaluate_accuracy():
    conn = sqlite3.connect(DB_PATH)
    try:
        import pandas as pd
        df = pd.read_sql("SELECT * FROM predictions", conn)
    except Exception:
        conn.close()
        return None
    conn.close()
    if df.empty:
        return None
    import yfinance as yf
    correct_total = 0
    total = 0
    results = []
    for _, row in df.iterrows():
        sym = row['symbol'] + '.CA'
        pred_dir = row['direction']
        pred_price = row['price']
        if pred_dir in ("--", "", None) or pred_price <= 0:
            continue
        try:
            h = yf.download(sym, period="1mo", progress=False)
            if h.empty or len(h) < 2:
                continue
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.droplevel(1)
            curr = float(h['Close'].iloc[-1])
            actual = "صاعد" if curr > pred_price else "هابط"
            correct = 1 if actual == pred_dir else 0
            correct_total += correct
            total += 1
            results.append({**row.to_dict(), "current": round(curr, 2), "actual": actual, "correct": correct})
        except Exception:
            continue
    if total == 0:
        return None
    return {"total": total, "correct": correct_total, "pct": round(correct_total / total * 100, 1),
            "details": sorted(results, key=lambda x: x['date'], reverse=True)}

def ai_analysis(symbol, name):
    try:
        from stock_analyzer import StockAnalyzer, AIAnalyzer
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
            decision = "شراء (دخول)"
        elif ai_dir == "هابط" and s['score'] <= -2:
            decision = "بيع (خروج)"
        return {
            "symbol": symbol, "name": name,
            "sector": STOCK_SECTORS.get(symbol, ""),
            "price": s['price'], "score": s['score'],
            "decision": decision, "rsi": s['rsi'],
            "stop_loss": s.get('stop_loss'), "target": s.get('target'),
            "historical_low": s.get('historical_low'),
            "proximity_to_low": s.get('proximity_to_low'),
            "ai_dir": ai_dir or "--", "ai_conf": ai_conf or 0,
            "reasons": s.get('reasons', [])
        }
    except Exception as e:
        return None

def format_report(sorted_stocks, acc=None):
    today_str = datetime.now().strftime("%Y-%m-%d")
    msg = f"[DATA] تقرير النخبة - البورصة المصرية\n"
    msg += f"تاريخ اليوم: {today_str}\n"
    msg += "========================================\n\n"

    strong_buy = [s for s in sorted_stocks if s['score'] >= 75]
    medium_buy = [s for s in sorted_stocks if 50 <= s['score'] < 75]
    watchlist = [s for s in sorted_stocks if 0 <= s['score'] < 50]

    if strong_buy:
        msg += "الفرص الذهبية (شراء قوي):\n"
        for s in strong_buy:
            msg += f"  {s['symbol']} - {s['name']}\n"
            msg += f"    التقييم: {s['score']}%\n"
            msg += f"    السعر: {s['price']:.2f} ج.م\n"
            if s.get('ai_dir') and s['ai_dir'] != '--':
                msg += f"    AI: {s['ai_dir']} (ثقة {s['ai_conf']*100:.0f}%)\n"
            if s.get('target'):
                msg += f"    الهدف: {s['target']:.2f}\n"
            if s.get('stop_loss'):
                msg += f"    الوقف: {s['stop_loss']:.2f}\n"
            if s.get('reasons'):
                for r in s['reasons'][:3]:
                    msg += f"      - {r}\n"
            msg += "    -------------------------\n"
        msg += "\n"

    if medium_buy:
        msg += "فرص مضاربية (ثقة متوسطة):\n"
        for s in medium_buy:
            msg += f"  {s['symbol']} - {s['name']} | تقييم: {s['score']}%\n"
            msg += f"    السعر: {s['price']:.2f} | الهدف: {s.get('target', 0):.2f}\n"
            if s.get('ai_dir') and s['ai_dir'] != '--':
                msg += f"    AI: {s['ai_dir']}\n"
            msg += "    -------------------------\n"
        msg += "\n"

    if watchlist:
        msg += "أسهم تحت المراقبة (انتظار):\n"
        for s in watchlist[:10]:
            msg += f"  {s['symbol']} (تقييم: {s['score']}%)\n"
        if len(watchlist) > 10:
            msg += f"  ...و {len(watchlist)-10} أخرى\n"

    if acc:
        msg += "\n========================================\n"
        msg += f"دقة التوقعات السابقة: {acc['pct']}% ({acc['correct']}/{acc['total']})\n"

    msg += "\n========================================\n"
    msg += "تم التحليل آلياً بواسطة المحرك المطور"
    return msg

def daily_report_job():
    today = datetime.now().weekday()
    if today in [4, 5]:
        return

    print(f"[DATA] بدء المسح الشامل للأسهم...")

    all_results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        fut = {executor.submit(quick_analysis, s, n): s for s, n in EGX_STOCKS.items()}
        for f in as_completed(fut):
            r = f.result()
            if r and "score" in r:
                all_results.append(r)

    scored = sorted(
        [r for r in all_results if int(r.get('score', 0)) >= 0],
        key=lambda x: int(x.get('score', 0)), reverse=True
    )

    ai_scored = []
    high_score = [r for r in scored if int(r.get('score', 0)) >= 3]
    if high_score:
        print(f"[OK] تشغيل AI لـ {len(high_score)} سهماً...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            fut = {executor.submit(ai_analysis, r['symbol'], r['name']): r for r in high_score}
            for f in as_completed(fut):
                r = f.result()
                if r:
                    ai_scored.append(r)
                    save_prediction(r['symbol'], r.get('ai_dir'), r.get('ai_conf'),
                                    r.get('price'), r.get('score'))

    merged = {r['symbol']: r for r in ai_scored}
    for r in scored:
        sym = r['symbol']
        if sym in merged:
            merged[sym].setdefault('score', int(r.get('score', 0)))
        else:
            r['ai_dir'] = '--'
            r['ai_conf'] = 0
            r['stop_loss'] = r.get('stop_loss')
            r['target'] = r.get('target')
            r['reasons'] = r.get('reasons', [])
            merged[sym] = r

    sorted_stocks = sorted(merged.values(), key=lambda x: x.get('score', 0), reverse=True)

    acc = evaluate_accuracy()
    msg = format_report(sorted_stocks, acc)

    send_telegram_message(msg)

    date_str = datetime.now().strftime('%Y%m%d_%H%M')
    with open(BASE_DIR / f"report_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(f"تقرير البورصة المصرية - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"إجمالي: {len(sorted_stocks)}\n\n")
        for r in sorted_stocks[:20]:
            f.write(f"{r['symbol']} | تقييم {r.get('score',0)} | سعر {r.get('price','')}")
            if r.get('target'):
                f.write(f" | هدف {r['target']}")
            f.write("\n")

    print(f"[OK] تم إرسال التقرير - {len(sorted_stocks)} سهماً")
    if acc:
        print(f"[OK] دقة التوقعات: {acc['pct']}% ({acc['correct']}/{acc['total']})")

schedule.every().hour.do(hourly_market_monitor)
schedule.every().day.at("16:00").do(daily_report_job)

if __name__ == "__main__":
    print("[OK] الروبوت مبرمج ويعمل الآن...")
    print("[OK] يراقب القيعان كل ساعة - التقرير الشامل الساعة 4 عصراً")

    hourly_market_monitor()

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n[STOP] تم إيقاف النظام.")
            sys.exit()
