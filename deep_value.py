import yfinance as yf
import pandas as pd
import numpy as np
import csv, os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from stock_analyzer import AIAnalyzer

CSV_PATH = os.path.join(os.path.dirname(__file__), 'egx_all.csv')

def compute_indicators(df):
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    d['sma_20'] = d['close'].rolling(20).mean()
    d['ema_12'] = d['close'].ewm(span=12).mean()
    d['ema_26'] = d['close'].ewm(span=26).mean()
    d['macd'] = d['ema_12'] - d['ema_26']
    delta = d['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['rsi'] = 100 - (100 / (1 + gain / loss))
    return d

def load_stocks():
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        return [(r['ticker'].strip() + '.CA', r['name'].strip(), r['sector'].strip()) for r in reader]

def analyze(symbol, name):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period="2y")
        if hist.empty:
            return None
        price = hist['Close'].iloc[-1]
        low_52w = info.get('fiftyTwoWeekLow', price)
        high_52w = info.get('fiftyTwoWeekHigh', price)
        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        div = info.get('dividendYield')
        mc = info.get('marketCap')
        from_low = (price - low_52w) / low_52w * 100
        from_high = (high_52w - price) / high_52w * 100

        score = 0
        if from_low <= 15: score += 3
        elif from_low <= 30: score += 1
        if from_high >= 50: score += 2
        elif from_high >= 30: score += 1
        if pe and pe < 8: score += 3
        elif pe and pe < 15: score += 1
        if pb and pb < 1: score += 2
        elif pb and pb < 1.5: score += 1
        if div and div > 0.03: score += 1

        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss)).iloc[-1]
        if rsi < 35: score += 2
        elif rsi < 45: score += 1
        score = min(score, 10)

        if score >= 7: decision = "🟢 قوي"
        elif score >= 5: decision = "🟡 متوسط"
        elif score >= 3: decision = "🔵 مراقبة"
        else: decision = "🔴 تجنب"

        ai_dir = None
        ai_conf = 0
        if score >= 5:
            try:
                ai_df = compute_indicators(hist)
                ai = AIAnalyzer()
                ai.train(ai_df)
                ai_dir, ai_conf = ai.get_signal()
            except:
                pass

        return {
            "symbol": symbol.replace('.CA',''), "name": name[:35],
            "price": price, "from_low": from_low, "from_high": from_high,
            "pe": pe, "pb": pb, "rsi": rsi, "score": score, "decision": decision,
            "mc": mc, "ai_dir": ai_dir, "ai_conf": ai_conf,
        }
    except:
        return None

def generate_report():
    stocks = load_stocks()
    results = []
    print(f"\nتحليل {len(stocks)} سهم...")
    for symbol, name, sector in stocks:
        r = analyze(symbol, name)
        if r:
            results.append(r)
    results.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "="*90)
    print(f"            بحث استثماري — البورصة المصرية — {datetime.now().strftime('%Y-%m-%d')}")
    ai_count = sum(1 for r in results if r['ai_dir'])
    print(f"            إجمالي الأسهم المحللة: {len(results)}")
    if ai_count:
        print(f"            تحليل AI: {ai_count} سهم | تدريب LSTM على MPS")
    print("="*90)

    strong = [r for r in results if r['decision'] == '🟢 قوي']
    medium = [r for r in results if r['decision'] == '🟡 متوسط']
    watch = [r for r in results if r['decision'] == '🔵 مراقبة']

    print(f"\n🟢 فرص استثمار قوية ({len(strong)}):")
    print("-"*90)
    for r in strong:
        pe_s = f"{r['pe']:.1f}" if r['pe'] else "--"
        pb_s = f"{r['pb']:.2f}" if r['pb'] else "--"
        ai_s = f" | AI {r['ai_dir']} (ثقة {r['ai_conf']:.0%})" if r['ai_dir'] else ""
        print(f"  {r['symbol']:8s} {r['name']:35s} | سعر {r['price']:>8.2f} | P/E {pe_s:>5s} | P/B {pb_s:>5s} | RSI {r['rsi']:.0f} | +{r['from_low']:.0f}% عن القاع{ai_s}")

    print(f"\n🟡 فرص متوسطة ({len(medium)}):")
    print("-"*90)
    for r in medium[:10]:
        pe_s = f"{r['pe']:.1f}" if r['pe'] else "--"
        ai_s = f" | AI {r['ai_dir']} (ثقة {r['ai_conf']:.0%})" if r['ai_dir'] else ""
        print(f"  {r['symbol']:8s} {r['name']:35s} | سعر {r['price']:>8.2f} | P/E {pe_s:>5s} | RSI {r['rsi']:.0f}{ai_s}")
    if watch:
        print(f"\n🔵 مراقبة ({len(watch)}):")
        print("-"*90)
        for r in watch[:10]:
            pe_s = f"{r['pe']:.1f}" if r['pe'] else "--"
            print(f"  {r['symbol']:8s} {r['name']:35s} | سعر {r['price']:>8.2f} | P/E {pe_s:>5s} | RSI {r['rsi']:.0f}")
        if len(watch) > 10:
            print(f"  ...و {len(watch)-10} آخرون")

    print("="*90)

    date_str = datetime.now().strftime('%Y%m%d')
    with open(f"/Users/maysre/AI-Learning/StockProject/deep_value_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(f"تقرير الاستثمار - {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"إجمالي: {len(results)} سهم\n\n")
        for r in strong + medium + watch:
            ai_s = f" | AI {r['ai_dir']} (ثقة {r['ai_conf']:.0%})" if r['ai_dir'] else ""
            f.write(f"{r['symbol']} | {r['decision']} | سعر {r['price']:.2f} | P/E {r['pe']} | P/B {r['pb']} | RSI {r['rsi']:.0f} | +{r['from_low']:.0f}%{ai_s}\n")

    print(f"\nتم الحفظ: deep_value_{date_str}.txt")
    print("="*90)

if __name__ == "__main__":
    generate_report()
