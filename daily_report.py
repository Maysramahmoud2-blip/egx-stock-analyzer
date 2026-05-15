from stock_analyzer import StockAnalyzer, AIAnalyzer
from egx_stocks import EGX_STOCKS, STOCK_SECTORS
import pandas as pd
from datetime import datetime

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
            "decision": s['decision'],
            "score": s['score'],
            "rsi": f"{s['rsi']:.1f}" if not pd.isna(s['rsi']) else "--",
            "support": f"{s['support']:.2f}" if s['support'] else "--",
            "resistance": f"{s['resistance']:.2f}" if s['resistance'] else "--",
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
        ai.train(a.data)
        ai_dir, ai_conf = ai.get_signal()
        s = a.signals
        decision = s['decision']
        if ai_dir == "صاعد" and s['score'] >= 1:
            decision = "شراء (دخول) ✅✅"
        elif ai_dir == "هابط" and s['score'] <= -1:
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
    except:
        return None

def generate_report():
    print("تحليل سريع لكل الأسهم...\n")
    quick_results = []
    for symbol, name in EGX_STOCKS.items():
        r = quick_analysis(symbol, name)
        quick_results.append(r)
        print(f"  {symbol:12s} {r.get('decision', r.get('error','?'))}")
    buys = [r for r in quick_results if "شراء" in r.get("decision", "")]
    top = sorted(buys, key=lambda x: int(x.get("score", 0)), reverse=True)[:10]
    top += [r for r in quick_results if "بيع" in r.get("decision", "")][:5]

    print(f"\nتحليل AI لأفضل {len(top)} سهم...")
    ai_results = []
    for r in top:
        print(f"  {r['symbol']} AI...")
        ar = ai_analysis(r["symbol"], r["name"])
        if ar:
            ai_results.append(ar)

    print("\n" + "="*75)
    print(f"           تقرير البورصة المصرية — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*75)

    buys_all = [r for r in quick_results if "شراء" in r.get("decision", "")]
    sells_all = [r for r in quick_results if "بيع" in r.get("decision", "")]
    waits_all = [r for r in quick_results if "انتظار" in r.get("decision", "")]

    if buys_all:
        print(f"\n🟢 فرص شراء ({len(buys_all)}):")
        print("-"*75)
        for r in sorted(buys_all, key=lambda x: int(x.get("score", 0)), reverse=True):
            ai = next((a for a in ai_results if a["symbol"] == r["symbol"]), None)
            ai_txt = f" | AI {ai['ai_dir']} ({ai['ai_conf']})" if ai else ""
            print(f"  {r['symbol']:12s} {r['name']:25s} | سعر {r['price']:>8s} | دعم {r['support']:>8s} | RSI {r['rsi']:>5s}{ai_txt}")
    if sells_all:
        print(f"\n🔴 فرص بيع ({len(sells_all)}):")
        print("-"*75)
        for r in sorted(sells_all, key=lambda x: int(x.get("score", 0))):
            ai = next((a for a in ai_results if a["symbol"] == r["symbol"]), None)
            ai_txt = f" | AI {ai['ai_dir']} ({ai['ai_conf']})" if ai else ""
            print(f"  {r['symbol']:12s} {r['name']:25s} | سعر {r['price']:>8s} | مقاومة {r['resistance']:>8s} | RSI {r['rsi']:>5s}{ai_txt}")
    print(f"\n⏳ مراقبة: {len(waits_all)} سهم")
    print("="*75)

    date_str = datetime.now().strftime('%Y%m%d_%H%M')
    with open(f"/Users/maysre/AI-Learning/StockProject/report_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(f"تقرير البورصة المصرية - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"إجمالي: {len(buys_all)} شراء | {len(sells_all)} بيع | {len(waits_all)} انتظار\n\n")
        for r in buys_all + sells_all + waits_all:
            f.write(f"{r['symbol']} | {r['decision']} | سعر {r['price']} | RSI {r['rsi']}\n")
    filename = f"report_{date_str}.txt"
    print(f"\nتم حفظ التقرير: {filename}")
    print("="*75)

if __name__ == "__main__":
    generate_report()
