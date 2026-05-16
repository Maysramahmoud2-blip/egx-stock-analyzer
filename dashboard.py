import streamlit as st
import sys, os, csv, io, sqlite3
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from stock_analyzer import StockAnalyzer, AIAnalyzer
from daily_report import quick_analysis, ai_analysis
from telegram_bot import load_config, save_config, send_now, send_report, send_report_from_cache, send_alert, send_portfolio, send_bottom_alerts
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False
import plotly.graph_objects as go
import yfinance as yf

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / 'egx_all.csv'
DB_PATH = BASE_DIR / 'predictions.db'
NEWS_CACHE_PATH = BASE_DIR / '.cache_news.json'

ticker_map_egx = {'AIHC': 'AIH'}

def load_names():
    names = {}
    with open(CSV_PATH, newline='') as f:
        for r in csv.DictReader(f):
            names[r['ticker'].strip()] = r['name'].strip()
    return names

names = load_names()

CACHE_DIR = BASE_DIR
def cache_save(key, data):
    with open(CACHE_DIR / f".cache_{key}.json", 'w') as f:
        import json
        json.dump(data, f, default=str, ensure_ascii=False)
def cache_load(key):
    import json
    p = CACHE_DIR / f".cache_{key}.json"
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return None

POS_AR = ["ارتفاع", "صعود", "أرباح", "ربح", "نمو", "إيجابي", "مكاسب", "تفاؤل",
          "استثمار", "توسع", "انتعاش", "تحسن", "زيادة", "قوي", "مضاعفة", "اختراق"]
NEG_AR = ["انخفاض", "هبوط", "خسائر", "خسارة", "تراجع", "سلبي", "ديون", "مشاكل",
          "إفلاس", "تضخم", "ركود", "أزمة", "عجز", "اضطراب", "تصفية", "تحذير"]

def news_sentiment(text):
    text = text.lower()
    pos = sum(1 for w in POS_AR if w in text)
    neg = sum(1 for w in NEG_AR if w in text)
    if pos > neg: return "🟢 إيجابي", pos - neg
    if neg > pos: return "🔴 سلبي", neg - pos
    return "⚪ محايد", 0

def fetch_egx_news():
    import requests, xml.etree.ElementTree as ET, json, re
    try:
        r = requests.get("https://news.google.com/rss/search?q=البورصة+المصرية&hl=ar&gl=EG&ceid=EG:ar",
                         timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        news = []
        stocks_data = []
        with open(CSV_PATH, newline='') as f:
            for row in csv.DictReader(f):
                stocks_data.append((row['ticker'].strip(), row['name'].strip()))
        for item in items[:30]:
            title = item.findtext("title", "")[:200]
            link = item.findtext("link", "")
            pub = item.findtext("pubDate", "")[:16]
            sent, score = news_sentiment(title)
            matched = []
            for ticker, name in stocks_data:
                if name.lower() in title.lower() or ticker.lower() in title.lower():
                    matched.append(ticker)
            news.append({"title": title, "link": link, "date": pub,
                         "sentiment": sent, "sent_score": score, "stocks": matched})
        return news
    except Exception as e:
        print(f"خطأ في جلب الأخبار: {e}")
        return []

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
        df = pd.read_sql("SELECT * FROM predictions", conn)
    except:
        conn.close()
        return None
    conn.close()
    if df.empty:
        return None
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
            h.columns = h.columns.droplevel(1)
            curr = float(h['Close'].iloc[-1])
            actual = "صاعد" if curr > pred_price else "هابط"
            correct = 1 if actual == pred_dir else 0
            correct_total += correct
            total += 1
            results.append({**row.to_dict(), "current": round(curr, 2), "actual": actual, "correct": correct})
        except Exception as e:
            print(f"خطأ تقييم {sym}: {e}")
            continue
    if total == 0:
        return None
    return {"total": total, "correct": correct_total, "pct": round(correct_total / total * 100, 1),
            "details": sorted(results, key=lambda x: x['date'], reverse=True)}

def rf_analysis(sym):
    if not SKLEARN_AVAILABLE:
        return None
    try:
        df = yf.download(sym, period="1y", progress=False)
        if df.empty or len(df) < 40:
            return None
        d = df.copy()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)
        d.columns = [c.lower() for c in d.columns]
        d['ma20'] = d['close'].rolling(20).mean()
        d['ma50'] = d['close'].rolling(50).mean()
        delta = d['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d['rsi'] = 100 - (100 / (1 + (gain / loss)))
        d['ret'] = d['close'].pct_change()
        d['vol_chg'] = d['volume'].pct_change()
        d['target'] = np.where(d['close'].shift(-1) > d['close'], 1, 0)
        d = d.dropna()
        if len(d) < 30:
            return None
        feats = ['close', 'rsi', 'ma20', 'ma50', 'ret', 'vol_chg']
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(d[feats].iloc[:-1], d['target'].iloc[:-1])
        pred = model.predict(d[feats].tail(1))[0]
        proba = model.predict_proba(d[feats].tail(1))
        conf = max(proba[0]) * 100
        imp = pd.Series(model.feature_importances_, index=feats).to_dict()
        dir_text = f"{'صاعد' if pred else 'هابط'} (ثقة {conf:.0f}%)"
        return {"dir": dir_text, "raw_pred": bool(pred), "conf": conf, "importance": imp}
    except Exception as e:
        print(f"خطأ RF {sym}: {e}")
        return None

def make_excel(sym, name):
    try:
        a = StockAnalyzer(sym)
        a.fetch_data()
        a.add_indicators()
        a.generate_signals()
        df = a.data.tail(100)[['open','high','low','close','volume','rsi','macd','sma_20','sma_50']].copy()
        df.columns = ['الافتتاح','الاعلى','الادنى','الاغلاق','الكمية','RSI','MACD','SMA20','SMA50']
        df.index.name = 'التاريخ'
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
            df.to_excel(w, sheet_name=name[:31])
        return buf.getvalue()
    except Exception as e:
        print(f"خطأ Excel {sym}: {e}")
        return None

def load_stocks():
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        return [(r['ticker'].strip() + '.CA', r['name'].strip(), r['sector'].strip()) for r in reader]

def compute_indicators(df):
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
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

def deep_analyze(symbol, name):
    import yfinance as yf
    try:
        t = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period="2y")
        if hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
        hist.columns = [c.lower() for c in hist.columns]
        price = float(hist['close'].iloc[-1])
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

        delta = hist['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_val = 100 - (100 / (1 + gain / loss)).iloc[-1]
        if rsi_val < 35: score += 2
        elif rsi_val < 45: score += 1
        score = min(score, 10)

        if score >= 7: decision = "🟢 قوي"
        elif score >= 5: decision = "🟡 متوسط"
        elif score >= 3: decision = "🔵 مراقبة"
        else: decision = "🔴 تجنب"

        ai_dir, ai_conf = None, 0
        if score >= 5:
            try:
                ai_df = compute_indicators(hist)
                ai = AIAnalyzer()
                ai.train_or_load(ai_df, symbol)
                ai_dir, ai_conf = ai.get_signal()
            except Exception as e:
                print(f"خطأ AI deep {symbol}: {e}")

        return {"symbol": symbol.replace('.CA',''), "name": name[:35], "price": price,
                "pe": pe, "pb": pb, "rsi": rsi_val, "score": score, "decision": decision,
                "ai_dir": ai_dir, "ai_conf": ai_conf}
    except Exception as e:
        print(f"خطأ deep_analyze {symbol}: {e}")
        return None

def investment_scan(symbol, name):
    import yfinance as yf
    try:
        df = yf.download(symbol, period="1y", progress=False)
        if df.empty or len(df) < 50:
            return None
        df.columns = df.columns.droplevel(1)
        price = float(df['Close'].iloc[-1])
        bottom = float(df['Low'].min())
        if price <= 0 or bottom <= 0:
            return None
        pct = (price - bottom) / bottom * 100
        if pct > 50:
            return None
        close = df['Close']
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        trend = "صاعد" if sma20 > sma50 else "هابط"
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + gain / loss))).iloc[-1])
        if np.isnan(rsi):
            rsi = 50
        if pct <= 30 and trend == "صاعد":
            decision = "🟢 استثمار"
        elif pct <= 30:
            decision = "🟡 مراقبة"
        else:
            decision = "🔵 قريب من القاع"
        score = (pct <= 10 and 3 or pct <= 20 and 2 or pct <= 30 and 1 or 0) + (2 if trend == "صاعد" else 0) + (1 if rsi < 40 else 0)
        ai_dir, ai_conf = None, 0
        if score >= 3:
            try:
                ai_df = compute_indicators(df)
                ai = AIAnalyzer()
                ai.train_or_load(ai_df, symbol)
                ai_dir, ai_conf = ai.get_signal()
            except Exception as e:
                print(f"خطأ AI invest {symbol}: {e}")
        return {"symbol": symbol.replace('.CA',''), "name": name[:35],
                "price": round(price, 2), "bottom": round(bottom, 2),
                "pct": round(pct, 1), "trend": trend,
                "rsi": round(rsi, 1), "score": score, "decision": decision,
                "ai_dir": ai_dir, "ai_conf": ai_conf}
    except Exception as e:
        print(f"خطأ investment_scan {symbol}: {e}")
        return None

st.set_page_config(page_title="البورصة المصرية", layout="wide")

def render_stock_detail(sym):
    with st.spinner(f"تحليل {sym}..."):
        a = StockAnalyzer(sym)
        a.run()
        s = a.signals
    dec_icon = {"شراء (دخول) ✅": "🟢", "بيع (خروج) ❌": "🔴", "شراء (دخول) ✅✅": "🟢", "بيع (خروج) ❌❌": "🔴"}.get(s['decision'], "⏳")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f"### {dec_icon} {s['decision']}")
        st.metric("السعر", f"{s['price']:.2f}")
        st.metric("RSI", f"{s['rsi']:.1f}")
        if s.get('stop_loss'):
            st.metric("🛡️ وقف الخسارة", f"{s['stop_loss']:.2f}")
        if s.get('target'):
            st.metric("🎯 الهدف", f"{s['target']:.2f}")
        if 'ai_prediction' in s:
            st.info(f"AI: {s['ai_prediction']}")
        if s.get('patterns'):
            st.markdown("#### أنماط الشموع")
            for p in s['patterns']:
                st.markdown(f"- {p}")
        for r in s['reasons']:
            st.markdown(f"- {r}")
    with col_b:
        fig = a.plot_candlestick()
        if fig:
            st.plotly_chart(fig, width='stretch')

COL_LABELS = {"symbol": "Code", "name": "الاسم", "price": "السعر", "rsi": "RSI",
               "decision": "القرار", "ai": "AI", "pe": "P/E", "pb": "P/B",
               "score": "النتيجة", "ai_dir": "AI", "ai_conf": "الثقة",
               "bottom": "قاع السعر", "pct": "% من القاع", "trend": "الاتجاه"}
COL_RATIOS = {"symbol": 1.5, "name": 2, "price": 1, "rsi": 0.8, "decision": 1,
              "ai": 1, "pe": 0.8, "pb": 0.8, "score": 0.8, "ai_dir": 1, "ai_conf": 0.8,
              "bottom": 1, "pct": 0.8, "trend": 0.8}

def show_stock_table(stocks, columns, section_key, limit=200):
    if not stocks:
        return
    display = stocks[:limit]
    rows_html = ""
    for r in display:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.2f}" if c != 'rsi' else f"{v:.0f}"
            elif c == 'pe' and v is None:
                v = "--"
            else:
                v = str(v)[:25]
            vals.append(v)
        cells = []
        for i, c in enumerate(columns):
            v = vals[i]
            cls = ""
            if c == "decision":
                if "شراء" in v or "🟢" in v:
                    cls = 'badge-buy'
                elif "بيع" in v or "🔴" in v:
                    cls = 'badge-sell'
                elif "مراقبة" in v or "🟡" in v or "🔵" in v:
                    cls = 'badge-watch'
                cells.append(f'<td><span class="{cls}">{v}</span></td>')
            elif c == "trend":
                color = "#00c853" if v == "صاعد" else "#ff5252" if v == "هابط" else "#d1d4dc"
                cells.append(f'<td><span style="color:{color};font-weight:700">{v}</span></td>')
            elif c == "pct":
                color = "#00c853" if float(v) <= 15 else "#ffa726" if float(v) <= 30 else "#ff5252"
                cells.append(f'<td><span style="color:{color}">{v}%</span></td>')
            else:
                cells.append(f'<td>{v}</td>')
        sym = r.get('symbol', '')
        sym_full = sym if sym.endswith('.CA') else sym + '.CA'
        cells.append(f'<td style="text-align:center"><a href="?detail_sym={sym_full}" style="text-decoration:none;font-size:18px;cursor:pointer">📊</a></td>')
        rows_html += "<tr>" + "".join(cells) + "</tr>"

    hdr_cells = "".join(f'<th>{COL_LABELS.get(c, c)}</th>' for c in columns) + '<th style="width:40px"></th>'

    html = f"""
<div style="max-height:600px;overflow-y:auto;border:1px solid #2a2e39;border-radius:8px;background:#1e222d">
<table style="width:100%;border-collapse:collapse;font-size:13px;direction:rtl">
<thead style="position:sticky;top:0;background:#2a2e39;z-index:10">
<tr>{hdr_cells}</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
    if len(stocks) > limit:
        with st.expander(f"عرض الكل ({len(stocks)})"):
            syms = {r['symbol']: r for r in stocks}
            sel = st.selectbox("اختر سهم", list(syms.keys()), key=f"sel_{section_key}")
            if sel and st.button("عرض", key=f"btn_{section_key}"):
                sym = sel if sel.endswith('.CA') else sel + '.CA'
                st.session_state.detail_sym = sym

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    * { font-family: 'Cairo', sans-serif !important; }
    .stApp { direction: rtl; background: #131722; color: #d1d4dc; }
    .stApp > header { background: #1e222d !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { background: #1e222d; border-radius: 8px; padding: 4px; gap: 2px; }
    .stTabs [data-baseweb="tab"] { color: #d1d4dc !important; font-weight: 600; border-radius: 6px; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { background: #2a2e39 !important; color: #ffffff !important; }
    .stButton button { background: #2962ff !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; padding: 8px 20px !important; }
    .stButton button:hover { background: #1e4bd2 !important; }
    .stButton button[kind="secondary"] { background: transparent !important; border: 1px solid #2a2e39 !important; color: #d1d4dc !important; }
    .stMetric { background: #1e222d; border-radius: 12px; padding: 16px; border: 1px solid #2a2e39; }
    .stMetric label { color: #787b86 !important; font-size: 13px; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px !important; font-weight: 700 !important; }
    .stExpander { background: #1e222d; border-radius: 12px !important; border: 1px solid #2a2e39 !important; margin: 8px 0; }
    .stExpander summary { color: #d1d4dc !important; font-weight: 600; padding: 12px; }
    [data-testid="stExpanderToggleIcon"] { color: #787b86; }
    .stDataFrame { background: #1e222d !important; border-radius: 12px !important; }
    .stInfo { background: #1e222d !important; border: 1px solid #2962ff !important; color: #d1d4dc !important; border-radius: 8px !important; }
    .stSuccess { background: #1e222d !important; border: 1px solid #00c853 !important; color: #d1d4dc !important; border-radius: 8px !important; }
    .stSpinner { color: #2962ff !important; }
    [data-testid="stColumn"] { gap: 8px; }
    .stMarkdown { color: #d1d4dc; }
    hr { border-color: #2a2e39 !important; }
    .stSelectbox [data-baseweb="select"] { background: #1e222d !important; border-color: #2a2e39 !important; border-radius: 8px !important; }
    .stSelectbox [data-baseweb="select"] > div { color: #d1d4dc !important; }
    .stock-row { display: flex; align-items: center; padding: 6px 0; border-bottom: 1px solid #2a2e39; }
    .stock-row:last-child { border-bottom: none; }
    .price-up { color: #00c853; }
    .price-down { color: #ff5252; }
    .badge-buy { background: #00c85320; color: #00c853; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 12px; }
    .badge-sell { background: #ff525220; color: #ff5252; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 12px; }
    .badge-watch { background: #ffa72620; color: #ffa726; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 12px; }
    table td, table th { padding: 8px 12px; border-bottom: 1px solid #2a2e39; text-align: right; color: #d1d4dc; }
    table th { color: #787b86; font-weight: 700; font-size: 12px; text-transform: uppercase; }
    table tbody tr:hover { background: #2a2e3940; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #131722; }
    ::-webkit-scrollbar-thumb { background: #2a2e39; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3e49; }
</style>
""", unsafe_allow_html=True)

st.title("البورصة المصرية — تحليل فني + AI + استثمار")

# Handle query param for stock detail (from HTML table 📊 links)
qs = st.query_params
if 'detail_sym' in qs:
    st.session_state.detail_sym = qs['detail_sym']
    st.query_params.clear()
    st.rerun()

# Stock detail — full page
if "detail_sym" in st.session_state and st.session_state.detail_sym:
    sym = st.session_state.detail_sym
    st.markdown(f"## 📊 {sym}")
    if st.button("✕ رجوع"):
        st.session_state.detail_sym = ""
        st.rerun()
    render_stock_detail(sym)
    st.stop()

    with st.sidebar:
        st.markdown("## 🤖 تليجرام")
        tg_ok = load_config()
        if tg_ok:
            if st.button("📤 إرسال التقرير"):
                ok, msg = send_report()
                st.success(msg) if ok else st.error(msg)
            if st.button("🚨 صيد القاع"):
                ok, msg = send_bottom_alerts()
                st.success(msg) if ok else st.error(msg)
            port_results = st.session_state.get("port_results", [])
            if port_results and st.button("📤 إرسال المحفظة"):
                ok, msg = send_portfolio(port_results)
                st.success(msg) if ok else st.error(msg)
        else:
            st.warning("⚙️ البوت مش مconfigured")
            with st.expander("إعداد البوت"):
                token = st.text_input("Bot Token", type="password")
                chat_id = st.text_input("Chat ID")
                if st.button("حفظ"):
                    if token and chat_id:
                        save_config(token, chat_id)
                        st.success("تم الحفظ! أعد تشغيل الصفحة")
                        st.rerun()
                    else:
                        st.error("ادخل التوكن و chat ID")
            st.caption("عشان تطلع التوكن:")
            st.markdown("1. افتح تليجرام وابحث عن **@BotFather**")
            st.markdown("2. أرسل `/newbot` وسمّيه (مثلاً `EGXAlertBot`)")
            st.markdown("3. انسخ الـ **token**")
            st.markdown("4. افتح **@userinfobot** وابدأ—هيديك رقمك (Chat ID)")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["تقرير السوق", "تحليل سهم", "جميع الأسهم", "القيمة العادلة", "الاستثمار", "الأخبار", "محفظتي"])

with tab1:
    st.subheader("آخر تحديث: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    if st.button("تحديث التقرير الآن"):
        with st.spinner("جاري تحليل الأسهم..."):
            stocks = load_stocks()
            results = []
            for symbol, name, _ in stocks[:38]:
                r = quick_analysis(symbol, name)
                if "error" not in r:
                    results.append(r)
            buys = [r for r in results if "شراء" in r.get("decision", "")]
            sells = [r for r in results if "بيع" in r.get("decision", "")]
            waits = [r for r in results if "انتظار" in r.get("decision", "")]
            ai_results = {}
            with st.spinner("تحليل AI للفرص..."):
                for r in buys[:5]:
                    ar = ai_analysis(r["symbol"], r["name"])
                    if ar:
                        ai_results[r["symbol"]] = ar
                        conf_str = str(ar.get("ai_conf", "0%")).replace("%","")
                        conf_val = float(conf_str) / 100 if conf_str not in ("--", "", "0") else 0
                        p = r.get("price", 0)
                        save_prediction(r["symbol"], ar.get("ai_dir", ""),
                                        conf_val, float(p) if p not in ("", "--") else 0, r.get("score", 0))
            cache_save("report", {"buys": buys, "sells": sells, "waits": waits, "ai": ai_results})
            st.session_state.report_buys = buys
            st.session_state.report_sells = sells
            st.session_state.report_waits = waits
            st.session_state.report_ai = ai_results
            st.success("تم التحديث")

    if "report_buys" not in st.session_state:
        cached = cache_load("report")
        if cached:
            st.session_state.report_buys = cached["buys"]
            st.session_state.report_sells = cached["sells"]
            st.session_state.report_waits = cached["waits"]
            st.session_state.report_ai = cached.get("ai", {})
    if "report_buys" in st.session_state:
        buys = st.session_state.report_buys
        sells = st.session_state.report_sells
        waits = st.session_state.report_waits
        ai_results = st.session_state.report_ai
        col1, col2, col3 = st.columns(3)
        col1.metric("فرص شراء", len(buys))
        col2.metric("فرص بيع", len(sells))
        col3.metric("مراقبة", len(waits))

        acc = evaluate_accuracy()
        if acc and acc["total"] >= 3:
            st.markdown(f'<div style="background:#1e222d;border:1px solid #2a2e39;border-radius:8px;padding:10px;text-align:center;margin-bottom:12px">'
                        f'🎯 <b>دقة التوقعات:</b> {acc["pct"]}% ({acc["correct"]}/{acc["total"]})'
                        f'</div>', unsafe_allow_html=True)

        if buys:
            st.markdown("### 🟢 فرص شراء")
            for r in buys:
                ai_txt = f"{ai_results.get(r['symbol'], {}).get('ai_dir', '')} ({ai_results.get(r['symbol'], {}).get('ai_conf', '')})" if r['symbol'] in ai_results else ""
                r['ai'] = ai_txt
            show_stock_table(buys, ["symbol","name","price","rsi","decision","ai"], "tab1_buys")

        if sells:
            st.markdown("### 🔴 فرص بيع")
            show_stock_table(sells, ["symbol","name","price","rsi","decision"], "tab1_sells")

        if waits:
            with st.expander(f"⏳ أسهم مراقبة ({len(waits)})"):
                show_stock_table(waits, ["symbol","name","price","rsi","decision"], "tab1_waits")

        if load_config() and "report_buys" in st.session_state:
            if st.button("📤 إرسال التقرير للتليجرام"):
                ok, res = send_report_from_cache(
                    st.session_state.get('report_buys', []),
                    st.session_state.get('report_sells', []),
                    st.session_state.get('report_waits', []),
                    st.session_state.get('report_ai', {})
                )
                st.success(res) if ok else st.error(res)

with tab2:
    col1, col2 = st.columns([1, 3])
    symbol = col1.text_input("رمز السهم", "COMI.CA").strip()
    chart_type = col2.radio("نوع الرسم", ["شموع تفاعلي", "خطوط كلاسيكي"], horizontal=True)
    stocks_list = load_stocks()
    egx_options = [s[0] for s in stocks_list]
    selected = st.selectbox("أو اختر من القائمة", egx_options, index=egx_options.index("COMI.CA") if "COMI.CA" in egx_options else 0)
    if st.button("تحليل"):
        sym = symbol if symbol else selected
        with st.spinner(f"تحليل {sym}..."):
            a = StockAnalyzer(sym)
            a.run()
            s = a.signals
            dec_icon = {"شراء (دخول) ✅": "🟢", "بيع (خروج) ❌": "🔴", "شراء (دخول) ✅✅": "🟢", "بيع (خروج) ❌❌": "🔴"}.get(s['decision'], "⏳")
            st.markdown(f"## {dec_icon} {s['decision']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("السعر", f"{s['price']:.2f}")
            m2.metric("RSI", f"{s['rsi']:.1f}")
            m3.metric("دعم", f"{s['support']:.2f}" if s['support'] else "--")
            m4.metric("مقاومة", f"{s['resistance']:.2f}" if s['resistance'] else "--")

            sl_txt = f"{s['stop_loss']:.2f}" if s.get('stop_loss') else "--"
            tg_txt = f"{s['target']:.2f}" if s.get('target') else "--"
            m5, m6 = st.columns(2)
            m5.metric("🛡️ وقف الخسارة", sl_txt)
            m6.metric("🎯 الهدف", tg_txt)

            col_ai1, col_ai2 = st.columns(2)
            with col_ai1:
                if 'ai_prediction' in s:
                    st.info(f"🧠 LSTM: {s['ai_prediction']}")
            with col_ai2:
                rf = rf_analysis(sym)
                if rf:
                    st.info(f"🌲 RandomForest: {rf['dir']}")
                else:
                    st.info("🌲 RandomForest: بيانات غير كافية")

            if s.get('patterns'):
                st.markdown("### أنماط الشموع")
                for p in s['patterns']:
                    st.markdown(f"- {p}")
            for r in s['reasons']:
                st.markdown(f"- {r}")

            col_chart, col_excel = st.columns([5, 1])
            with col_chart:
                if chart_type == "شموع تفاعلي":
                    fig = a.plot_candlestick()
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                else:
                    a.plot_chart()
            with col_excel:
                excel_data = make_excel(sym, sym)
                if excel_data:
                    st.download_button("📥 Excel", data=excel_data,
                        file_name=f"{sym}_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab3:
    st.subheader("جميع الأسهم — إشارات فنية")
    if st.button("مسح الجميع"):
        with st.spinner("جاري تحليل جميع الأسهم..."):
            stocks = load_stocks()
            df_list = []
            for symbol, name, _ in stocks:
                r = quick_analysis(symbol, name)
                if "error" not in r:
                    df_list.append(r)
            df = pd.DataFrame(df_list)
            col_filter = st.selectbox("تصفية حسب", ["الكل", "شراء", "بيع", "انتظار"])
            if col_filter != "الكل":
                keywords = {"شراء": "شراء", "بيع": "بيع", "انتظار": "انتظار"}
                df = df[df["decision"].str.contains(keywords[col_filter], na=False)]
            st.dataframe(df[["symbol","name","price","decision","rsi","support","resistance"]].sort_values("symbol"), use_container_width=True, height=600)

with tab4:
    st.subheader("بحث القيمة العادلة — استثمار طويل المدى")
    st.markdown("تحليل أساسي + AI للأسهم القريبة من قاعها")

    if st.button("بدء البحث"):
        with st.spinner("جاري تحليل القيمة العادلة... قد يستغرق 10-15 دقيقة"):
            stocks = load_stocks()
            results = []
            progress = st.progress(0)
            for i, (symbol, name, _) in enumerate(stocks):
                r = deep_analyze(symbol, name)
                if r:
                    results.append(r)
                progress.progress((i + 1) / len(stocks))
            cache_save("deepvalue", results)
            st.session_state.dv_results = results
    if "dv_results" not in st.session_state:
        cached = cache_load("deepvalue")
        if cached:
            st.session_state.dv_results = cached
    if "dv_results" in st.session_state:
        results = st.session_state.dv_results
        results.sort(key=lambda x: x['score'], reverse=True)
        strong = [r for r in results if r['decision'] == '🟢 قوي']
        medium = [r for r in results if r['decision'] == '🟡 متوسط']
        watch = [r for r in results if r['decision'] == '🔵 مراقبة']
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 قوي", len(strong))
        col2.metric("🟡 متوسط", len(medium))
        col3.metric("🔵 مراقبة", len(watch))
        if strong:
            st.markdown("### 🟢 فرص استثمار قوية")
            for r in strong:
                r['ai'] = f"{r['ai_dir']} (ثقة {r['ai_conf']:.0%})" if r['ai_dir'] else "—"
            show_stock_table(strong, ["symbol","name","price","pe","rsi","score","ai"], "tab4_strong")
        if medium:
            st.markdown("### 🟡 فرص متوسطة")
            for r in medium:
                r['ai'] = f"{r['ai_dir']} (ثقة {r['ai_conf']:.0%})" if r['ai_dir'] else "—"
            show_stock_table(medium, ["symbol","name","price","pe","rsi","score","ai"], "tab4_medium")
        if watch:
            st.markdown("### 🔵 مراقبة")
            show_stock_table(watch, ["symbol","name","price","rsi","score"], "tab4_watch")

with tab5:
    st.subheader("الاستثمار — قاع سعري + اتجاه صاعد")
    st.markdown("يفلتر الأسهم اللي في قاعها التاريخي وبدأ الاتجاه يقلب لصاعد")
    if st.button("بدء مسح الاستثمار"):
        with st.spinner("جاري مسح الأسهم..."):
            stocks = load_stocks()
            results = []
            progress = st.progress(0)
            for i, (symbol, name, _) in enumerate(stocks):
                r = investment_scan(symbol, name)
                if r:
                    results.append(r)
                progress.progress((i + 1) / len(stocks))
            cache_save("invest", results)
            st.session_state.invest_results = results
    if "invest_results" not in st.session_state:
        cached = cache_load("invest")
        if cached:
            st.session_state.invest_results = cached
    if "invest_results" in st.session_state:
        results = st.session_state.invest_results
        seen = set()
        deduped = []
        for r in results:
            k = r.get('symbol','')
            if k not in seen:
                seen.add(k)
                deduped.append(r)
        results = deduped
        invest = [r for r in results if r['decision'] == '🟢 استثمار']
        watch = [r for r in results if r['decision'] == '🟡 مراقبة']
        near = [r for r in results if r['decision'] == '🔵 قريب من القاع']
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 استثمار", len(invest))
        col2.metric("🟡 مراقبة", len(watch))
        col3.metric("🔵 قريب من القاع", len(near))
        if invest:
            st.markdown("### 🟢 فرص استثمار")
            for r in invest:
                ac = r.get('ai_conf', 0)
                r['ai'] = f"{r.get('ai_dir','')} ({ac:.0%})" if r.get('ai_dir') and ac else "—"
            show_stock_table(invest, ["symbol","name","price","bottom","pct","trend","rsi","ai"], "tab5_invest")
        if watch:
            with st.expander(f"🟡 مراقبة - قاع ولكن بدون اتجاه ({len(watch)})"):
                show_stock_table(watch, ["symbol","name","price","bottom","pct","trend","rsi","decision"], "tab5_watch")
        if near:
            with st.expander(f"🔵 قريب من القاع - 30-50% ({len(near)})"):
                show_stock_table(near, ["symbol","name","price","bottom","pct","trend","rsi","decision"], "tab5_near")

with tab6:
    st.subheader("أخبار البورصة المصرية")
    st.caption("آخر الأخبار المؤثرة على السوق")
    news = st.session_state.get("news_cache")
    if news is None:
        news = fetch_egx_news()
        if news:
            st.session_state.news_cache = news
            cache_save("news", news)
    if not news:
        if st.button("تحميل الأخبار"):
            with st.spinner("جاري تحميل الأخبار..."):
                news = fetch_egx_news()
                if news:
                    st.session_state.news_cache = news
                    cache_save("news", news)
                    st.rerun()
                else:
                    st.warning("لا توجد أخبار متاحة حالياً")
    if news:
        for n in news:
            sent = n["sentiment"]
            stocks = n["stocks"]
            matched_stocks = ""
            if stocks:
                tags = " ".join([f'<span style="background:#2962ff;color:#fff;padding:1px 6px;border-radius:4px;font-size:12px;margin:0 2px">{s}</span>' for s in stocks])
                matched_stocks = f'<div style="margin:6px 0">{tags}</div>'
            bg = "#1e222d"
            border = "#ff525240" if "سلبي" in sent else "#00c85340" if "إيجابي" in sent else "#2a2e39"
            st.markdown(f'<div style="background:{bg};border:1px solid {border};border-radius:8px;padding:12px;margin:6px 0">'
                        f'<div style="font-size:13px;color:#787b86">{n["date"]} | {sent}</div>'
                        f'<div style="font-size:15px;color:#d1d4dc;margin:4px 0">{n["title"]}</div>'
                        f'{matched_stocks}'
                        f'<a href="{n["link"]}" target="_blank" style="color:#2962ff;font-size:12px">اقرأ المزيد →</a>'
                        f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔗 ربط الأخبار بفرص الاستثمار")
    st.caption("عرض الأخبار المرتبطة بأسهم الاستثمار المتوقع صعودها")
    invest_results = st.session_state.get("invest_results") or cache_load("invest")
    news = st.session_state.get("news_cache") or cache_load("news")
    if invest_results and news:
        invest_stocks = [r for r in invest_results if r.get('ai_dir') == 'صاعد' and r.get('decision') in ('🟢 استثمار', '🔵 قريب من القاع')]
        if invest_stocks:
            for inv in invest_stocks:
                sym = inv['symbol']
                related = [n for n in news if sym in n.get('stocks', [])]
                if related:
                    sentiment_match = any("إيجابي" in n.get('sentiment','') for n in related)
                    color = "#00c853" if sentiment_match else "#ffa726"
                    st.markdown(f'<div style="background:#1e222d;border:1px solid {color}40;border-radius:8px;padding:10px;margin:6px 0">'
                                f'<span style="font-weight:700;color:{color}">{sym}</span>'
                                f' — {inv["name"]} | سعر {inv["price"]} | {inv["decision"]}'
                                f' | AI {inv.get("ai_dir","")} ({inv.get("ai_conf",0):.0%})'
                                f' | أخبار مرتبطة: {len(related)}'
                                f'</div>', unsafe_allow_html=True)
        else:
            st.info("لا توجد أخبار مرتبطة بفرص الاستثمار الحالية")
    else:
        st.info("شغّل مسح الاستثمار أولاً من تبويب الاستثمار")

with tab7:
    st.subheader("تقييم المحفظة الشخصية")
    st.markdown("أضف أسهم محفظتك لتحليل شامل (فني + أساسي + AI)")

    default_symbols = "BIGP, AIH, SPMD, BWA"
    portfolio_input = st.text_input("رموز الأسهم (مفصولة بفاصلة)", default_symbols).strip()
    symbols = [s.strip().upper() for s in portfolio_input.split(",") if s.strip()]

    portfolio = []
    for sym in symbols:
        if sym in names:
            portfolio.append((names[sym], sym))
        elif sym in ticker_map_egx:
            portfolio.append((names.get(ticker_map_egx[sym], sym), ticker_map_egx[sym]))
        else:
            portfolio.append((sym, sym))

    def port_analysis(name, sym):
        is_egx = sym in names.values() or sym in ticker_map_egx.values()
        actual = ticker_map_egx.get(sym, sym)
        full_sym = actual + '.CA' if is_egx else actual
        try:
            t = yf.Ticker(full_sym)
            hist = t.history(period="1y")
            if hist.empty:
                return None
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)
            col = 'close' if 'close' in hist.columns else 'Close'
            price = float(hist[col].iloc[-1])
            perf_1m = ((price - float(hist[col].iloc[-22])) / float(hist[col].iloc[-22]) * 100) if len(hist) >= 22 else None
            d = hist.copy()
            d.columns = [c.lower() for c in d.columns]
            d['sma_20'] = d['close'].rolling(20).mean()
            d['sma_50'] = d['close'].rolling(50).mean()
            delta = d['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = float((100 - (100 / (1 + gain / loss))).iloc[-1])
            sma20 = float(d['sma_20'].iloc[-1])
            sma50 = float(d['sma_50'].iloc[-1])
            trend = "صاعد" if sma20 > sma50 else "هابط"
            adx_raw = d['close']
            tr = np.maximum(d['high'] - d['low'],
                 np.maximum(abs(d['high'] - d['close'].shift(1)), abs(d['low'] - d['close'].shift(1))))
            atr = tr.rolling(14).mean()
            up = d['high'].diff()
            down = -d['low'].diff()
            plus_dm = np.where((up > down) & (up > 0), up, 0)
            minus_dm = np.where((down > up) & (down > 0), down, 0)
            plus_di = 100 * pd.Series(plus_dm, index=d.index).rolling(14).mean() / atr
            minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(14).mean() / atr
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = float(dx.iloc[-1]) if not dx.empty else 0

            info = t.info
            pe = info.get('trailingPE')
            pb = info.get('priceToBook')
            low_52w = info.get('fiftyTwoWeekLow', price)
            from_low = (price - low_52w) / low_52w * 100

            ai_dir, ai_conf = None, 0
            try:
                ai_df = d.copy()
                ai = AIAnalyzer()
                ai.train_or_load(ai_df, actual)
                ai_dir, ai_conf = ai.get_signal()
            except:
                pass

            return {"name": name, "symbol": sym, "price": price,
                    "perf_1m": perf_1m, "rsi": rsi, "trend": trend,
                    "adx": adx, "pe": pe, "pb": pb, "from_low": from_low,
                    "ai_dir": ai_dir, "ai_conf": ai_conf}
        except Exception as e:
            return {"name": name, "symbol": sym, "error": str(e)[:80]}

    if st.button("تقييم المحفظة"):
        with st.spinner("جاري تحليل المحفظة..."):
            results = []
            for name, sym in portfolio:
                r = port_analysis(name, sym)
                if r:
                    results.append(r)
            st.session_state.port_results = results

    if "port_results" in st.session_state:
        results = st.session_state.port_results
        valid = [r for r in results if 'error' not in r]
        errors = [r for r in results if 'error' in r]

        if errors:
            st.warning(f"أسهم بها مشاكل: {', '.join(r['symbol'] for r in errors)}")

        if valid:
            avg_perf = sum(r.get('perf_1m') or 0 for r in valid) / len(valid)
            up = sum(1 for r in valid if (r.get('perf_1m') or 0) > 0)
            down = sum(1 for r in valid if (r.get('perf_1m') or 0) < 0)

            m1, m2, m3 = st.columns(3)
            m1.metric("متوسط أداء الشهر", f"{avg_perf:+.1f}%")
            m2.metric("صاعد", up)
            m3.metric("هابط", down)

            st.markdown("### تفاصيل المحفظة")
            for r in valid:
                color = "#00c853" if r.get('trend') == "صاعد" else "#ff5252"
                ai_txt = f"{r.get('ai_dir','')} ({r.get('ai_conf',0):.0%})" if r.get('ai_dir') else "—"
                perf_txt = f"{r['perf_1m']:+.1f}%" if r.get('perf_1m') is not None else "--"
                st.markdown(f'<div style="background:#1e222d;border:1px solid #2a2e39;border-radius:8px;padding:12px;margin:6px 0">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center">'
                            f'<span style="font-weight:700;font-size:16px">{r["symbol"]}</span>'
                            f'<span style="color:{color}">{r["trend"]}</span>'
                            f'</div>'
                            f'<div style="display:flex;gap:16px;margin-top:6px;color:#787b86;font-size:13px">'
                            f'<span>سعر: <b style="color:#d1d4dc">{r["price"]}</b></span>'
                            f'<span>شهري: <b style="color:#d1d4dc">{perf_txt}</b></span>'
                            f'<span>RSI: <b style="color:#d1d4dc">{r["rsi"]:.0f}</b></span>'
                            f'<span>ADX: <b style="color:#d1d4dc">{r["adx"]:.0f}</b></span>'
                            f'<span>P/E: <b style="color:#d1d4dc">{r["pe"] or "--"}</b></span>'
                            f'<span>من القاع: <b style="color:#d1d4dc">{r["from_low"]:+.0f}%</b></span>'
                            f'<span>AI: <b style="color:#d1d4dc">{ai_txt}</b></span>'
                            f'</div></div>', unsafe_allow_html=True)

        if load_config() and "port_results" in st.session_state and st.session_state.port_results:
            valid_port = [r for r in st.session_state.port_results if 'error' not in r]
            if st.button("📤 إرسال المحفظة للتليجرام"):
                ok, res = send_portfolio(valid_port)
                st.success(res) if ok else st.error(res)
