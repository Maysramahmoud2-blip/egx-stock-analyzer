import streamlit as st
import sys, os, csv, io
sys.path.insert(0, os.path.dirname(__file__))
from stock_analyzer import StockAnalyzer, AIAnalyzer
from daily_report import quick_analysis, ai_analysis
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import yfinance as yf

CSV_PATH = os.path.join(os.path.dirname(__file__), 'egx_all.csv')

def rf_analysis(sym):
    try:
        df = yf.download(sym, period="1y", progress=False)
        if df.empty or len(df) < 40:
            return None
        d = df.copy()
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
    except:
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
    except:
        return None

def load_stocks():
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        return [(r['ticker'].strip() + '.CA', r['name'].strip(), r['sector'].strip()) for r in reader]

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

def deep_analyze(symbol, name):
    import yfinance as yf
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
                ai.train(ai_df)
                ai_dir, ai_conf = ai.get_signal()
            except:
                pass

        return {"symbol": symbol.replace('.CA',''), "name": name[:35], "price": price,
                "pe": pe, "pb": pb, "rsi": rsi_val, "score": score, "decision": decision,
                "ai_dir": ai_dir, "ai_conf": ai_conf}
    except:
        return None

st.set_page_config(page_title="البورصة المصرية", layout="wide")

def stock_button_row(r, cols):
    sym = r['symbol']
    if not sym.endswith('.CA'):
        sym += '.CA'
    if cols[-1].button("📊", key=f"sb_{sym.replace('.','_')}", help="اضغط لعرض التحليل"):
        st.session_state.detail_sym = sym

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

# Show detail card if a stock is selected (renders on every rerun)
if "detail_sym" in st.session_state and st.session_state.detail_sym:
    sym = st.session_state.detail_sym
    with st.container(border=True):
        col_x, _ = st.columns([1, 10])
        with col_x:
            if st.button("✕", key="close_detail", help="إغلاق"):
                st.session_state.detail_sym = ""
                st.rerun()
        render_stock_detail(sym)

def show_stock_table(stocks, columns, section_key, limit=50):
    if not stocks:
        return
    display = stocks[:limit]
    for r in display:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.2f}" if c != 'rsi' else f"{v:.0f}")
            elif c == 'pe' and v is None:
                vals.append("--")
            else:
                vals.append(str(v)[:25])
        cols = st.columns([1.5, 2, 1, 0.8, 0.8, 0.5])
        cols[0].markdown(f"**{vals[0]}**")
        for i in range(1, len(vals)):
            cols[i].write(vals[i])
        stock_button_row(r, cols)
    if len(stocks) > limit:
        with st.expander(f"عرض الكل ({len(stocks)})"):
            syms = {r['symbol']: r for r in stocks}
            sel = st.selectbox("اختر سهم", list(syms.keys()), key=f"sel_{section_key}")
            if sel and st.button("عرض", key=f"btn_{section_key}"):
                stock_dialog(sel if sel.endswith('.CA') else sel + '.CA')

st.markdown("""
    <style>
        .stApp { direction: rtl; }
        .reportview-container { font-family: 'Cairo', sans-serif; }
        h1, h2, h3 { text-align: right; }
        .stDataFrame [data-testid="stColumn"] { cursor: pointer; }
    </style>
""", unsafe_allow_html=True)

st.title("البورصة المصرية — تحليل فني + AI + استثمار")

tab1, tab2, tab3, tab4 = st.tabs(["تقرير السوق", "تحليل سهم", "جميع الأسهم", "القيمة العادلة"])

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
            col1, col2, col3 = st.columns(3)
            col1.metric("فرص شراء", len(buys))
            col2.metric("فرص بيع", len(sells))
            col3.metric("مراقبة", len(waits))

            if buys:
                st.markdown("### 🟢 فرص شراء")
                ai_results = {}
                with st.spinner("تحليل AI للفرص..."):
                    for r in buys[:5]:
                        ar = ai_analysis(r["symbol"], r["name"])
                        if ar:
                            ai_results[r["symbol"]] = ar
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
            st.success("تم التحديث")

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

            results.sort(key=lambda x: x['score'], reverse=True)
            strong = [r for r in results if r['decision'] == '🟢 قوي']
            medium = [r for r in results if r['decision'] == '🟡 متوسط']
            watch = [r for r in results if r['decision'] == '🔵 مراقبة']
            avoid = [r for r in results if r['decision'] == '🔴 تجنب']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🟢 قوي", len(strong))
            col2.metric("🟡 متوسط", len(medium))
            col3.metric("🔵 مراقبة", len(watch))
            col4.metric("🔴 تجنب", len(avoid))

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

            st.success(f"تم تحليل {len(results)} سهم")
