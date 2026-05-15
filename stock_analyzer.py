import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
plt.rcParams['font.family'] = 'Arial'

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=5, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class AIAnalyzer:
    def __init__(self, device=None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.pred = None
        self.confidence = 0

    def _prepare_data(self, df, seq_len=20):
        prices = df[['close', 'volume', 'sma_20', 'rsi', 'macd']].dropna().values
        prices = (prices - prices.mean(axis=0)) / (prices.std(axis=0) + 1e-8)
        xs, ys = [], []
        for i in range(seq_len, len(prices) - 5):
            xs.append(prices[i-seq_len:i])
            ys.append(1 if prices[i+5, 0] > prices[i, 0] else 0)
        t = int(len(xs) * 0.8)
        return (torch.tensor(np.array(xs[:t]), dtype=torch.float32),
                torch.tensor(np.array(ys[:t]), dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(xs[t:]), dtype=torch.float32),
                torch.tensor(np.array(ys[t:]), dtype=torch.float32).unsqueeze(1))

    def train(self, df, seq_len=20, epochs=30):
        x_train, y_train, x_test, y_test = self._prepare_data(df, seq_len)
        if len(x_train) < 50:
            return
        self.model = LSTMPredictor().to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(epochs):
            for b in range(0, len(x_train), 32):
                bx, by = x_train[b:b+32].to(self.device), y_train[b:b+32].to(self.device)
                opt.zero_grad()
                loss_fn(self.model(bx), by).backward()
                opt.step()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_test.to(self.device))
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds.cpu() == y_test).float().mean().item()
            latest_input = x_train[-1:].to(self.device)
            self.pred = torch.sigmoid(self.model(latest_input)).item()
            self.confidence = abs(self.pred - 0.5) * 2

    def get_signal(self):
        if self.pred is None:
            return None, 0
        direction = "صاعد" if self.pred > 0.5 else "هابط"
        return direction, self.confidence

class StockAnalyzer:
    def __init__(self, symbol, period="2y"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.signals = None

    def fetch_data(self):
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        if self.data.empty:
            raise Exception(f"مافيش بيانات للسهم {self.symbol}")
        self.data.columns = [c.lower() for c in self.data.columns]

    def add_indicators(self):
        df = self.data.copy()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        self.data = df

    def find_support_resistance(self, lookback=60):
        df = self.data.tail(lookback).copy()
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        resistance = []
        support = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support.append(lows[i])
        r_level = np.mean(resistance[-3:]) if len(resistance) >= 3 else max(resistance) if resistance else None
        s_level = np.mean(support[-3:]) if len(support) >= 3 else min(support) if support else None
        return s_level, r_level

    def generate_signals(self):
        df = self.data.tail(200).copy()
        score = 0
        reasons = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        if latest['sma_20'] > latest['sma_50']:
            score += 1
            reasons.append("SMA20 فوق SMA50 (اتجاه صاعد)")
        elif latest['sma_20'] < latest['sma_50']:
            score -= 1
            reasons.append("SMA20 تحت SMA50 (اتجاه هابط)")

        if prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            score += 1
            reasons.append("MACD تقاطع شراء")
        elif prev['macd'] >= prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
            score -= 1
            reasons.append("MACD تقاطع بيع")

        if latest['rsi'] < 30:
            score += 2
            reasons.append(f"RSI {latest['rsi']:.1f} — منطقة ذروة بيع (شراء)")
        elif latest['rsi'] > 70:
            score -= 2
            reasons.append(f"RSI {latest['rsi']:.1f} — منطقة ذروة شراء (بيع)")
        elif 40 <= latest['rsi'] <= 60:
            reasons.append(f"RSI {latest['rsi']:.1f} — منطقة محايدة")

        s_level, r_level = self.find_support_resistance()
        if s_level and latest['close'] <= s_level * 1.03:
            score += 1
            reasons.append(f"قرب مستوى دعم ({s_level:.2f})")
        if r_level and latest['close'] >= r_level * 0.97:
            score -= 1
            reasons.append(f"قرب مستوى مقاومة ({r_level:.2f})")

        if score >= 2:
            decision = "شراء (دخول) ✅"
        elif score <= -2:
            decision = "بيع (خروج) ❌"
        else:
            decision = "انتظار ⏳"

        patterns = self.detect_patterns()
        if patterns:
            reasons.extend(patterns)

        self.signals = {
            'decision': decision,
            'score': score,
            'price': latest['close'],
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'support': s_level,
            'resistance': r_level,
            'reasons': reasons,
            'date': latest.name,
            'patterns': patterns,
        }

    def show_analysis(self):
        print(f"\n{'='*50}")
        print(f"تحليل سهم: {self.symbol}")
        print(f"تاريخ التحليل: {self.signals['date']}")
        print(f"السعر الحالي: {self.signals['price']:.2f}")
        print(f"{'='*50}")
        print(f"القرار: {self.signals['decision']}")
        if 'ai_prediction' in self.signals:
            print(f"AI: {self.signals['ai_prediction']}")
        print(f"RSI: {self.signals['rsi']:.2f}")
        print(f"MACD: {self.signals['macd']:.5f}")
        if self.signals['support']:
            print(f"مستوى دعم: {self.signals['support']:.2f}")
        if self.signals['resistance']:
            print(f"مستوى مقاومة: {self.signals['resistance']:.2f}")
        print(f"{'='*50}")
        print("الأسباب:")
        for r in self.signals['reasons']:
            print(f"  - {r}")
        print(f"{'='*50}\n")

    def plot_chart(self, save=False):
        df = self.data.tail(100)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        ax1.plot(df.index, df['close'], label='سعر الإغلاق', color='#2196F3', linewidth=2)
        ax1.plot(df.index, df['sma_20'], label='SMA 20', color='#FF9800', linestyle='--')
        ax1.plot(df.index, df['sma_50'], label='SMA 50', color='#F44336', linestyle='--')
        ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
        s_level, r_level = self.find_support_resistance()
        if s_level:
            ax1.axhline(y=s_level, color='#4CAF50', linestyle=':', linewidth=1.5, label=f'دعم {s_level:.2f}')
        if r_level:
            ax1.axhline(y=r_level, color='#F44336', linestyle=':', linewidth=1.5, label=f'مقاومة {r_level:.2f}')
        ax1.set_title(f'{self.symbol} — تحليل السهم', fontsize=14)
        ax1.set_ylabel('السعر')
        ax1.legend()
        ax1.grid(alpha=0.2)

        ax2.plot(df.index, df['rsi'], color='#9C27B0', linewidth=2)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(alpha=0.2)

        ax3.plot(df.index, df['macd'], label='MACD', color='#2196F3', linewidth=2)
        ax3.plot(df.index, df['macd_signal'], label='Signal', color='#FF9800', linestyle='--')
        ax3.bar(df.index, df['macd'] - df['macd_signal'], color=np.where(df['macd'] > df['macd_signal'], '#4CAF50', '#F44336'), alpha=0.3)
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(alpha=0.2)

        plt.tight_layout()
        if save:
            plt.savefig(f'/Users/maysre/AI-Learning/StockProject/{self.symbol}_analysis.png', dpi=150)
            print("تم حفظ الشارت")
        plt.show()

    def detect_patterns(self):
        df = self.data.tail(60).copy()
        patterns = []
        for i in range(1, len(df)):
            p, c = df.iloc[i-1], df.iloc[i]
            body_p = abs(p['close'] - p['open'])
            body_c = abs(c['close'] - c['open'])
            range_p = p['high'] - p['low']
            range_c = c['high'] - c['low']
            if range_p == 0 or range_c == 0:
                continue

            # doji
            if body_c / range_c < 0.1:
                if c['close'] < p['close'] - body_p * 0.3:
                    patterns.append(f"🔴 دوجي عند القمة — احتمال انعكاس هابط")
                elif c['close'] > p['close'] + body_p * 0.3:
                    patterns.append(f"🟢 دوجي عند القاع — احتمال انعكاس صاعد")
                else:
                    patterns.append(f"⚪ دوجي — حيرة في السوق")
            # hammer
            upper = c['high'] - max(c['close'], c['open'])
            lower = min(c['close'], c['open']) - c['low']
            if lower > body_c * 2 and upper < body_c * 0.3 and body_c / range_c > 0.1:
                patterns.append(f"🟢 شمعة مطرقة (Hammer) — احتمال انعكاس صاعد")
            # shooting star
            if upper > body_c * 2 and lower < body_c * 0.3 and body_c / range_c > 0.1:
                patterns.append(f"🔴 شمعة نجم ثاقب (Shooting Star) — احتمال انعكاس هابط")
            # engulfing
            if p['close'] > p['open'] and c['close'] < c['open'] and c['open'] > p['close'] and c['close'] < p['open']:
                patterns.append(f"🔴 شمعة ابتلاع هابطة (Bearish Engulfing)")
            if p['close'] < p['open'] and c['close'] > c['open'] and c['open'] < p['close'] and c['close'] > p['open']:
                patterns.append(f"🟢 شمعة ابتلاع صاعدة (Bullish Engulfing)")
            # three white soldiers / three black crows (simplified)
            if i >= 3:
                c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
                if (c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open'] and
                    c2['close'] > c1['close'] and c3['close'] > c2['close']):
                    patterns.append(f"🟢 ثلاثة جنود بيض (Three White Soldiers) — اتجاه صاعد قوي")
                if (c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open'] and
                    c2['close'] < c1['close'] and c3['close'] < c2['close']):
                    patterns.append(f"🔴 ثلاثة غربان سود (Three Black Crows) — اتجاه هابط قوي")
        return patterns

    def plot_candlestick(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        df = self.data.tail(120).copy()
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.6, 0.2, 0.2])

        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                      low=df['low'], close=df['close'], name='السعر',
                      increasing_line_color='#4CAF50', decreasing_line_color='#F44336'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], line=dict(color='#FF9800', width=1), name='SMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], line=dict(color='#9C27B0', width=1), name='SMA 50'), row=1, col=1)
        s, r = self.find_support_resistance()
        if s:
            fig.add_hline(y=s, line_color='green', line_dash='dot', annotation_text=f'دعم {s:.2f}', row=1, col=1)
        if r:
            fig.add_hline(y=r, line_color='red', line_dash='dot', annotation_text=f'مقاومة {r:.2f}', row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#9C27B0'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_color='red', line_dash='dash', row=2, col=1)
        fig.add_hline(y=30, line_color='green', line_dash='dash', row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], line=dict(color='#2196F3'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], line=dict(color='#FF9800'), name='Signal'), row=3, col=1)

        fig.update_layout(title=f'{self.symbol} — رسم بياني تفاعلي', xaxis_rangeslider_visible=False,
                          height=700, template='plotly_white')
        return fig

    def run(self):
        print(f"جلب بيانات {self.symbol}...")
        self.fetch_data()
        print("تحليل المؤشرات...")
        self.add_indicators()
        self.generate_signals()
        print("تدريب نموذج AI...")
        ai = AIAnalyzer()
        ai.train(self.data)
        ai_dir, ai_conf = ai.get_signal()
        if ai_dir:
            self.signals['ai_prediction'] = f"{ai_dir} (ثقة {ai_conf:.0%})"
            if ai_dir == "صاعد" and self.signals['score'] >= 1:
                self.signals['decision'] = "شراء (دخول) ✅✅"
                self.signals['reasons'].append(f"AI يتوقع {ai_dir} بثقة {ai_conf:.0%}")
            elif ai_dir == "هابط" and self.signals['score'] <= -1:
                self.signals['decision'] = "بيع (خروج) ❌❌"
                self.signals['reasons'].append(f"AI يتوقع {ai_dir} بثقة {ai_conf:.0%}")
            else:
                self.signals['reasons'].append(f"AI يتوقع {ai_dir} بثقة {ai_conf:.0%}")
        self.show_analysis()

if __name__ == "__main__":
    print("أسهم البورصة المصرية (EGX)")
    print("مثال: COMI.CA (البنك التجاري الدولي)")
    symbol = input("اكتب رمز السهم: ").strip()
    if not symbol:
        symbol = "COMI.CA"
    analyzer = StockAnalyzer(symbol)
    analyzer.run()
    show = input("\nعايز تشوف الشارت؟ (y/n): ")
    if show.lower() == 'y':
        analyzer.plot_chart()
