import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import torch
import torch.nn as nn
plt.rcParams['font.family'] = 'Arial'

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=7, hidden=64, layers=2):
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
        self.mean = None
        self.std = None

    def _prepare_data(self, df, seq_len=20):
        d = df.copy()
        d['returns'] = d['close'].pct_change()
        d['volatility'] = d['close'].rolling(window=10).std()
        raw = d[['close', 'volume', 'sma_20', 'rsi', 'macd', 'returns', 'volatility']].dropna().values
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0) + 1e-8
        prices = (raw - self.mean) / self.std
        xs, ys = [], []
        for i in range(seq_len, len(prices) - 5):
            xs.append(prices[i-seq_len:i])
            ys.append(1 if prices[i+5, 0] > prices[i, 0] else 0)
        t = int(len(xs) * 0.8)
        return (torch.tensor(np.array(xs[:t]), dtype=torch.float32),
                torch.tensor(np.array(ys[:t]), dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(xs[t:]), dtype=torch.float32),
                torch.tensor(np.array(ys[t:]), dtype=torch.float32).unsqueeze(1))

    def save(self, symbol):
        if self.model is None or self.mean is None:
            return
        torch.save({
            'model_state': self.model.state_dict(),
            'mean': self.mean,
            'std': self.std,
        }, MODELS_DIR / f'{symbol.replace(".CA","")}.pth')

    def load(self, symbol):
        path = MODELS_DIR / f'{symbol.replace(".CA","")}.pth'
        if not path.exists():
            return False
        chk = torch.load(path, map_location=self.device, weights_only=False)
        self.model = LSTMPredictor().to(self.device)
        self.model.load_state_dict(chk['model_state'])
        self.model.eval()
        self.mean = chk['mean']
        self.std = chk['std']
        return True

    def train(self, df, seq_len=20, epochs=30):
        try:
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
                latest_input = x_train[-1:].to(self.device)
                self.pred = torch.sigmoid(self.model(latest_input)).item()
                self.confidence = abs(self.pred - 0.5) * 2
        except Exception as e:
            print(f"⚠️ فشل تدريب الـ AI للسهم: {e}")
            self.pred = None

    def train_or_load(self, df, symbol, seq_len=20, epochs=30):
        if self.load(symbol):
            d = df.copy()
            d['returns'] = d['close'].pct_change()
            d['volatility'] = d['close'].rolling(window=10).std()
            cols = ['close', 'volume', 'sma_20', 'rsi', 'macd', 'returns', 'volatility']
            raw = d[cols].dropna().values
            if len(raw) >= seq_len + 5 and raw.shape[1] == len(self.mean):
                data = (raw[-seq_len:] - self.mean) / self.std
                inp = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    self.pred = torch.sigmoid(self.model(inp)).item()
                    self.confidence = abs(self.pred - 0.5) * 2
                return
        self.train(df, seq_len, epochs)
        self.save(symbol)

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
        low14 = df['low'].rolling(14).min()
        high14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['tr'] = np.maximum(df['high'] - df['low'],
                     np.maximum(abs(df['high'] - df['close'].shift(1)),
                                abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        up = df['high'].diff()
        down = -df['low'].diff()
        df['plus_dm'] = np.where((up > down) & (up > 0), up, 0)
        df['minus_dm'] = np.where((down > up) & (down > 0), down, 0)
        df['plus_di'] = 100 * df['plus_dm'].rolling(14).mean() / df['atr']
        df['minus_di'] = 100 * df['minus_dm'].rolling(14).mean() / df['atr']
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()
        df['ichimoku_conv'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['ichimoku_base'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        df['ichimoku_span_a'] = ((df['ichimoku_conv'] + df['ichimoku_base']) / 2).shift(26)
        span_b_high = df['high'].rolling(52).max()
        span_b_low = df['low'].rolling(52).min()
        df['ichimoku_span_b'] = ((span_b_high + span_b_low) / 2).shift(26)
        self.data = df

    def find_support_resistance(self, lookback=60):
        df = self.data.tail(lookback).copy()
        highs = df['high'].values
        lows = df['low'].values
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

    def generate_signals(self, ai_dir=None, ai_conf=0):
        df = self.data.tail(200).copy()
        score = 0
        reasons = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        historical_low = self.data['low'].min()
        proximity_to_low = ((latest['close'] - historical_low) / historical_low) * 100
        if proximity_to_low < 10:
            score += 25
            reasons.append(f"💎 سهم لقطة: قريب جداً من قاعه التاريخي ({historical_low:.2f})")
        elif proximity_to_low < 25:
            score += 15
            reasons.append("⚖️ السعر في مناطق دعم تاريخية جيدة")

        if ai_dir == "صاعد":
            ai_weight = int(ai_conf * 35)
            score += ai_weight
            reasons.append(f"🚀 ذكاء اصطناعي صاعد: يدعم الصعود بقوة +{ai_weight}")
        elif ai_dir == "هابط":
            ai_weight = int(ai_conf * 35)
            score -= ai_weight
            reasons.append(f"⚠️ ذكاء اصطناعي هابط: يحذر من الهبوط بقوة -{ai_weight}")

        if latest['close'] > latest['sma_20']:
            score += 15
            reasons.append("✅ السعر فوق المتوسط المتحرك SMA 20")

        if prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            score += 10
            reasons.append("📈 MACD تقاطع شراء إيجابي")

        if 40 <= latest['rsi'] <= 65:
            score += 15
            reasons.append(f"📊 RSI مثالي وفي منطقة زخم آمنة ({latest['rsi']:.1f})")
        elif latest['rsi'] < 30:
            score += 10
            reasons.append(f"⚡ RSI ذروة بيع - ارتداد محتمل ({latest['rsi']:.1f})")
        elif latest['rsi'] > 70:
            score -= 20
            reasons.append(f"🚨 RSI تضخم صعودي حاد ({latest['rsi']:.1f})")

        if latest['stoch_k'] < 20:
            score += 5
            reasons.append("Stochastic في منطقة ذروة البيع")

        if latest['adx'] > 25 and latest['plus_di'] > latest['minus_di']:
            score += 10
            reasons.append("ADX يؤكد قوة الاتجاه الصاعد الحالي")

        if score >= 75:
            decision = "🔥 شراء قوي (High Confidence)"
        elif score >= 50:
            decision = "⚡ شراء مضاربي (Medium Confidence)"
        elif score <= -20:
            decision = "بيع (خروج) ❌"
        else:
            decision = "انتظار ⏳"

        patterns = self.detect_patterns()
        if patterns:
            reasons.extend(patterns)

        atr_val = latest['atr']
        stop_loss = latest['close'] - (atr_val * 2)
        target = latest['close'] + (atr_val * 2)

        reasons.append(f"🛡️ حماية: وقف الخسارة الصارم عند {stop_loss:.2f}")
        reasons.append(f"🎯 مستهدف: الهدف الأول الفني عند {target:.2f}")

        s_level, r_level = self.find_support_resistance()

        self.signals = {
            'decision': decision, 'score': score, 'price': latest['close'],
            'rsi': latest['rsi'], 'macd': latest['macd'],
            'support': s_level, 'resistance': r_level,
            'stop_loss': round(stop_loss, 2), 'target': round(target, 2),
            'historical_low': round(historical_low, 2),
            'proximity_to_low': round(proximity_to_low, 1),
            'reasons': reasons, 'date': latest.name, 'patterns': patterns,
        }

    def show_analysis(self):
        print(f"\n{'='*50}")
        print(f"تحليل سهم: {self.symbol}")
        print(f"تاريخ التحليل: {self.signals['date']}")
        print(f"السعر الحالي: {self.signals['price']:.2f} ج.م")
        print(f"{'='*50}")
        print(f"القرار النهائي: {self.signals['decision']}")
        print(f"درجة تقييم النظام الكلية (Score): {self.signals['score']}%")
        if 'ai_prediction' in self.signals:
            print(f"تنبؤ الـ AI العصبوني: {self.signals['ai_prediction']}")
        print(f"البعد عن القاع التاريخي: % {self.signals['proximity_to_low']}+")
        print(f"{'='*50}")
        print("الأسباب الفنية واستراتيجية الدخول:")
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
            plt.savefig(BASE_DIR / f'{self.symbol}_analysis.png', dpi=150)
            print("تم حفظ الشارت بنجاح")
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
            if body_c / range_c < 0.1:
                if c['close'] < p['close'] - body_p * 0.3:
                    patterns.append("🔴 دوجي عند القمة — احتمال انعكاس هابط")
                elif c['close'] > p['close'] + body_p * 0.3:
                    patterns.append("🟢 دوجي عند القاع — احتمال انعكاس صاعد")
            upper = c['high'] - max(c['close'], c['open'])
            lower = min(c['close'], c['open']) - c['low']
            if lower > body_c * 2 and upper < body_c * 0.3 and body_c / range_c > 0.1:
                patterns.append("🟢 شمعة مطرقة (Hammer) — احتمال انعكاس صاعد")
            if upper > body_c * 2 and lower < body_c * 0.3 and body_c / range_c > 0.1:
                patterns.append("🔴 شمعة نجم ثاقب (Shooting Star) — احتمال انعكاس هابط")
            if p['close'] > p['open'] and c['close'] < c['open'] and c['open'] > p['close'] and c['close'] < p['open']:
                patterns.append("🔴 شمعة ابتلاع هابطة (Bearish Engulfing)")
            if p['close'] < p['open'] and c['close'] > c['open'] and c['open'] < p['close'] and c['close'] > p['open']:
                patterns.append("🟢 شمعة ابتلاع صاعدة (Bullish Engulfing)")
        return patterns

    def run(self):
        self.fetch_data()
        self.add_indicators()
        ai = AIAnalyzer()
        ai.train_or_load(self.data, self.symbol)
        ai_dir, ai_conf = ai.get_signal()
        self.generate_signals(ai_dir, ai_conf)
        if ai_dir:
            self.signals['ai_prediction'] = f"{ai_dir} (ثقة {ai_conf:.0%})"
        self.show_analysis()


LEADERS_TICKERS = ["COMI.CA", "FWRY.CA", "SKPC.CA", "EKHO.CA", "ABUK.CA", "TMGH.CA"]


def check_volume_spike(symbol, df):
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    col_key = 'volume' if 'volume' in d.columns else 'Volume'
    if len(d) < 20:
        return None
    latest_volume = d[col_key].iloc[-1]
    avg_volume = d[col_key].iloc[-20:-1].mean()
    volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0
    if volume_ratio >= 3.0:
        close_col = 'close' if 'close' in d.columns else 'Close'
        alert_msg = (
            f"🌊 **إنذار تدفق سيولة (Volume Spike)**\n"
            f"-----------------------------------\n"
            f"⚠️ السهم: *{symbol}*\n"
            f"📊 حجم التداول الآن أعلى من المعتاد بـ `{volume_ratio:.1f}` مرة!\n"
            f"💰 السعر الحالي: `{d[close_col].iloc[-1]:.2f} ج.م`\n"
            f"💡 *هناك دخول قوي للسيولة، راقب اختراق المقاومات.*"
        )
        return alert_msg
    return None


def check_historical_bottoms(symbol, df):
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    d.columns = [c.lower() for c in d.columns]
    latest_price = d['close'].iloc[-1]
    hist_low = d['low'].min()
    proximity = ((latest_price - hist_low) / hist_low) * 100
    if proximity <= 1.5:
        alert_msg = (
            f"🚨 **تنبيه صيد القاع (Bottom Alert)**\n"
            f"-----------------------------------\n"
            f"⚠️ السهم القيادي: *{symbol}*\n"
            f"💰 السعر الحالي: `{latest_price:.2f} ج.م`\n"
            f"📉 القاع التاريخي: `{hist_low:.2f} ج.م`\n"
            f"🎯 المسافة عن القاع: `% {proximity:.1f}` فقط!\n"
            f"💡 *فرصة ارتداد قوية محتملة (High Margin of Safety)*"
        )
        return alert_msg
    return None


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
