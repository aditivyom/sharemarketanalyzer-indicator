import queue
import threading
import datetime as dt
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------------------------------------------------ DATA
SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT",
    "NIFTY FIN SERVICE": "NIFTY_FIN_SERVICE.NS",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "INFOSYS": "INFY.NS",
    "SBI": "SBIN.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "ITC": "ITC.NS",
    "ADANI ENT": "ADANIENT.NS",
    "GOLD (USD)": "GC=F",
    "BITCOIN (USD)": "BTC-USD",
}

# timeframe label -> (interval, period)
TIMEFRAMES = {
    "1 min": ("1m", "1d"),
    "5 min": ("5m", "5d"),
    "15 min": ("15m", "5d"),
    "1 hour": ("60m", "1mo"),
    "1 day": ("1d", "1y"),
}

BG, PANEL, FG = "#0e1117", "#161b22", "#e6edf3"
GREEN, RED, YELLOW, BLUE, PURPLE = "#26a69a", "#ef5350", "#f0b90b", "#58a6ff", "#bc8cff"


def fetch(symbol, interval, period):
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError("No data received (market closed / wrong symbol / no internet)")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ------------------------------------------------------------ INDICATORS
def add_indicators(df):
    d = df.copy()
    c = d["Close"]
    d["EMA9"] = c.ewm(span=9, adjust=False).mean()
    d["EMA21"] = c.ewm(span=21, adjust=False).mean()
    d["SMA50"] = c.rolling(50).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["HIST"] = d["MACD"] - d["SIG"]

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    d["BBU"], d["BBM"], d["BBL"] = mid + 2 * sd, mid, mid - 2 * sd

    tr = pd.concat(
        [d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return d


# --------------------------------------------------------------- ANALYSIS
def analyze(d):
    """Score-based multi-indicator engine. Returns dict with signal + reasons."""
    last, prev = d.iloc[-1], d.iloc[-2]
    price = float(last["Close"])
    score, reasons = 0, []

    # 1) Trend
    if pd.notna(last["SMA50"]):
        if price > last["EMA21"] > last["SMA50"]:
            score += 2; reasons.append(("+2", "Strong uptrend: Price > EMA21 > SMA50"))
        elif price < last["EMA21"] < last["SMA50"]:
            score -= 2; reasons.append(("-2", "Strong downtrend: Price < EMA21 < SMA50"))
        elif price > last["SMA50"]:
            score += 1; reasons.append(("+1", "Price above SMA50 (mild bullish)"))
        else:
            score -= 1; reasons.append(("-1", "Price below SMA50 (mild bearish)"))
    else:
        if price > last["EMA21"]:
            score += 1; reasons.append(("+1", "Price above EMA21"))
        else:
            score -= 1; reasons.append(("-1", "Price below EMA21"))

    # 2) EMA crossover
    now_up = last["EMA9"] > last["EMA21"]
    was_up = prev["EMA9"] > prev["EMA21"]
    if now_up and not was_up:
        score += 2; reasons.append(("+2", "Fresh bullish EMA9/21 crossover"))
    elif not now_up and was_up:
        score -= 2; reasons.append(("-2", "Fresh bearish EMA9/21 crossover"))
    elif now_up:
        score += 1; reasons.append(("+1", "EMA9 above EMA21"))
    else:
        score -= 1; reasons.append(("-1", "EMA9 below EMA21"))

    # 3) RSI
    rsi = float(last["RSI"])
    if rsi < 30:
        score += 2; reasons.append(("+2", f"RSI {rsi:.1f}: oversold, bounce likely"))
    elif rsi > 70:
        score -= 2; reasons.append(("-2", f"RSI {rsi:.1f}: overbought, pullback likely"))
    elif rsi >= 55:
        score += 1; reasons.append(("+1", f"RSI {rsi:.1f}: bullish momentum"))
    elif rsi <= 45:
        score -= 1; reasons.append(("-1", f"RSI {rsi:.1f}: bearish momentum"))
    else:
        reasons.append(("0", f"RSI {rsi:.1f}: neutral"))

    # 4) MACD
    if last["HIST"] > 0 and last["HIST"] > prev["HIST"]:
        score += 2; reasons.append(("+2", "MACD positive & rising"))
    elif last["HIST"] > 0:
        score += 1; reasons.append(("+1", "MACD positive but weakening"))
    elif last["HIST"] < 0 and last["HIST"] < prev["HIST"]:
        score -= 2; reasons.append(("-2", "MACD negative & falling"))
    else:
        score -= 1; reasons.append(("-1", "MACD negative but recovering"))

    # 5) Bollinger Bands
    if pd.notna(last["BBL"]):
        if price <= last["BBL"]:
            score += 1; reasons.append(("+1", "Price at lower Bollinger band (stretched down)"))
        elif price >= last["BBU"]:
            score -= 1; reasons.append(("-1", "Price at upper Bollinger band (stretched up)"))

    # 6) Short momentum
    if len(d) > 6:
        mom = (price / float(d["Close"].iloc[-6]) - 1) * 100
        if mom > 0.3:
            score += 1; reasons.append(("+1", f"5-bar momentum +{mom:.2f}%"))
        elif mom < -0.3:
            score -= 1; reasons.append(("-1", f"5-bar momentum {mom:.2f}%"))

    max_score = 10
    if score >= 5:
        signal = "STRONG BUY"
    elif score >= 3:
        signal = "BUY"
    elif score <= -5:
        signal = "STRONG SELL"
    elif score <= -3:
        signal = "SELL"
    else:
        signal = "HOLD / WAIT"

    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.005
    if "BUY" in signal:
        sl, tgt = price - 1.5 * atr, price + 3.0 * atr
    elif "SELL" in signal:
        sl, tgt = price + 1.5 * atr, price - 3.0 * atr
    else:
        sl = tgt = None

    # Smart tip
    tips = []
    if abs(score) < 3:
        tips.append("Indicators conflict - no edge. Wait for a clear breakout.")
    if rsi > 70 and "BUY" in signal:
        tips.append("Buy signal but RSI is hot - avoid chasing, wait for a dip.")
    if rsi < 30 and "SELL" in signal:
        tips.append("Sell signal but RSI oversold - avoid shorting the bottom.")
    if pd.notna(last["BBU"]) and (last["BBU"] - last["BBL"]) / price < 0.004:
        tips.append("Bollinger squeeze: a big move is brewing - wait for direction.")
    if "BUY" in signal or "SELL" in signal:
        tips.append("Risk max 1-2% of capital per trade and always use the stop-loss.")
    tips.append("Confirm with a higher timeframe before entering.")

    return dict(signal=signal, score=score, confidence=min(100, abs(score) / max_score * 100),
                price=price, sl=sl, target=tgt, reasons=reasons, tips=tips,
                rsi=rsi, atr=atr)


# -------------------------------------------------------------------- GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Live Market Analyzer - Nifty / BankNifty / Stocks")
        self.geometry("1400x850")
        self.configure(bg=BG)
        self.q = queue.Queue()
        self.running = False
        self.fetching = False
        self.job = None

        self._style()
        self._build_top()
        self._build_body()
        self.after(200, self._poll)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ---------- layout
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG)
        s.configure("TLabel", background=BG, foreground=FG)

    def _build_top(self):
        bar = tk.Frame(self, bg=PANEL, pady=8, padx=10)
        bar.pack(fill="x")

        def lab(t):
            tk.Label(bar, text=t, bg=PANEL, fg=FG, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(8, 3))

        lab("Market:")
        self.sym_var = tk.StringVar(value="NIFTY 50")
        self.sym_cb = ttk.Combobox(bar, textvariable=self.sym_var, values=list(SYMBOLS),
                                   width=20, font=("Segoe UI", 11))
        self.sym_cb.pack(side="left")
        self.sym_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_now())

        lab("Timeframe:")
        self.tf_var = tk.StringVar(value="5 min")
        tf = ttk.Combobox(bar, textvariable=self.tf_var, values=list(TIMEFRAMES),
                          width=8, state="readonly", font=("Segoe UI", 11))
        tf.pack(side="left")
        tf.bind("<<ComboboxSelected>>", lambda e: self.refresh_now())

        lab("Refresh (sec):")
        self.ref_var = tk.StringVar(value="15")
        ttk.Combobox(bar, textvariable=self.ref_var, values=["5", "10", "15", "30", "60"],
                     width=5, state="readonly", font=("Segoe UI", 11)).pack(side="left")

        self.btn = tk.Button(bar, text="▶ START LIVE", bg=GREEN, fg="white", bd=0, padx=14, pady=5,
                             font=("Segoe UI", 10, "bold"), command=self.toggle)
        self.btn.pack(side="left", padx=14)
        tk.Button(bar, text="⟳ Analyze Now", bg=BLUE, fg="white", bd=0, padx=12, pady=5,
                  font=("Segoe UI", 10, "bold"), command=self.refresh_now).pack(side="left")

        self.status = tk.Label(bar, text="Ready", bg=PANEL, fg=YELLOW, font=("Segoe UI", 9))
        self.status.pack(side="right", padx=10)

    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # chart
        self.fig = Figure(facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        # side panel
        side = tk.Frame(body, bg=PANEL, width=360, padx=14, pady=12)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        self.name_lbl = tk.Label(side, text="-", bg=PANEL, fg=FG, font=("Segoe UI", 15, "bold"))
        self.name_lbl.pack(anchor="w")
        self.price_lbl = tk.Label(side, text="--", bg=PANEL, fg=FG, font=("Segoe UI", 24, "bold"))
        self.price_lbl.pack(anchor="w")

        self.sig_lbl = tk.Label(side, text="WAITING", bg=BG, fg=YELLOW, font=("Segoe UI", 22, "bold"),
                                pady=12)
        self.sig_lbl.pack(fill="x", pady=8)
        self.conf_lbl = tk.Label(side, text="Confidence: --", bg=PANEL, fg=FG, font=("Segoe UI", 10))
        self.conf_lbl.pack(anchor="w")
        self.conf_bar = ttk.Progressbar(side, maximum=100, length=300)
        self.conf_bar.pack(anchor="w", pady=(2, 10))

        self.lvl_lbl = tk.Label(side, text="", bg=PANEL, fg=FG, justify="left",
                                font=("Consolas", 11, "bold"))
        self.lvl_lbl.pack(anchor="w", pady=4)

        tk.Label(side, text="WHY THIS SIGNAL", bg=PANEL, fg=BLUE,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 2))
        self.reason_box = tk.Text(side, height=11, bg=BG, fg=FG, bd=0, wrap="word",
                                  font=("Segoe UI", 9))
        self.reason_box.pack(fill="x")
        self.reason_box.tag_config("pos", foreground=GREEN)
        self.reason_box.tag_config("neg", foreground=RED)
        self.reason_box.tag_config("neu", foreground=YELLOW)

        tk.Label(side, text="SMART SUGGESTION", bg=PANEL, fg=PURPLE,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.tip_box = tk.Text(side, height=7, bg=BG, fg=FG, bd=0, wrap="word", font=("Segoe UI", 9))
        self.tip_box.pack(fill="x")

        tk.Label(side, text="Educational only - not financial advice.", bg=PANEL, fg="#8b949e",
                 font=("Segoe UI", 8)).pack(side="bottom", anchor="w")

    # ---------- control
    def toggle(self):
        self.running = not self.running
        if self.running:
            self.btn.config(text="■ STOP LIVE", bg=RED)
            self.refresh_now()
        else:
            self.btn.config(text="▶ START LIVE", bg=GREEN)
            if self.job:
                self.after_cancel(self.job)
                self.job = None
            self.status.config(text="Live stopped")

    def refresh_now(self):
        if self.fetching:
            return
        name = self.sym_var.get().strip()
        symbol = SYMBOLS.get(name, name.upper())   # allows typing any Yahoo symbol
        interval, period = TIMEFRAMES[self.tf_var.get()]
        self.fetching = True
        self.status.config(text="Fetching data...")
        threading.Thread(target=self._worker, args=(name, symbol, interval, period),
                         daemon=True).start()

    def _worker(self, name, symbol, interval, period):
        try:
            df = add_indicators(fetch(symbol, interval, period))
            if len(df) < 30:
                raise ValueError("Not enough candles yet - try a bigger timeframe")
            self.q.put(("ok", name, df, analyze(df)))
        except Exception as e:
            self.q.put(("err", str(e)))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self.fetching = False
                if msg[0] == "ok":
                    _, name, df, res = msg
                    self.draw(name, df)
                    self.show(name, res)
                    self.status.config(text="Updated " + dt.datetime.now().strftime("%H:%M:%S"))
                else:
                    self.status.config(text="Error: " + msg[1][:70])
                if self.running:
                    secs = int(self.ref_var.get())
                    self.job = self.after(secs * 1000, self.refresh_now)
        except queue.Empty:
            pass
        self.after(200, self._poll)

    # ---------- display
    def show(self, name, r):
        self.name_lbl.config(text=name)
        self.price_lbl.config(text=f"{r['price']:,.2f}")
        col = GREEN if "BUY" in r["signal"] else RED if "SELL" in r["signal"] else YELLOW
        self.sig_lbl.config(text=r["signal"], fg=col)
        self.conf_lbl.config(text=f"Confidence: {r['confidence']:.0f}%   (score {r['score']:+d})")
        self.conf_bar["value"] = r["confidence"]
        if r["sl"]:
            rr = abs(r["target"] - r["price"]) / abs(r["price"] - r["sl"])
            self.lvl_lbl.config(text=f"Entry : {r['price']:,.2f}\nStop  : {r['sl']:,.2f}\n"
                                     f"Target: {r['target']:,.2f}\nR:R   : 1 : {rr:.1f}")
        else:
            self.lvl_lbl.config(text="No trade setup - stay out")

        self.reason_box.config(state="normal")
        self.reason_box.delete("1.0", "end")
        for pts, txt in r["reasons"]:
            tag = "pos" if pts.startswith("+") else "neg" if pts.startswith("-") else "neu"
            self.reason_box.insert("end", f"{pts:>2}  {txt}\n", tag)
        self.reason_box.config(state="disabled")

        self.tip_box.config(state="normal")
        self.tip_box.delete("1.0", "end")
        for t in r["tips"]:
            self.tip_box.insert("end", "• " + t + "\n")
        self.tip_box.config(state="disabled")

    def draw(self, name, df, n=90):
        d = df.tail(n).reset_index()
        x = np.arange(len(d))
        self.fig.clear()
        gs = self.fig.add_gridspec(3, 1, height_ratios=[5, 1.6, 1.6], hspace=0.06,
                                   left=0.06, right=0.98, top=0.95, bottom=0.07)
        ax = self.fig.add_subplot(gs[0])
        ax2 = self.fig.add_subplot(gs[1], sharex=ax)
        ax3 = self.fig.add_subplot(gs[2], sharex=ax)

        for a in (ax, ax2, ax3):
            a.set_facecolor(BG)
            a.tick_params(colors="#8b949e", labelsize=8)
            a.grid(color="#21262d", lw=0.5)
            for sp in a.spines.values():
                sp.set_color("#30363d")

        # candles
        up = d["Close"] >= d["Open"]
        col = np.where(up, GREEN, RED)
        ax.vlines(x, d["Low"], d["High"], colors=col, lw=1)
        body = (d["Close"] - d["Open"]).abs().clip(lower=d["Close"].max() * 1e-5)
        ax.bar(x, body, bottom=np.minimum(d["Open"], d["Close"]), color=col, width=0.65)
        ax.plot(x, d["EMA9"], color=YELLOW, lw=1, label="EMA9")
        ax.plot(x, d["EMA21"], color=BLUE, lw=1, label="EMA21")
        ax.plot(x, d["SMA50"], color=PURPLE, lw=1, label="SMA50")
        ax.fill_between(x, d["BBL"], d["BBU"], color="#58a6ff", alpha=0.07)
        ax.axhline(d["Close"].iloc[-1], color="#8b949e", ls="--", lw=0.7)
        ax.set_title(f"{name}  -  Live Chart", color=FG, fontsize=12, loc="left")
        ax.legend(loc="upper left", facecolor=PANEL, edgecolor="#30363d", labelcolor=FG, fontsize=8)
        ax.tick_params(labelbottom=False)

        # RSI
        ax2.plot(x, d["RSI"], color=PURPLE, lw=1.1)
        ax2.axhline(70, color=RED, ls="--", lw=0.6)
        ax2.axhline(30, color=GREEN, ls="--", lw=0.6)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", color="#8b949e", fontsize=8)
        ax2.tick_params(labelbottom=False)

        # MACD
        ax3.bar(x, d["HIST"], color=np.where(d["HIST"] >= 0, GREEN, RED), width=0.65)
        ax3.plot(x, d["MACD"], color=BLUE, lw=1)
        ax3.plot(x, d["SIG"], color=YELLOW, lw=1)
        ax3.set_ylabel("MACD", color="#8b949e", fontsize=8)

        # x labels
        date_col = d.columns[0]
        intraday = self.tf_var.get() != "1 day"
        fmt = "%H:%M" if intraday else "%d %b"
        idx = np.linspace(0, len(d) - 1, 8, dtype=int)
        ax3.set_xticks(idx)
        ax3.set_xticklabels([pd.Timestamp(d[date_col].iloc[i]).strftime(fmt) for i in idx])
        ax.set_xlim(-1, len(d) + 2)
        self.canvas.draw_idle()


if __name__ == "__main__":
    App().mainloop()
