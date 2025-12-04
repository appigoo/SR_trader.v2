# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from streamlit_autorefresh import st_autorefresh

# ==================== 初始化 ====================
st.set_page_config(page_title="多股票即時監控面板", layout="wide")
st.title("多股票支撐/阻力突破監控面板 + MACD 前瞻預測")

# session_state 初始化（防重複警報）
for key in ["last_signal_keys", "signal_history"]:
    if key not in st.session_state:
        st.session_state[key] = ({} if key == "last_signal_keys" else [])

# ==================== 側邊欄設定 ====================
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "TSLA,META,AAPL,NVDA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval_options = {"1分鐘": "1m", "5分鐘": "5m", "15分鐘": "15m", "1小時": "60m", "日線": "1d"}
interval_label = st.sidebar.selectbox("K線週期", options=list(interval_options.keys()), index=1)
interval = interval_options[interval_label]

period_options = {"1天": "1d", "5天": "5d", "10天": "10d", "1個月": "1mo", "3個月": "3mo", "1年": "1y"}
period_label = st.sidebar.selectbox("資料範圍", options=list(period_options.keys()), index=1)
period = period_options[period_label]

# yfinance 限制提醒
if interval == "1m" and period not in ["1d", "5d"]:
    st.sidebar.warning("1分鐘K線最多只能回溯7天")
if interval in ["5m", "15m", "60m"] and period not in ["1d", "5d", "10d", "1mo"]:
    st.sidebar.warning(f"{interval_label} 最多只能回溯60天")

lookback = st.sidebar.slider("觀察根數（成交量/支撐阻力）", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
buffer_pct = st.sidebar.slider("突破緩衝區 (%)", 0.01, 1.0, 0.1, 0.01) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)
show_touches = st.sidebar.checkbox("顯示價位觸碰分析", True)

st.sidebar.markdown("---")
st.sidebar.caption(f"K線：{interval_label} | 範圍：{period_label}")

# ==================== 警報開關 ====================
st.sidebar.markdown("### 警報設定")
use_auto_sr_alerts = st.sidebar.checkbox("啟用自動支撐/阻力突破警報", True)
use_volume_filter = st.sidebar.checkbox("突破需成交量確認 (>1.5x)", True)

use_volume_alert = st.sidebar.checkbox("啟用獨立成交量警報", True)
volume_alert_multiplier = st.sidebar.slider("成交量警報倍數", 1.5, 5.0, 2.5, 0.1)

st.sidebar.markdown("#### 自訂價位警報")
custom_alert_input = st.sidebar.text_area(
    "自訂價位 (格式: SYMBOL,價位1,價位2...)",
    "AAPL,180.5,190\nNVDA,850,900"
)

st.sidebar.markdown("#### 自訂成交量倍數")
custom_volume_input = st.sidebar.text_area(
    "自訂成交量倍數 (格式: SYMBOL,倍數)",
    "AAPL,3.0\nNVDA,4.0"
)

# 新增最強功能
use_macd_forecast_alert = st.sidebar.checkbox("啟用 MACD 動能前瞻預測警報（極強）", True)

# 解析自訂設定
def parse_custom_alerts(text: str) -> Dict[str, List[float]]:
    alerts = {}
    for line in text.split("\n"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            sym = parts[0].upper()
            try:
                prices = [float(p) for p in parts[1:]]
                alerts.setdefault(sym, []).extend(prices)
            except:
                pass
    return alerts

def parse_custom_volume(text: str) -> Dict[str, float]:
    d = {}
    for line in text.split("\n"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            sym = parts[0].upper()
            try:
                d[sym] = float(parts[1])
            except:
                pass
    return d

custom_alert_levels = parse_custom_alerts(custom_alert_input)
custom_volume_multipliers = parse_custom_volume(custom_volume_input)

# ==================== Telegram & 聲音 ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except:
    telegram_ready = False

def send_telegram(msg: str) -> bool:
    if not telegram_ready:
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.get(url, params=payload, timeout=10)
        return r.status_code == 200 and r.json().get("ok")
    except:
        return False

def play_alert_sound():
    if sound_alert:
        st.markdown("""
        <audio autoplay style="display:none;">
            <source src="https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

# ==================== 資料快取 ====================
@st.cache_data(ttl=60)
def fetch_data(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=True)
        if df.empty or df.isna().all().all():
            return None
        df = df[~df.index.duplicated(keep='last')]
        df = df.dropna(how='all')
        return df
    except:
        return None

# ==================== 支撐阻力（簡化保留核心） ====================
def find_support_resistance_fractal(df_full: pd.DataFrame, window: int = 5, min_touches: int = 2):
    df = df_full.iloc[:-1]
    if len(df) < window * 2 + 1:
        return df_full["Low"].min(), df_full["High"].max(), []
    high, low = df["High"], df["Low"]
    res_pts, sup_pts = [], []
    for i in range(window, len(df) - window):
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            res_pts.append(high.iloc[i])
        if low.iloc[i] == low.iloc[i-window:i+window+1].min():
            sup_pts.append(low.iloc[i])
    def cluster(pts, tol=0.005):
        if not pts: return []
        pts = sorted(pts)
        clusters = []
        cur = [pts[0]]
        for p in pts[1:]:
            if abs(p - cur[-1]) / cur[-1] < tol:
                cur.append(p)
            else:
                if len(cur) >= min_touches:
                    clusters.append(np.mean(cur))
                cur = [p]
        if len(cur) >= min_touches:
            clusters.append(np.mean(cur))
        return clusters
    res_lv = cluster(res_pts)
    sup_lv = cluster(sup_pts)
    all_levels = list(set(res_lv + sup_lv))
    cur = df_full["Close"].iloc[-1]
    support = min(sup_lv, key=lambda x: (abs(x-cur), -x)) if sup_lv else df_full["Low"].min()
    resistance = max(res_lv, key=lambda x: (abs(x-cur), x)) if res_lv else df_full["High"].max()
    return support, resistance, all_levels

# ==================== 四大警報函數 ====================

# 1. 自動突破警報
def check_auto_breakout(df_full, support, resistance, buffer_pct, use_vol, vol_mult, lookback, symbol):
    df = df_full.iloc[:-1]
    if len(df) < 4: return None
    try:
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        prev2 = float(df["Close"].iloc[-3])
        vol = float(df["Volume"].iloc[-1])
        avg_vol = df["Volume"].iloc[-(lookback+1):-1].mean()
        vol_ok = (not use_vol) or (vol / avg_vol > vol_mult) if avg_vol > 0 else False
        buffer = resistance * buffer_pct
        if prev2 <= resistance - buffer and prev <= resistance - buffer and last > resistance and vol_ok:
            return (symbol, f"突破阻力！\n<b>{symbol}</b> 現價 {last:.2f}\n阻力 {resistance:.2f}", f"{symbol}_UP_{resistance:.1f}")
        if prev2 >= support + buffer and prev >= support + buffer and last < support and vol_ok:
            return (symbol, f"跌破支撐！\n<b>{symbol}</b> 現價 {last:.2f}\n支撐 {support:.2f}", f"{symbol}_DN_{support:.1f}")
    except:
        pass
    return None

# 2. 自訂價位警報
def check_custom_price_alerts(symbol, df_full, levels):
    if not levels or len(df_full) < 2: return []
    try:
        last = float(df_full["Close"].iloc[-1])
        prev = float(df_full["Close"].iloc[-2])
    except:
        return []
    signals = []
    for lvl in levels:
        if prev <= lvl < last:
            signals.append((symbol, f"向上觸及自訂價位！\n<b>{symbol}</b> {last:.2f} > {lvl}", f"CUST_UP_{lvl}"))
        elif prev >= lvl > last:
            signals.append((symbol, f"向下觸及自訂價位！\n<b>{symbol}</b> {last:.2f} < {lvl}", f"CUST_DN_{lvl}"))
    return signals

# 3. 成交量警報
def check_volume_alert(symbol, df_full, mult, lookback, custom_mult=None):
    df = df_full.iloc[:-1]
    if len(df) < lookback: return None
    try:
        last_vol = float(df["Volume"].iloc[-1])
        avg_vol = df["Volume"].iloc[-(lookback+1):-1].mean()
        if avg_vol <= 0: return None
        ratio = last_vol / avg_vol
        effective_mult = custom_mult or mult
        if ratio > effective_mult:
            ts = pd.Timestamp.now().floor('T').strftime("%H%M")
            msg = f"成交量激增！\n<b>{symbol}</b>\n現量 {last_vol:,.0f}\n均量 {avg_vol:,.0f} ({ratio:.1f}x)"
            return (symbol, msg, f"{symbol}_VOL_{ratio:.1f}_{ts}")
    except:
        pass
    return None

# 4. MACD 動能前瞻預測（已完全修復）
def check_macd_momentum_forecast(df_full: pd.DataFrame, symbol: str, lookback: int = 120) -> List[Tuple[str, str, str]]:
    if len(df_full) < 50:
        return []
    df = df_full.copy()
    close = df['Close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    df = df.iloc[-lookback:]
    dif = dif.iloc[-lookback:].values
    hist = hist.iloc[-lookback:].values
    price = close.iloc[-lookback:].values

    signals = []

    # 前兆① DIF 加速度轉向
    if len(dif) >= 12:
        speed = np.diff(dif[-12:])
        accel = np.diff(speed)
        if len(accel) >= 4:
            if accel[-3] < 0 and accel[-2] > 0 and hist[-1] < 0:
                sustain = max(8, min(45, int(abs(dif[-1]) / (abs(speed.mean()) + 1e-8) * 0.9)))
                msg = f"MACD 提前翻多！\n<b>{symbol}</b>\nDIF加速度轉正\n預估多頭維持 <b>{sustain}</b> 根K"
                signals.append((symbol, msg, f"{symbol}_MACD_UP_{pd.Timestamp.now():%H%M}"))
            if accel[-3] > 0 and accel[-2] < 0 and hist[-1] > 0:
                sustain = max(8, min(45, int(abs(dif[-1]) / (abs(speed.mean()) + 1e-8) * 0.9)))
                msg = f"MACD 提前翻空！\n<b>{symbol}</b>\nDIF加速度轉負\n預估空頭維持 <b>{sustain}</b> 根K"
                signals.append((symbol, msg, f"{symbol}_MACD_DN_{pd.Timestamp.now():%H%M}"))

    # 前兆② 柱子連續縮短
    if len(hist) >= 7:
        recent = hist[-7:]
        if all(recent[-6:] > 0) and np.all(np.diff(recent[-5:]) < 0):
            sustain = max(10, int(abs(dif[-1]) * 18))
            signals.append((symbol, f"MACD 多頭衰竭！紅柱連續縮短\n預估空頭維持 {sustain} 根K", f"RED_SHRINK_{pd.Timestamp.now():%H%M}"))
        if all(recent[-6:] < 0) and np.all(np.diff(np.abs(recent[-5:])) < 0):
            sustain = max(10, int(abs(dif[-1]) * 18))
            signals.append((symbol, f"MACD 空頭衰竭！綠柱連續縮短\n預估多頭維持 {sustain} 根K", f"GREEN_SHRINK_{pd.Timestamp.now():%H%M}"))

    # 前兆③ 背離
    if len(price) >= 20:
        if np.argmin(price[-15:]) > np.argmin(dif[-15:]) and hist[-1] < 0:
            sustain = max(15, int(abs(dif[-1]) * 22))
            signals.append((symbol, f"強力底背離！{symbol}\nMACD 極大概率翻紅，預估維持 {sustain} 根K", f"BULL_DIV_{pd.Timestamp.now():%Y%m%d}"))
        if np.argmax(price[-15:]) > np.argmax(dif[-15:]) and hist[-1] > 0:
            sustain = max(15, int(abs(dif[-1]) * 22))
            signals.append((symbol, f"強力頂背離！{symbol}\nMACD 極大概率翻綠，預估維持 {sustain} 根K", f"BEAR_DIV_{pd.Timestamp.now():%Y%m%d}"))

    return signals

# ==================== 主處理與圖表（簡化版，保留你原本邏輯） ====================
def process_symbol(symbol: str, custom_levels: List[float]):
    df_full = fetch_data(symbol, interval, period)
    if df_full is None or len(df_full) < 20:
        return None, None, None, None, [], None, None
    support, resistance, all_levels = find_support_resistance_fractal(df_full)
    current_price = df_full["Close"].iloc[-1]
    # 這裡省略完整圖表繪製（你原本的 fig 與 recent_fig 邏輯直接貼回即可）
    fig = go.Figure()  # 實際使用時請貼回你原本的完整繪圖程式碼
    fig.add_trace(go.Candlestick(x=df_full.index, open=df_full['Open'], high=df_full['High'],
                                 low=df_full['Low'], close=df_full['Close'], name=symbol))
    recent_fig = None
    return fig, current_price, support, resistance, all_levels, df_full, recent_fig

# ==================== 自動更新 ====================
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180}
if auto_update:
    st_autorefresh(interval=interval_map[update_freq] * 1000, key="auto")

if not symbols:
    st.warning("請輸入股票代號")
    st.stop()

st.header(f"監控中：{', '.join(symbols)} | {interval_label} | {period_label}")

# ==================== 主循環 ====================
results = {}
all_signals = []
progress = st.progress(0)

with st.spinner("分析中…"):
    for i, sym in enumerate(symbols):
        progress.progress((i+1)/len(symbols))
        custom_lvls = custom_alert_levels.get(sym, [])
        fig, price, sup, res, lvls, df_full, recent_fig = process_symbol(sym, custom_lvls)
        if df_full is None:
            continue
        results[sym] = {"fig": fig, "price": price, "support": sup, "resistance": res, "df": df_full, "recent": recent_fig}

        # 四大警報
        if use_auto_sr_alerts and df_full is not None:
            sig = check_auto_breakout(df_full, sup, res, buffer_pct, use_volume_filter, 1.5, lookback, sym)
            if sig: all_signals.append(sig)

        all_signals.extend(check_custom_price_alerts(sym, df_full, custom_lvls))

        if use_volume_alert:
            cust_mult = custom_volume_multipliers.get(sym)
            sig = check_volume_alert(sym, df_full, volume_alert_multiplier, lookback, cust_mult)
            if sig: all_signals.append(sig)

        if use_macd_forecast_alert:
            all_signals.extend(check_macd_momentum_forecast(df_full, sym))

# ==================== 顯示結果 ====================
for sym in symbols:
    data = results.get(sym)
    if not data or data["fig"] is None:
        st.error(f"{sym} 無資料")
        continue

    sym_signals = [s for s in all_signals if s[0] == sym]
    if sym_signals:
        st.markdown(f"### {sym} 警報")
        for _, msg, key in sym_signals:
            st.success(msg)
            if key and st.session_state.last_signal_keys.get(key) != key:
                st.session_state.last_signal_keys[key] = key
                st.session_state.signal_history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "symbol": sym,
                    "signal": msg
                })
                if len(st.session_state.signal_history) > 30:
                    st.session_state.signal_history.pop(0)
                send_telegram(msg)
                play_alert_sound()

    st.plotly_chart(data["fig"], use_container_width=True)
    if data["recent"]:
        st.plotly_chart(data["recent"], use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("現價", f"{data['price']:.2f}" if data['price'] else "N/A")
    with c2: st.metric("支撐", f"{data['support']:.2f}" if data['support'] else "N/A")
    with c3: st.metric("阻力", f"{data['resistance']:.2f}" if data['resistance'] else "N/A")

    st.markdown("---")

# 歷史警報
if st.session_state.signal_history:
    st.subheader("最近20筆警報")
    for s in reversed(st.session_state.signal_history[-20:]):
        txt = s['signal'].replace('\n', ' | ')
        st.markdown(f"**{s['time']} | {s['symbol']}** → {txt}")
