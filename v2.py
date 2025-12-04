# app.py - 終極版：多股票即時監控 + MACD 提前預測 + Telegram 警報
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
st.title("多股票支撐/阻力 + MACD 提前預測監控面板")

for key in ["last_signal_keys", "signal_history"]:
    if key not in st.session_state:
        st.session_state[key] = ({} if key == "last_signal_keys" else [])

# ==================== 側邊欄 ====================
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "TSLA,META,NVDA,AAPL")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval_options = {"1分鐘": "1m", "5分鐘": "5m", "15分鐘": "15m", "1小時": "60m", "日線": "1d"}
interval_label = st.sidebar.selectbox("K線週期", options=list(interval_options.keys()), index=1)
interval = interval_options[interval_label]

period_options = {"1天": "1d", "5天": "5d", "10天": "10d", "1個月": "1mo", "3個月": "3mo", "1年": "1y", "10年": "10y"}
period_label = st.sidebar.selectbox("資料範圍", options=list(period_options.keys()), index=2)
period = period_options[period_label]

if interval == "1m" and period not in ["1d", "5d", "7d"]:
    st.sidebar.warning("1分鐘K線最多只能回溯7天")
if interval in ["5m", "15m", "60m"] and period in ["3mo", "1y", "10y"]:
    st.sidebar.warning(f"{interval_label}最多只能回溯60天，已自動調整")
    period = "60d" if period in ["3mo", "1y", "10y"] else period

lookback = st.sidebar.slider("觀察根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
buffer_pct = st.sidebar.slider("突破緩衝區 (%)", 0.01, 1.0, 0.1, 0.01) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)
show_touches = st.sidebar.checkbox("顯示價位觸碰分析", True)

st.sidebar.markdown("---")
st.sidebar.caption(f"**K線**：{interval_label} | **範圍**：{period_label}")

# ==================== 警報設定 ====================
st.sidebar.markdown("### 警報設定")
use_auto_sr_alerts = st.sidebar.checkbox("啟用自動 S/R 突破警報", True)
use_volume_filter = st.sidebar.checkbox("S/R 突破需成交量確認 (>1.5x)", True)
use_volume_alert = st.sidebar.checkbox("啟用獨立成交量警報", True)
volume_alert_multiplier = st.sidebar.slider("成交量警報倍數", 1.5, 5.0, 2.5, 0.1)

st.sidebar.markdown("#### 自訂價位警報")
custom_alert_input = st.sidebar.text_area(
    "自訂警報價位 (格式: SYMBOL,價位1,價位2...)",
    "AAPL,180.5,190\nNVDA,850,900.5"
)

def parse_custom_alerts(text: str) -> Dict[str, List[float]]:
    alerts = {}
    for line in text.split("\n"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            sym = parts[0].upper()
            try:
                alerts[sym] = [float(p) for p in parts[1:]]
            except:
                continue
    return alerts

custom_alert_levels = parse_custom_alerts(custom_alert_input)

# ==================== Telegram ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except:
    BOT_TOKEN = CHAT_ID = None
    telegram_ready = False

def send_telegram_alert(msg: str) -> bool:
    if not telegram_ready:
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.get(url, params=payload, timeout=10)
        return r.json().get("ok", False)
    except:
        return False

def play_alert_sound():
    if sound_alert:
        st.markdown("""
        <audio autoplay><source src="https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3" type="audio/mpeg"></audio>
        """, unsafe_allow_html=True)

# ==================== 資料快取 ====================
@st.cache_data(ttl=55)
def fetch_data(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or df["Close"].isna().all():
            return None
        df = df[~df.index.duplicated(keep='last')]
        return df.dropna(how='all')
    except:
        return None

# ==================== S/R 與價位觸碰 ====================
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
    all_lv = list(set(res_lv + sup_lv))
    support = min(sup_lv, key=lambda x: abs(x - df_full["Close"].iloc[-1])) if sup_lv else df_full["Low.min()
    resistance = max(res_lv, key=lambda x: abs(x - df_full["Close"].iloc[-1])) if res_lv else df_full["High"].max()
    return support, resistance, all_lv

# ==================== 警報函數 ====================
def check_auto_breakout(df_full, support, resistance, buffer, use_vol, vol_mult, lookback, symbol):
    df = df_full.iloc[:-1]
    if len(df) < 10: return None
    last, prev, prev2 = df["Close"].iloc[-1], df["Close"].iloc[-2], df["Close"].iloc[-3]
    vol_ok = (not use_vol) or (df["Volume"].iloc[-1] > df["Volume"].iloc[-(lookback+1):-1].mean() * vol_mult)
    buf = resistance * buffer
    if prev2 <= resistance - buf and prev <= resistance - buf and last > resistance and vol_ok:
        return (symbol, f"突破阻力！<b>{symbol}</b> 現價 {last:.2f} 阻力 {resistance:.2f}", f"{symbol}_UP_{resistance:.1f}")
    if prev2 >= support + buf and prev >= support + buf and last < support and vol_ok:
        return (symbol, f"跌破支撐！<b>{symbol}</b> 現價 {last:.2f} 支撐 {support:.2f}", f"{symbol}_DN_{support:.1f}")
    return None

def check_volume_alert(symbol, df_full, mult, lookback, custom_mult=None):
    df = df_full.iloc[:-1]
    if len(df) < lookback: return None
    curr_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-(lookback+1):-1].mean()
    effective_mult = custom_mult if custom_mult else mult
    if curr_vol > avg_vol * effective_mult:
        ratio = curr_vol / avg_vol
        key = f"{symbol}_VOL_{ratio:.1f}x_{pd.Timestamp.now().strftime('%H%M")}"
        return (symbol, f"成交量激增！<b>{symbol}</b> {ratio:.1f}x", key)
    return None

# ==================== 核心：process_symbol（含MACD提前預測） ====================
def process_symbol(symbol: str, custom_levels: List[float]):
    df_full = fetch_data(symbol, interval, period)
    if df_full is None or len(df_full) < 40:
        return None, None, None, None, [], None, None, False, False

    df = df_full.iloc[:-1]

    # === MACD 計算 ===
    exp12 = df_full['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df_full['Close'].ewm(span=26, adjust=False).mean()
    dif = exp12 - exp26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    df_full['DIF'], df_full['DEA'], df_full['HIST'] = dif, dea, hist

    # === 四大提前訊號 ===
    dif_speed = dif.diff()
    dif_accel = dif_speed.diff()
    accel_up   = (dif_accel > 0) & (dif_accel.shift(1) <= 0)
    accel_dn   = (dif_accel < 0) & (dif_accel.shift(1) >= 0)

    hist_shrink_up = (hist < 0) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2)) & (hist.shift(2) > hist.shift(3))
    hist_shrink_dn = (hist > 0) & (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2)) & (hist.shift(2) < hist.shift(3))

    bull_div = (df_full['Low'] < df_full['Low'].shift(1)) & (dif > dif.shift(1)) & (hist < 0)
    bear_div = (df_full['High'] > df_full['High'].shift(1)) & (dif < dif.shift(1)) & (hist > 0)

    ema12 = df_full['Close'].ewm(span=12, adjust=False).mean()
    slope = ema12.diff(5)
    slope_flat_up = (slope.shift(1) > 0) & (slope <= slope.shift(1))
    slope_flat_dn = (slope.shift(1) < 0) & (slope >= slope.shift(1))

    df_full['MACD_Early_Up']   = accel_up | hist_shrink_up | bull_div | ((hist < 0) & slope_flat_dn)
    df_full['MACD_Early_Down'] = accel_dn | hist_shrink_dn | bear_div | ((hist > 0) & slope_flat_up)

    atr14 = (df_full['High'] - df_full['Low']).rolling(14).mean()
    df_full['MACD_Persistence'] = (dif.abs() / atr14).fillna(0)

    # === S/R ===
    support, resistance, all_levels = find_support_resistance_fractal(df_full)

    # === 圖表 ===
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_full.index, open=df_full['Open'], high=df_full['High'],
                                low=df_full['Low'], close=df_full['Close'], name="K線",
                                increasing_line_color='lime', decreasing_line_color='red'))
    colors = ['cyan', 'blue', 'orange', 'magenta', 'yellow']
    for i, p in enumerate([5,10,20,40,60]):
        df_full[f'EMA_{p}'] = df_full['Close'].ewm(span=p, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df_full.index, y=df_full[f'EMA_{p}'], name=f'EMA{p}', line=dict(color=colors[i], width=1)))
    fig.add_hline(y=support, line_dash="dash", line_color="green", annotation_text=f"支撐 {support:.2f}")
    fig.add_hline(y=resistance, line_dash="dash", line_color="red", annotation_text=f"阻力 {resistance:.2f}")
    fig.update_layout(height=500, plot_bgcolor='black', paper_bgcolor='black', font_color='white', title=f"{symbol} 完整圖")

    # === 最近K線 + MACD提前訊號圖 ===
    recent_fig = None
    if len(df_full) >= 20:
        recent = df_full.tail(20).copy()
        recent['OBV'] = (np.sign(recent['Close'].diff()) * recent['Volume']).cumsum()
        from plotly.subplots import make_subplots
        recent_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                   subplot_titles=('價格與成交量', 'OBV', 'MACD 提前預測'),
                                   row_heights=[0.6, 0.25, 0.15])
        recent_fig.add_trace(go.Candlestick(x=recent.index, open=recent['Open'], high=recent['High'],
                                           low=recent['Low'], close=recent['Close'], name="K線",
                                           increasing_line_color='lime', decreasing_line_color='red'), row=1, col=1)
        for i, p in enumerate([5,10,20,40,60]):
            recent[f'EMA_{p}'] = recent['Close'].ewm(span=p, adjust=False).mean()
            recent_fig.add_trace(go.Scatter(x=recent.index, y=recent[f'EMA_{p}'], name=f'EMA{p}',
                                           line=dict(color=colors[i], width=1.5)), row=1, col=1)
        recent_fig.add_hline(y=recent["Close"].iloc[-1], line_color="purple", line_width=2,
                            annotation_text=f"現價 {recent['Close'].iloc[-1]:.2f}", row=1, col=1)
        recent['Vol_Color'] = np.where(recent['Close'] >= recent['Open'], 'lime', 'red')
        recent_fig.add_trace(go.Bar(x=recent.index, y=recent['Volume'], name="成交量",
                                   marker_color=recent['Vol_Color']), row=1, col=1)
        recent_fig.add_trace(go.Scatter(x=recent.index, y=recent['OBV'], name="OBV",
                                       line=dict(color='yellow', width=2)), row=2, col=1)

        up_idx = recent[recent['MACD_Early_Up']].index
        dn_idx = recent[recent['MACD_Early_Down']].index
        recent_fig.add_trace(go.Scatter(x=up_idx, y=recent.loc[up_idx, 'Low']*0.993,
                                       mode='markers', name='MACD 即將上升',
                                       marker=dict(symbol='triangle-up', size=20, color='lime',
                                                   line=dict(color='white', width=2))), row=3, col=1)
        recent_fig.add_trace(go.Scatter(x=dn_idx, y=recent.loc[dn_idx, 'High']*1.007,
                                       mode='markers', name='MACD 即將下跌',
                                       marker=dict(symbol='triangle-down', size=20, color='red',
                                                   line=dict(color='white', width=2))), row=3, col=1)
        pers = recent['MACD_Persistence'].iloc[-2]
        recent_fig.update_layout(title=f"{symbol} ・ MACD提前預測 (趨勢強度 {pers:.1f})", height=720,
                                plot_bgcolor='black', paper_bgcolor='black', font_color='white')

    current_price = df_full["Close"].iloc[-1]
    macd_up_alert = df_full['MACD_Early_Up'].iloc[-2]
    macd_dn_alert = df_full['MACD_Early_Down'].iloc[-2]

    return fig, current_price, support, resistance, all_levels, df_full, recent_fig, macd_up_alert, macd_dn_alert

# ==================== 自動更新 ====================
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180}
if auto_update:
    st_autorefresh(interval=interval_map[update_freq] * 1000, key="auto")
    st.sidebar.caption(f"自動更新：{update_freq}")

if not symbols:
    st.stop()

st.header(f"監控中：{', '.join(symbols)} | {interval_label} | {period_label}")

# ==================== 主迴圈 ====================
results = []
progress = st.progress(0)
for i, symbol in enumerate(symbols):
    progress.progress((i+1)/len(symbols))
    custom_lv = custom_alert_levels.get(symbol, [])

    fig, price, sup, res, levels, df_full, recent_fig, macd_up, macd_dn = process_symbol(symbol, custom_lv)

    if fig is None:
        st.error(f"{symbol} 無資料")
        continue

    # === 產生所有警報 ===
    if df_full is not None:
        # S/R 突破
        if use_auto_sr_alerts:
            sr_sig = check_auto_breakout(df_full, sup, res, buffer_pct, use_volume_filter, 1.5, lookback, symbol)
            if sr_sig: all_signals.append(sr_sig)

        # 成交量警報
        if use_volume_alert:
            vol_sig = check_volume_alert(symbol, df_full, volume_alert_multiplier, lookback)
            if vol_sig: all_signals.append(vol_sig)

        # MACD 提前警報
        if macd_up:
            pers = df_full['MACD_Persistence'].iloc[-2]
            msg = f"MACD 提前翻多！\n股票: <b>{symbol}</b>n現價: <b>{price:.2f}</b>n強度 {pers:.1f}（越大越持久）"
            key = f"{symbol}_MACD_UP_{df_full.index[-2].strftime('%Y%m%d%H%M')}"
            all_signals.append((symbol, msg, key))

        if macd_dn:
            pers = df_full['MACD_Persistence'].iloc[-2]
            msg = f"MACD 提前翻空！\n股票: <b>{symbol}</b>\n現價: <b>{price:.2f}</b>\n強度 {pers:.1f}"
            key = f"{symbol}_MACD_DN_{df_full.index[-2].strftime('%Y%m%d%H%M')}"
            all_signals.append((symbol, msg, key))

    # === 顯示 ===
    st.plotly_chart(fig, use_container_width=True)
    if recent_fig:
        st.plotly_chart(recent_fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("現價", f"{price:.2f}")
    col2.metric("支撐", f"{sup:.2f}", f"{price-sup:+.2f}")
    col3.metric("阻力", f"{res:.2f}", f"{res-price:+.2f}")

    if show_touches and levels:
        touches = []
        for lv in levels:
            touch = ((df_full['High'] >= lv*0.995) & (df_full['High'] <= lv*1.005)).sum() + \
                    ((df_full['Low'] >= lv*0.995) & (df_full['Low'] <= lv*1.005)).sum()
            if touch >= 2:
                touches.append(f"{lv:.2f} ({touch}次)")
        if touches:
            st.caption("強力支阻：" + " | ".join(touches))

    st.markdown("---")

# ==================== 警報觸發 ====================
for sym, msg, key in all_signals:
    if st.success(msg)
    if st.session_state.last_signal_keys.get(key) != key:
        st.session_state.last_signal_keys[key] = key
        st.session_state.signal_history.append({"time": datetime.now().strftime("%H:%M:%S"), "symbol": sym, "signal": msg})
        send_telegram_alert(msg)
        play_alert_sound()

# 歷史訊號
if st.session_state.signal_history:
    st.subheader("最近警報")
    for s in reversed(st.session_state.signal_history[-20:]):
        st.markdown(f"**{s['time']} | {s['symbol']}** → {s['signal'].replace('n', ' ')}")
