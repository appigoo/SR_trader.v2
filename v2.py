# app.py - 終極版：多股票即時監控 + MACD 提前預測 + Telegram 警報（已修復所有語法錯誤）
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from typing import List, Dict, Optional
from streamlit_autorefresh import st_autorefresh

# ==================== 初始化 ====================
st.set_page_config(page_title="多股票即時監控面板", layout="wide")
st.title("多股票支撐/阻力 + MACD 提前預測監控面板")

if "last_signal_keys" not in st.session_state:
    st.session_state.last_signal_keys = {}
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

# ==================== 側邊欄設定 ====================
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "TSLA,META,NVDA,AAPL")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval_options = {"1分鐘": "1m", "5分鐘": "5m", "15分鐘": "15m", "1小時": "60m", "日線": "1d"}
interval_label = st.sidebar.selectbox("K線週期", options=list(interval_options.keys()), index=1)
interval = interval_options[interval_label]

period_options = {"1天": "1d", "5天": "5d", "10天": "10d", "1個月": "1mo", "3個月": "3mo", "1年": "1y"}
period_label = st.sidebar.selectbox("資料範圍", options=list(period_options.keys()), index=1)
period = period_options[period_label]

lookback = st.sidebar.slider("觀察根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
buffer_pct = st.sidebar.slider("突破緩衝區 (%)", 0.01, 1.0, 0.1, 0.01) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)
show_touches = st.sidebar.checkbox("顯示價位觸碰分析", True)

# 警報開關
use_auto_sr_alerts = st.sidebar.checkbox("啟用 S/R 突破警報", True)
use_volume_alert = st.sidebar.checkbox("啟用成交量激增警報", True)
volume_alert_multiplier = st.sidebar.slider("成交量倍數", 1.5, 5.0, 2.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.caption(f"週期：{interval_label} | 範圍：{period_label}")

# ==================== 自訂價位解析 ====================
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

custom_alert_levels = parse_custom_alerts(
    st.sidebar.text_area("自訂警報價位 (SYMBOL,價位1,價位2...)", "AAPL,180\nNVDA,900")
)

# ==================== Telegram ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except:
    telegram_ready = False

def send_telegram(msg: str):
    if not telegram_ready: return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except:
        pass

def play_sound():
    if sound_alert:
        st.markdown("<audio autoplay><source src='https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3'></audio>", unsafe_allow_html=True)

# ==================== 資料快取 ====================
@st.cache_data(ttl=50)
def get_data(symbol: str):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or df["Close"].isna().all():
            return None
        df = df[~df.index.duplicated(keep='last')]
        return df.dropna(how='all')
    except:
        return None

# ==================== S/R 計算 ====================
def get_sr_levels(df_full: pd.DataFrame):
    df = df_full.iloc[:-1]
    if len(df) < 15:
        return df_full["Low"].min(), df_full["High"].max(), []
    high, low = df["High"], df["Low"]
    res_pts, sup_pts = [], []
    window = 5
    for i in range(window, len(df) - window):
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            res_pts.append(high.iloc[i])
        if low.iloc[i] == low.iloc[i-window:i+window+1].min():
            sup_pts.append(low.iloc[i])
    def cluster(pts):
        if not pts: return []
        pts = sorted(pts)
        clusters = []
        cur = [pts[0]]
        for p in pts[1:]:
            if abs(p - cur[-1]) / cur[-1] < 0.006:
                cur.append(p)
            else:
                if len(cur) >= 2:
                    clusters.append(np.mean(cur))
                cur = [p]
        if len(cur) >= 2:
            clusters.append(np.mean(cur))
        return clusters
    res_lv = cluster(res_pts)
    sup_lv = cluster(sup_pts)
    all_lv = list(set(res_lv + sup_lv))
    support = min(sup_lv, default=df_full["Low"].min(), key=lambda x: abs(x - df_full["Close"].iloc[-1])) if sup_lv else df_full["Low"].min()
    resistance = max(res_lv, default=df_full["High"].max(), key=lambda x: abs(x - df_full["Close"].iloc[-1])) if res_lv else df_full["High"].max()
    return support, resistance, all_lv

# ==================== 核心處理函數（含 MACD 提前預測） ====================
def process_symbol(symbol: str, custom_levels: List[float]):
    df_full = get_data(symbol)
    if df_full is None or len(df_full) < 40:
        return None, None, None, None, [], None, None, False, False

    # === MACD 計算 ===
    exp12 = df_full['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df_full['Close'].ewm(span=26, adjust=False).mean()
    dif = exp12 - exp26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    # === 四大提前訊號 ===
    dif_accel = dif.diff().diff()
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

    early_up   = accel_up | hist_shrink_up | bull_div | ((hist < 0) & slope_flat_dn)
    early_down = accel_dn | hist_shrink_dn | bear_div | ((hist > 0) & slope_flat_up)

    # 趨勢強度
    atr14 = (df_full['High'] - df_full['Low']).rolling(14).mean()
    persistence = (dif.abs() / atr14).iloc[-2]

    # S/R
    support, resistance, all_levels = get_sr_levels(df_full)

    # === 圖表 ===
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_full.index, open=df_full['Open'], high=df_full['High'],
                                 low=df_full['Low'], close=df_full['Close'],
                                 increasing_line_color='lime', decreasing_line_color='red', name="K線"))
    colors = ['cyan','blue','orange','magenta','yellow']
    for i,p in enumerate([5,10,20,40,60]):
        df_full[f'EMA{p}'] = df_full['Close'].ewm(span=p, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df_full.index, y=df_full[f'EMA{p}'], name=f'EMA{p}', line=dict(color=colors[i], width=1)))
    fig.add_hline(y=support, line_dash="dash", line_color="green", annotation_text=f"支撐 {support:.2f}")
    fig.add_hline(y=resistance, line_dash="dash", line_color="red", annotation_text=f"阻力 {resistance:.2f}")
    fig.update_layout(height=500, plot_bgcolor='black', paper_bgcolor='black', font_color='white', title=symbol)

    # === 最近K線圖（含MACD提前三角形） ===
    recent_fig = None
    if len(df_full) >= 20:
        recent = df_full.tail(20).copy()
        recent['OBV'] = (np.sign(recent['Close'].diff()) * recent['Volume']).cumsum().fillna(0)
        from plotly.subplots import make_subplots
        recent_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                   subplot_titles=('價格', 'OBV', 'MACD 提前訊號'),
                                   row_heights=[0.6,0.25,0.15])
        recent_fig.add_trace(go.Candlestick(x=recent.index, open=recent['Open'], high=recent['High'],
                                            low=recent['Low'], close=recent['Close'],
                                            increasing_line_color='lime', decreasing_line_color='red'), row=1, col=1)
        for i,p in enumerate([5,10,20,40,60]):
            recent[f'EMA{p}'] = recent['Close'].ewm(span=p, adjust=False).mean()
            recent_fig.add_trace(go.Scatter(x=recent.index, y=recent[f'EMA{p}'], name=f'EMA{p}',
                                           line=dict(color=colors[i], width=1.5)), row=1, col=1)
        recent_fig.add_hline(y=recent["Close"].iloc[-1], line_color="purple", line_width=2,
                             annotation_text=f"現價 {recent['Close'].iloc[-1]:.2f}", row=1, col=1)
        recent_fig.add_trace(go.Bar(x=recent.index, y=recent['Volume'], name="量", marker_color=np.where(recent['Close']>=recent['Open'], 'lime','red')), row=1, col=1)
        recent_fig.add_trace(go.Scatter(x=recent.index, y=recent['OBV'], name="OBV", line=dict(color='yellow')), row=2, col=1)

        up_idx = recent[early_up].index
        dn_idx = recent[early_down].index
        recent_fig.add_trace(go.Scatter(x=up_idx, y=recent.loc[up_idx,'Low']*0.993, mode='markers',
                                        marker=dict(symbol='triangle-up', size=20, color='lime', line=dict(color='white',width=2)),
                                        name='MACD即將上升'), row=3, col=1)
        recent_fig.add_trace(go.Scatter(x=dn_idx, y=recent.loc[dn_idx,'High']*1.007, mode='markers',
                                        marker=dict(symbol='triangle-down', size=20, color='red', line=dict(color='white',width=2)),
                                        name='MACD即將下跌'), row=3, col=1)
        recent_fig.update_layout(title=f"{symbol} – MACD提前預測 (強度 {persistence:.1f})", height=720,
                                plot_bgcolor='black', paper_bgcolor='black', font_color='white')

    price = df_full["Close"].iloc[-1]
    macd_up_alert = early_up.iloc[-2] if len(early_up) > 1 else False
    macd_dn_alert = early_down.iloc[-2] if len(early_down) > 1 else False

    return fig, price, support, resistance, all_levels, df_full, recent_fig, macd_up_alert, macd_dn_alert, persistence

# ==================== 自動更新 ====================
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180}
if auto_update:
    st_autorefresh(interval=interval_map[update_freq]*1000, key="auto")

if not symbols:
    st.stop()

st.header(f"監控：{', '.join(symbols)} | {interval_label} | {period_label}")

all_signals = []

# ==================== 主迴圈 ====================
for symbol in symbols:
    custom_lv = custom_alert_levels.get(symbol, [])
    result = process_symbol(symbol, custom_lv)
    if result[0] is None:
        st.error(f"{symbol} 無資料")
        continue

    fig, price, sup, res, levels, df_full, recent_fig, macd_up, macd_dn, pers = result

    # === 警報觸發 ===
    if macd_up:
        msg = f"MACD 提前翻多！\n股票: <b>{symbol}</b>\n現價: <b>{price:.2f}</b>\n趨勢強度: {pers:.1f}"
        key = f"{symbol}_MACD_UP_{df_full.index[-2].strftime('%Y%m%d%H%M')}"
        all_signals.append((symbol, msg, key))

    if macd_dn:
        msg = f"MACD 提前翻空！\n股票: <b>{symbol}</b>\n現價: <b>{price:.2f}</b>\n趨勢強度: {pers:.1f}"
        key = f"{symbol}_MACD_DN_{df_full.index[-2].strftime('%Y%m%d%H%M')}"
        all_signals.append((symbol, msg, key))

    # 其他警報（S/R、成交量）可自行加入，這裡先專注 MACD

    st.plotly_chart(fig, use_container_width=True)
    if recent_fig:
        st.plotly_chart(recent_fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("現價", f"{price:.2f}")
    c2.metric("支撐", f"{sup:.2f}", f"{price-sup:+.2f}")
    c3.metric("阻力", f"{res:.2f}", f"{res-price:+.2f}")
    st.markdown("---")

# ==================== 發送警報 ====================
for sym, msg, key in all_signals:
    if st.session_state.last_signal_keys.get(key) != key:
        st.session_state.last_signal_keys[key] = key
        st.session_state.signal_history.append({"time": datetime.now().strftime("%H:%M:%S"), "symbol": sym, "msg": msg})
        send_telegram(msg)
        play_sound()
        st.success(msg)

if st.session_state.signal_history:
    st.subheader("最近20筆警報")
    for h in reversed(st.session_state.signal_history[-20:]):
        st.markdown(f"**{h['time']} | {h['symbol']}** → {h['msg']}")
