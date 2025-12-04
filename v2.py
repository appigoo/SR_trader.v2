# ================================================
#   多股票即時監控面板 — 終極完整版
#   含：支撐阻力、MACD前瞻預測、四大警報、自動更新
# ================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from streamlit_autorefresh import st_autorefresh

# ==================== 頁面設定 ====================
st.set_page_config(page_title="頂級多股監控", layout="wide")
st.title("多股票支撐/阻力 + MACD 動能前瞻 監控面板")

# session_state（防重複警報）
for k in ["last_signal_keys", "signal_history"]:
    if k not in st.session_state:
        st.session_state[k] = {} if k == "last_signal_keys" else []

# ==================== 側邊欄 ====================
st.sidebar.header("設定")
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "TSLA,AAPL,NVDA,META")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval_opt = {"1分鐘":"1m","5分鐘":"5m","15分鐘":"15m","1小時":"60m","日線":"1d"}
interval_label = st.sidebar.selectbox("K線週期", list(interval_opt.keys()), index=1)
interval = interval_opt[interval_label]

period_opt = {"1天":"1d","5天":"5d","10天":"10d","1個月":"1mo","3個月":"3mo","1年":"1y"}
period_label = st.sidebar.selectbox("資料範圍", list(period_opt.keys()), index=1)
period = period_opt[period_label]

lookback = st.sidebar.slider("觀察根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒","60秒","3分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
buffer_pct = st.sidebar.slider("突破緩衝區(%)", 0.01, 1.0, 0.10, 0.01) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)
show_touches = st.sidebar.checkbox("顯示觸碰分析", True)

# 警報開關
st.sidebar.markdown("### 警報類型")
use_sr_alert   = st.sidebar.checkbox("支撐/阻力突破警報", True)
use_vol_filter = st.sidebar.checkbox("突破需放量確認", True)
use_vol_alert  = st.sidebar.checkbox("獨立成交量爆量警報", True)
vol_mult       = st.sidebar.slider("成交量倍數", 1.5, 6.0, 2.5, 0.1)

st.sidebar.markdown("#### 自訂價位警報")
custom_price_text = st.sidebar.text_area("SYMBOL,價位1,價位2...", "AAPL,180\nNVDA,900")

st.sidebar.markdown("#### 自訂成交量倍數")
custom_vol_text = st.sidebar.text_area("SYMBOL,倍數", "NVDA,4.0\nTSLA,3.5")

use_macd_alert = st.sidebar.checkbox("MACD 動能前瞻預測警報（極強）", True)

# ==================== 解析自訂 ====================
def parse_custom(text, is_price=True):
    d = {}
    for line in text.split("\n"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2: continue
        sym = parts[0].upper()
        try:
            vals = [float(p) for p in parts[1:]]
            if is_price:
                d.setdefault(sym, []).extend(vals)
            else:
                d[sym] = vals[0]
        except: pass
    return d

custom_prices = parse_custom(custom_price_text, True)
custom_vol_mult = parse_custom(custom_vol_text, False)

# ==================== Telegram & 聲音 ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID  = st.secrets["telegram"]["CHAT_ID"]
    tg_ok = True
except:
    tg_ok = False

def send_tg(msg): 
    if not tg_ok: return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id":CHAT_ID, "text":msg, "parse_mode":"HTML"}, timeout=8)
    except: pass

def play_sound():
    if sound_alert:
        st.markdown("<audio autoplay><source src='https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3' type='audio/mpeg'></audio>", 
                    unsafe_allow_html=True)

# ==================== 資料快取 ====================
@st.cache_data(ttl=55)
def get_data(symbol: str):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        df = df[~df.index.duplicated(keep='last')]
        return df.dropna(how='all')
    except:
        return None

# ==================== 永不崩潰版支撐阻力 ====================
def find_sr_levels(df_full: pd.DataFrame, window: int = 5):
    if len(df_full) < 15:
        return df_full["Low"].min(), df_full["High"].max(), []
    df = df_full.iloc[:-1].copy()
    h = pd.to_numeric(df["High"], errors='coerce').values
    l = pd.to_numeric(df["Low"],  errors='coerce').values
    res, sup = [], []
    for i in range(window, len(h)-window):
        if np.isnan(h[i]): continue
        if h[i] >= np.nanmax(h[i-window:i]) and h[i] >= np.nanmax(h[i+1:i+window+1]):
            res.append(float(h[i]))
        if np.isnan(l[i]): continue
        if l[i] <= np.nanmin(l[i-window:i]) and l[i] <= np.nanmin(l[i+1:i+window+1]):
            sup.append(float(l[i]))
    # 聚類
    def cluster(pts):
        if len(pts) < 2: return pts
        pts = sorted(pts)
        out, cur = [], [pts[0]]
        for p in pts[1:]:
            if abs(p-cur[-1])/cur[-1] < 0.006:
                cur.append(p)
            else:
                if len(cur) >= 2: out.append(np.mean(cur))
                cur = [p]
        if len(cur) >= 2: out.append(np.mean(cur))
        return out
    r_lv = cluster(res)
    s_lv = cluster(sup)
    cur_p = df_full["Close"].iloc[-1]
    support    = min(s_lv, default=df_full["Low"].min())
    resistance = max(r_lv, default=df_full["High"].max())
    return support, resistance, s_lv + r_lv

# ==================== MACD 動能前瞻（四大前兆） ====================
def macd_forecast(df_full: pd.DataFrame, symbol: str):
    if len(df_full) < 50: return []
    df = df_full.copy()
    close = df['Close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2

    dif = dif.values[-120:]
    hist = hist.values[-120:]

    signals = []
    now = pd.Timestamp.now().strftime("%H%M")

    # ① 加速度轉向
    if len(dif) > 12:
        speed = np.diff(dif[-12:])
        accel = np.diff(speed)
        if len(accel) >= 4:
            if accel[-3] < 0 and accel[-2] > 0 and hist[-1] < 0:
                sustain = max(8, min(45, int(abs(dif[-1])*12)))
                signals.append((symbol, f"MACD 提前翻多！\n<b>{symbol}</b>\nDIF加速度轉正\n預估多頭維持 <b>{sustain}</b> 根K", f"MAC_UP_{now}"))
            if accel[-3] > 0 and accel[-2] < 0 and hist[-1] > 0:
                sustain = max(8, min(45, int(abs(dif[-1])*12)))
                signals.append((symbol, f"MACD 提前翻空！\n<b>{symbol}</b>\nDIF加速度轉負\n預估空頭維持 <b>{sustain}</b> 根K", f"MAC_DN_{now}"))

    # ② 柱子連續縮短
    if len(hist) >= 7:
        r = hist[-7:]
        if all(r[-6:] > 0) and np.all(np.diff(r[-5:]) < 0):
            signals.append((symbol, f"MACD 多頭衰竭！紅柱連縮\n極大概率翻綠", f"RED_SHRINK_{now}"))
        if all(r[-6:] < 0) and np.all(np.diff(np.abs(r[-5:])) < 0):
            signals.append((symbol, f"MACD 空頭衰竭！綠柱連縮\n極大概率翻紅", f"GREEN_SHRINK_{now}"))

    # ③ 背離（簡化但極有效）
    if len(hist) >= 20:
        p_low = np.argmin(df['Low'].values[-15:])
        d_low = np.argmin(dif[-15:])
        if p_low > d_low and hist[-1] < 0:
            signals.append((symbol, f"強力底背離！\n{symbol} 即將翻紅", f"BULL_DIV_{now}"))
        p_high = np.argmax(df['High'].values[-15:])
        d_high = np.argmax(dif[-15:])
        if p_high > d_high and hist[-1] > 0:
            signals.append((symbol, f"強力頂背離！\n{symbol} 即將翻綠", f"BEAR_DIV_{now}"))

    return signals

# ==================== 其他警報函數（保持簡潔穩定） ====================
def sr_breakout(df, sup, res, symbol):
    if len(df) < 5: return None
    try:
        c1, c2, c3 = df["Close"].iloc[-3:]
        vol_ok = df["Volume"].iloc[-1] > df["Volume"].iloc[-lookback:-1].mean() * 1.5 or not use_vol_filter
        buf = res * buffer_pct
        if c1 <= res - buf and c2 <= res - buf and c3 > res and vol_ok:
            return (symbol, f"突破阻力！\n<b>{symbol}</b> {c3:.2f} > {res:.2f}", "SR_UP")
        if c1 >= sup + buf and c2 >= sup + buf and c3 < sup and vol_ok:
            return (symbol, f"跌破支撐！\n<b>{symbol}</b> {c3:.2f} < {sup:.2f}", "SR_DN")
    except: pass
    return None

def volume_spike(df, symbol):
    if len(df) < lookback: return None
    vol = df["Volume"].iloc[-1]
    avg = df["Volume"].iloc[-lookback-1:-1].mean()
    if avg <= 0: return None
    ratio = vol / avg
    mult = custom_vol_mult.get(symbol, vol_mult)
    if ratio > mult:
        return (symbol, f"成交量激增！\n<b>{symbol}</b> {ratio:.1f}x", f"VOL_{ratio:.1f}")

# ==================== 主圖表繪製 ====================
def draw_chart(df_full: pd.DataFrame, symbol: str, sup, res, levels, custom_lvls):
    df = df_full.copy()
    # EMA
    for p in [5,10,20,40,60]:
        df[f'EMA_'+str(p)] = df['Close'].ewm(span=p, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name=symbol, increasing_line_color='lime', decreasing_line_color='red'))

    colors = ['cyan','magenta','yellow','orange','white']
    for i,p in enumerate([5,10,20,40,60]):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{p}'], name=f'EMA{p}', line=dict(color=colors[i], width=1.5)))

    fig.add_hline(y=sup, line_dash="dash", line_color="green", annotation_text=f"支撐 {sup:.2f}")
    fig.add_hline(y=res, line_dash="dash", line_color="red", annotation_text=f"阻力 {res:.2f}")
    for lv in levels + custom_lvls:
        fig.add_hline(y=lv, line_dash="dot", line_color="blue", opacity=0.6)

    fig.update_layout(height=550, plot_bgcolor='black', paper_bgcolor='black',
                      font_color='white', xaxis_rangeslider_visible=False, hovermode='x unified')
    return fig

# ==================== 自動更新 ====================
if auto_update:
    st_autorefresh(interval={"30秒":30,"60秒":60,"3分鐘":180}[update_freq]*1000, key="refresh")

if not symbols:
    st.stop()

st.header(f"即時監控：{', '.join(symbols)}  |  {interval_label}  |  {period_label}")

# ==================== 主循環 ====================
all_signals = []
progress = st.progress(0)

for idx, sym in enumerate(symbols):
    progress.progress((idx+1)/len(symbols))
    df = get_data(sym)
    if df is None or len(df) < 20:
        st.error(f"{sym} 無資料")
        continue

    sup, res, levels = find_sr_levels(df)
    custom_lvls = custom_prices.get(sym, [])
    fig = draw_chart(df, sym, sup, res, levels, custom_lvls)

    # 警報收集
    if use_sr_alert:
        sig = sr_breakout(df, sup, res, sym)
        if sig: all_signals.append(sig)

    if use_vol_alert:
        sig = volume_spike(df, sym)
        if sig: all_signals.append(sig)

    if use_macd_alert:
        all_signals.extend(macd_forecast(df, sym))

    # 顯示
    st.subheader(f"{sym}  現價 {df['Close'].iloc[-1]:.2f}")
    st.plotly_chart(fig, use_container_width=True)

    sym_sigs = [s for s in all_signals if s[0]==sym]
    for _, msg, key in sym_sigs:
        if key and st.session_state.last_signal_keys.get(key) != key:
            st.session_state.last_signal_keys[key] = key
            st.session_state.signal_history.append({"time":datetime.now().strftime("%H:%M:%S"), "msg":msg})
            send_tg(msg)
            play_sound()
        st.success(msg)

    col1, col2, col3 = st.columns(3)
    col1.metric("現價", f"{df['Close'].iloc[-1]:.2f}")
    col2.metric("支撐", f"{sup:.2f}")
    col3.metric("阻力", f"{res:.2f}")
    st.markdown("---")

# 歷史警報
if st.session_state.signal_history:
    st.subheader("最近警報")
    for h in reversed(st.session_state.signal_history[-15:]):
        st.markdown(f"**{h['time']}** {h['msg']}")
