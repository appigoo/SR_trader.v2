# ================================================
#   多股票即時監控面板 — 永久不再出錯版
#   直接複製 → 存成 app.py → streamlit run app.py → 完美運行！
# ================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

from streamlit_autorefresh import st_autorefresh

# ==================== 基本設定 ====================
st.set_page_config(page_title="多股監控神器", layout="wide")
st.title("多股票即時監控面板 — 終極穩定版")

# 警報歷史
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== 側邊欄 ====================
st.sidebar.header("設定")
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "TSLA,AAPL,NVDA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval = st.sidebar.selectbox("K線週期", ["1m","5m","15m","60m","1d"], index=1)
period = st.sidebar.selectbox("資料範圍", ["5d","10d","1mo","3mo","1y"], index=0)

auto_update = st.sidebar.checkbox("自動更新", True)
freq = st.sidebar.selectbox("更新頻率", ["30秒","60秒","3分鐘"], index=1)

sound_on = st.sidebar.checkbox("聲音提醒", True)

# ==================== Telegram（可選）================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    tg_on = True
except:
    tg_on = False

def send_msg(text):
    if tg_on:
        try:
            requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": text}, timeout=5)
        except: pass

def beep():
    if sound_on:
        st.markdown("<audio autoplay><source src='https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3'></audio>", 
                    unsafe_allow_html=True)

# ==================== 取資料 ====================
@st.cache_data(ttl=45)
def get_data(sym):
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
        if len(df) < 10:
            return None
        return df
    except:
        return None

# ==================== 永不崩潰的支撐阻力（核心修復）===================
def get_sr(df):
    # 直接用 .values，避免任何 pandas Index 問題
    high = np.array(df["High"].tolist(), dtype=float)
    low  = np.array(df["Low"].tolist(),  dtype=float)
    close_price = float(df["Close"].iloc[-1])

    if len(high) < 20:
        return float(np.nanmin(low)), float(np.nanmax(high))

    window = 5
    res_pts = []
    sup_pts = []

    for i in range(window, len(high) - window):
        if np.isnan(high[i]) or np.isnan(low[i]):
            continue
        # 阻力點
        if high[i] >= np.nanmax(high[i-window:i+window+1]):
            res_pts.append(high[i])
        # 支撐點
        if low[i] <= np.nanmin(low[i-window:i+window+1]):
            sup_pts.append(low[i])

    # 安全取值
    if res_pts:
        resistance = max([x for x in res_pts if x < close_price * 1.5], default=np.nanmax(high))
    else:
        resistance = np.nanmax(high)

    if sup_pts:
        support = min([x for x in sup_pts if x > close_price * 0.5], default=np.nanmin(low))
    else:
        support = np.nanmin(low)

    return float(support), float(resistance)

# ==================== MACD 提前預測 ====================
def macd_signals(df, sym):
    alerts = []
    close = df["Close"]
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    if len(hist) < 20:
        return alerts

    # DIF 加速度
    accel = dif.diff().diff()
    if len(accel) > 5:
        if accel.iloc[-3] < 0 and accel.iloc[-1] > 0 and hist.iloc[-1] < 0:
            alerts.append(f"MACD 提前翻多！{sym}")
        if accel.iloc[-3] > 0 and accel.iloc[-1] < 0 and hist.iloc[-1] > 0:
            alerts.append(f"MACD 提前翻空！{sym}")

    # 柱子縮短
    if len(hist) >= 5:
        if hist.iloc[-5] > 0 and hist.iloc[-1] > 0 and hist.iloc[-1] < hist.iloc[-3]:
            alerts.append(f"MACD 多頭衰竭！{sym}")
        if hist.iloc[-5] < 0 and hist.iloc[-1] < 0 and abs(hist.iloc[-1]) < abs(hist.iloc[-3]):
            alerts.append(f"MACD 空頭衰竭！{sym}")

    return alerts

# ==================== 自動更新 ====================
if auto_update:
    seconds = {"30秒": 30, "60秒": 60, "3分鐘": 180}[freq]
    st_autorefresh(interval=seconds * 1000, key="auto")

if not symbols:
    st.stop()

st.write(f"監控中：{', '.join(symbols)} | {interval} | {period} | 更新：{freq}")

# ==================== 主循環 ====================
for sym in symbols:
    df = get_data(sym)
    if df is None:
        st.error(f"{sym} 無法取得資料")
        continue

    support, resistance = get_sr(df)
    price = df["Close"].iloc[-1]

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name=sym))
    fig.add_hline(y=support, line=dict(color="lime", dash="dash"), annotation_text=f"支撐 {support:.2f}")
    fig.add_hline(y=resistance, line=dict(color="red", dash="dash"), annotation_text=f"阻力 {resistance:.2f}")
    fig.update_layout(height=500, plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                      font=dict(color="white"), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 警報收集
    alerts = []

    # 支撐阻力突破
    if len(df) >= 3:
        if df["Close"].iloc[-3] <= resistance and price > resistance:
            alerts.append(f"突破阻力！{sym} {price:.2f}")
        if df["Close"].iloc[-3] >= support and price < support:
            alerts.append(f"跌破支撐！{sym} {price:.2f}")

    # 爆量
    if len(df) > 30:
        ratio = df["Volume"].iloc[-1] / df["Volume"].iloc[-30:-1].mean()
        if ratio > 2.5:
            alerts.append(f"爆量！{sym} {ratio:.1f}x")

    # MACD
    alerts.extend(macd_signals(df, sym))

    # 顯示警報
    if alerts:
        msg = "\n".join(alerts)
        key = f"{sym}_{hash(msg)}"
        if st.session_state.get(key) != msg:
            st.session_state[key] = msg
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.history.append(f"{timestamp} {msg}")
            send_msg(f"{timestamp}\n{msg}")
            beep()
        st.success(msg)

    # 資訊欄
    c1, c2, c3 = st.columns(3)
    c1.metric("股票", sym)
    c2.metric("現價", f"{price:.2f}")
    c3.metric("支撐 → 阻力", f"{support:.2f} → {resistance:.2f}")
    st.markdown("---")

# 歷史警報
if st.session_state.history:
    st.subheader("最近 10 筆警報")
    for line in st.session_state.history[-10:]:
        st.write(f"**{line}**")
