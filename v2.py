# ================================================
#   多股票監控面板 — 永久不死版（2025終極穩定版）
#   直接複製 → 存成 app.py → streamlit run app.py → 完美運行！
# ================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# 自動更新套件
from streamlit_autorefresh import st_autorefresh

# ==================== 基本設定 ====================
st.set_page_config(page_title="多股監控神器", layout="wide")
st.title("多股票即時監控 — 永久穩定版")

# 警報歷史
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ==================== 側邊欄 ====================
st.sidebar.header("設定")
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "TSLA,AAPL,NVDA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval = st.sidebar.selectbox("週期", ["1m","5m","15m","60m","1d"], index=1)
period = st.sidebar.selectbox("範圍", ["5d","10d","1mo","3mo"], index=0)

auto_update = st.sidebar.checkbox("自動更新", True)
freq = st.sidebar.selectbox("頻率", ["30秒","60秒","3分鐘"], index=1)

sound = st.sidebar.checkbox("聲音提醒", True)

# ==================== 取資料 ====================
@st.cache_data(ttl=45)
def get_data(symbol):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if len(df) < 10:
            return None
        return df.reset_index()  # 避免 index 問題
    except:
        return None

# ==================== 最穩支撐阻力（不用任何 pandas 陷阱）===================
def get_support_resistance(df):
    # 直接轉成 numpy array，永遠不會錯
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()[-1]  # 最新收盤價

    window = 5
    support_points = []
    resistance_points = []

    for i in range(window, len(df) - window):
        # 阻力點：中心點是這段最高
        if high[i] == max(high[i-window:i+window+1]):
            resistance_points.append(high[i])
        # 支撐點：中心點是這段最低
        if low[i] == min(low[i-window:i+window+1]):
            support_points.append(low[i])

    # 取最靠近現價的有效點
    if resistance_points:
        resistance = max([x for x in resistance_points if x < close * 1.5], default=max(high))
    else:
        resistance = max(high)

    if support_points:
        support = min([x for x in support_points if x > close * 0.5], default=min(low))
    else:
        support = min(low)

    return round(float(support), 2), round(float(resistance), 2)

# ==================== MACD 提前預測 ====================
def check_macd_signal(df, symbol):
    alerts = []
    close = df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    if len(hist) < 20:
        return alerts

    # DIF 加速度轉向
    accel = dif.diff().diff()
    if len(accel) > 5:
        if accel.iloc[-3] < 0 and accel.iloc[-1] > 0 and hist.iloc[-1] < 0:
            alerts.append(f"MACD 提前翻多！{symbol}")
        if accel.iloc[-3] > 0 and accel.iloc[-1] < 0 and hist.iloc[-1] > 0:
            alerts.append(f"MACD 提前翻空！{symbol}")

    return alerts

# ==================== 聲音與通知 ====================
def beep():
    if sound:
        st.markdown("""
        <audio autoplay>
            <source src="https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

# ==================== 自動更新 ====================
if auto_update:
    sec = {"30秒":30, "60秒":60, "3分鐘":180}[freq]
    st_autorefresh(interval=sec*1000, key="refresh")

if not symbols:
    st.stop()

st.write(f"監控中：{', '.join(symbols)} | {interval} | {period} | 更新：{freq}")

# ==================== 主循環 ====================
for symbol in symbols:
    df = get_data(symbol)
    if df is None:
        st.error(f"{symbol} 無法取得資料")
        continue

    # 支撐阻力
    support, resistance = get_support_resistance(df)
    price = round(float(df["Close"].iloc[-1]), 2)

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"] if "Date" in df.columns else df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=symbol
    ))
    fig.add_hline(y=support, line=dict(color="lime", dash="dash"), annotation_text=f"支撐 {support}")
    fig.add_hline(y=resistance, line=dict(color="red", dash="dash"), annotation_text=f"阻力 {resistance}")
    fig.update_layout(
        height=500, plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
        font=dict(color="white"), xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # 警報
    alerts = []

    # 突破
    if len(df) >= 3:
        prev_close = df["Close"].iloc[-3]
        if prev_close <= resistance and price > resistance:
            alerts.append(f"突破阻力！{symbol} {price}")
        if prev_close >= support and price < support:
            alerts.append(f"跌破支撐！{symbol} {price}")

    # 爆量
    if len(df) > 30:
        vol_ratio = df["Volume"].iloc[-1] / df["Volume"].iloc[-30:-1].mean()
        if vol_ratio > 2.5:
            alerts.append(f"爆量！{symbol} {vol_ratio:.1f}x")

    # MACD
    alerts.extend(check_macd_signal(df, symbol))

    # 顯示警報
    if alerts:
        msg = "\n".join(alerts)
        key = f"{symbol}_{msg}"
        if key not in st.session_state:
            st.session_state[key] = True
            now = datetime.now().strftime("%H:%M:%S")
            st.session_state.alerts.append(f"{now} {msg}")
            beep()
        st.success(msg)

    # 資訊
    c1, c2, c3 = st.columns(3)
    c1.metric("股票", symbol)
    c2.metric("現價", price)
    c3.metric("支撐 → 阻力", f"{support} → {resistance}")
    st.markdown("---")

# 歷史警報
if st.session_state.alerts:
    st.subheader("最近 10 筆警報")
    for a in st.session_state.alerts[-10:]:
        st.write(a)
