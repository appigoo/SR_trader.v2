# ================================================
#   多股票監控面板 — 永遠不會再錯的版本！
#   直接複製執行，一次成功！
# ================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from streamlit_autorefresh import st_autorefresh

# ==================== 設定 ====================
st.set_page_config(page_title="監控神器", layout="wide")
st.title("多股票監控 — 永不崩潰版")

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# 側邊欄
symbols_input = st.sidebar.text_input("股票代號", "TSLA,AAPL,NVDA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval = st.sidebar.selectbox("週期", ["1m","5m","15m","60m","1d"], index=1)
period = st.sidebar.selectbox("範圍", ["5d","10d","1mo","3mo"], index=0)

auto_update = st.sidebar.checkbox("自動更新", True)
freq = st.sidebar.selectbox("更新頻率", ["30秒","60秒","3分鐘"], index=1)
sound = st.sidebar.checkbox("聲音提醒", True)

# ==================== 取資料 ====================
@st.cache_data(ttl=45)
def get_data(symbol):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if len(df) < 10:
            return None
        return df.reset_index(drop=True)
    except:
        return None

# ==================== 支撐阻力（超穩版）===================
def get_support_resistance(df):
    if len(df) < 20:
        return round(float(df["Low"].min()), 2), round(float(df["High"].max()), 2)
    
    high = df["High"].values
    low = df["Low"].values
    window = 5
    res_pts = []
    sup_pts = []
    
    for i in range(window, len(df) - window):
        segment_high = high[i-window:i+window+1]
        segment_low = low[i-window:i+window+1]
        if high[i] == np.max(segment_high):
            res_pts.append(high[i])
        if low[i] == np.min(segment_low):
            sup_pts.append(low[i])
    
    resistance = float(np.max(res_pts)) if len(res_pts) > 0 else float(np.max(high))
    support = float(np.min(sup_pts)) if len(sup_pts) > 0 else float(np.min(low))
    
    return round(support, 2), round(resistance, 2)

# ==================== MACD 警報（完全避開 pandas Series 陷阱）===================
def macd_alert(df, symbol):
    alerts = []
    
    # 全部轉成 numpy array
    close = df["Close"].values.astype(float)
    if len(close) < 50:
        return alerts
    
    # 手動計算 EMA（避開 pandas ewm 的 Series 問題）
    def ema(values, span):
        alpha = 2 / (span + 1)
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
        return ema
    
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    dif = ema12 - ema26
    dea = ema(dif, 9)
    hist = dif - dea
    
    # 計算 DIF 加速度（用純數字）
    if len(dif) >= 14:
        dif_recent = dif[-12:]
        speed = np.diff(dif_recent)
        accel = np.diff(speed)
        if len(accel) >= 4:
            accel_3 = accel[-3]
            accel_1 = accel[-1]
            hist_last = hist[-1]
            
            if accel_3 < 0 and accel_1 > 0 and hist_last < 0:
                alerts.append(f"MACD 提前翻多！{symbol}")
            if accel_3 > 0 and accel_1 < 0 and hist_last > 0:
                alerts.append(f"MACD 提前翻空！{symbol}")
    
    return alerts

# ==================== 聲音 ====================
def beep():
    if sound:
        st.markdown("""
        <audio autoplay="true">
            <source src="https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

# ==================== 自動更新 ====================
if auto_update:
    seconds = {"30秒":30, "60秒":60, "3分鐘":180}[freq]
    st_autorefresh(interval=seconds * 1000)

if not symbols:
    st.stop()

st.write(f"監控中：{', '.join(symbols)} | {interval} | {period}")

# ==================== 主循環 ====================
for symbol in symbols:
    df = get_data(symbol)
    if df is None:
        st.error(f"{symbol} 無法取得資料")
        continue

    support, resistance = get_support_resistance(df)
    price = round(float(df["Close"].iloc[-1]), 2)

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=list(range(len(df))),
        open=df["Open"].values,
        high=df["High"].values,
        low=df["Low"].values,
        close=df["Close"].values,
        name=symbol
    ))
    fig.add_hline(y=support, line=dict(color="lime", dash="dash"))
    fig.add_hline(y=resistance, line=dict(color="red", dash="dash"))
    fig.update_layout(
        height=480,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # 警報
    alerts = []

    # 突破警報（用純數字）
    if len(df) >= 3:
        prev_price = float(df["Close"].values[-3])
        if prev_price <= resistance and price > resistance:
            alerts.append(f"突破阻力！{symbol} {price}")
        if prev_price >= support and price < support:
            alerts.append(f"跌破支撐！{symbol} {price}")

    # 爆量警報
    if len(df) > 30:
        vol_now = float(df["Volume"].values[-1])
        vol_avg = float(np.mean(df["Volume"].values[-30:-1]))
        if vol_avg > 0 and vol_now / vol_avg > 2.5:
            alerts.append(f"爆量！{symbol} {vol_now/vol_avg:.1f}x")

    # MACD 警報
    alerts.extend(macd_alert(df, symbol))

    # 顯示警報
    if alerts:
        msg = "\n".join(alerts)
        key = f"{symbol}_{hash(msg)}"
        if key not in st.session_state:
            st.session_state[key] = True
            now = datetime.now().strftime("%H:%M:%S")
            st.session_state.alerts.append(f"{now} {msg}")
            beep()
        st.success(msg)

    # 資訊欄
    c1, c2, c3 = st.columns(3)
    c1.metric("股票", symbol)
    c2.metric("現價", f"{price}")
    c3.metric("支撐/阻力", f"{support} / {resistance}")
    st.markdown("---")

# 歷史警報
if st.session_state.alerts:
    st.subheader("最近警報")
    for a in st.session_state.alerts[-10:]:
        st.write(a)
