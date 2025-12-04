# ================================================
#   多股票監控面板 — 永遠不死的宇宙最穩版
#   直接複製執行，保證一次成功！
# ================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# 自動更新
from streamlit_autorefresh import st_autorefresh

# ==================== 設定 ====================
st.set_page_config(page_title="監控神器", layout="wide")
st.title("多股票即時監控 — 永不崩潰版")

# 警報歷史
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ==================== 側邊欄 ====================
symbols_input = st.sidebar.text_input("股票代號", "TSLA,AAPL,NVDA")
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
        return df.reset_index(drop=True)
    except:
        return None

# ==================== 支撐阻力（永不錯版）===================
def get_sr(df):
    if len(df) < 20:
        return round(df["Low"].min(), 2), round(df["High"].max(), 2)
    
    high = df["High"].values
    low = df["Low"].values
    price = df["Close"].values[-1]
    
    window = 5
    res_pts = []
    sup_pts = []
    
    for i in range(window, len(df) - window):
        # 阻力點
        if high[i] == np.max(high[i-window:i+window+1]):
            res_pts.append(high[i])
        # 支撐點
        if low[i] == np.min(low[i-window:i+window+1]):
            sup_pts.append(low[i])
    
    # 安全取值
    resistance = max(res_pts) if res_pts else high.max()
    support = min(sup_pts) if sup_pts else low.min()
    
    return round(float(support), 2), round(float(resistance), 2)

# ==================== MACD 警報 ====================
def macd_alert(df, symbol):
    alerts = []
    close = df["Close"]
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    
    if len(hist) < 20:
        return alerts
    
    accel = dif.diff().diff()
    if len(accel) > 5:
        if accel.iloc[-3] < 0 and accel.iloc[-1] > 0 and hist.iloc[-1] < 0:
            alerts.append(f"MACD 提前翻多！{symbol}")
        if accel.iloc[-3] > 0 and accel.iloc[-1] < 0 and hist.iloc[-1] > 0:
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

for symbol in symbols:
    df = get_data(symbol)
    if df is None:
        st.error(f"{symbol} 無資料")
        continue

    support, resistance = get_sr(df)
    price = round(float(df["Close"].iloc[-1]), 2)

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=list(range(len(df))),
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
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

    # 突破
    if len(df) >= 3:
        prev_price = float(df["Close"].iloc[-3])
        if prev_price <= resistance and price > resistance:
            alerts.append(f"突破阻力！{symbol} {price}")
        if prev_price >= support and price < support:
            alerts.append(f"跌破支撐！{symbol} {price}")

    # 爆量
    if len(df) > 30:
        vol_now = float(df["Volume"].iloc[-1])
        vol_avg = float(df["Volume"].iloc[-30:-1].mean())
        if vol_avg > 0 and vol_now / vol_avg > 2.5:
            alerts.append(f"爆量！{symbol} {vol_now/vol_avg:.1f}x")

    # MACD
    alerts.extend(macd_alert(df, symbol))

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
    c3.metric("支撐/阻力", f"{support} / {resistance}")
    st.markdown("---")

# 歷史
if st.session_state.alerts:
    st.subheader("最近警報")
    for a in st.session_state.alerts[-10:]:
        st.write(a)
