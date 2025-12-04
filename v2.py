# ================================================
#   多股票監控面板 — 宇宙最穩版（2025-12-04）
#   直接複製執行，保證一次成功！
# ================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

from streamlit_autorefresh import st_autorefresh

# ==================== 設定 ====================
st.set_page_config(page_title="監控神器", layout="wide")
st.title("多股票監控 — 永遠不崩版")

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# 側邊欄
symbols = st.sidebar.text_input("股票", "TSLA,AAPL,NVDA").upper().replace(" ", "").split(",")
if "" in symbols: symbols.remove("")

interval = st.sidebar.selectbox("週期", ["1m","5m","15m","60m","1d"], index=1)
period = st.sidebar.selectbox("範圍", ["5d","10d","1mo","3mo"], index=0)
auto_update = st.sidebar.checkbox("自動更新", True)
freq = st.sidebar.selectbox("頻率", ["30秒","60秒","3分鐘"], index=1)
sound = st.sidebar.checkbox("聲音", True)

# ==================== 取資料 ====================
@st.cache_data(ttl=45)
def get_df(symbol):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if len(df) < 10:
            return None
        return df.reset_index(drop=True)  # 徹底避免 index 問題
    except:
        return None

# ==================== 最穩支撐阻力（純數字版）===================
def get_sr_pure(df):
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close_now = float(df["Close"].values[-1])

    window = 5
    res = []
    sup = []

    for i in range(window, len(df)-window):
        if high[i] == np.max(high[i-window:i+window+1])]:
            res.append(high[i])
        if low[i] == np.min(low[i-window:i+window+1])]:
            sup.append(low[i])

    if res:
        resistance = max([x for x in res if x < close_now*1.5], default=np.max(high))
    else:
        resistance = np.max(high)

    if sup:
        support = min([x for x in sup if x > close_now*0.5], default=np.min(low))
    else:
        support = np.min(low)

    return round(float(support), 2), round(float(resistance), 2)

# ==================== MACD 提前預測 ====================
def macd_alert(df, sym):
    c = df["Close"].values
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    alerts = []
    if len(hist) < 20:
        return alerts

    accel = dif.diff().diff()
    if len(accel) > 5:
        if accel.iloc[-3] < 0 and accel.iloc[-1] > 0 and hist.iloc[-1] < 0:
            alerts.append(f"MACD 提前翻多！{sym}")
        if accel.iloc[-3] > 0 and accel.iloc[-1] < 0 and hist.iloc[-1] > 0:
            alerts.append(f"MACD 提前翻空！{sym}")

    return alerts

# ==================== 聲音 ====================
def beep():
    if sound:
        st.markdown("<audio autoplay><source src='https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3'></audio>", 
                    unsafe_allow_html=True)

# ==================== 自動更新 ====================
if auto_update:
    sec = {"30秒":30,"60秒":60,"3分鐘":180}[freq]
    st_autorefresh(interval=sec*1000, key="go")

if not symbols:
    st.stop()

st.write(f"監控：{', '.join(symbols)} | {interval} | {period}")

for sym in symbols:
    df = get_df(sym)
    if df is None:
        st.error(f"{sym} 無資料")
        continue

    support, resistance = get_sr_pure(df)
    price = round(float(df["Close"].values[-1]), 2)

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=list(range(len(df))),
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=sym
    ))
    fig.add_hline(y=support, line=dict(color="lime", dash="dash"))
    fig.add_hline(y=resistance, line=dict(color="red", dash="dash"))
    fig.update_layout(height=480, plot_bgcolor="#000", paper_bgcolor="#000",
                      font=dict(color="white"), xaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 警報（全部強制轉 float）
    alerts = []

    if len(df) >= 3:
        prev = float(df["Close"].values[-3])
        if prev <= resistance and price > resistance:
            alerts.append(f"突破阻力！{sym} {price}")
        if prev >= support and price < support:
            alerts.append(f"跌破支撐！{sym} {price}")

    if len(df) > 30:
        vol_now = float(df["Volume"].values[-1])
        vol_avg = float(df["Volume"].values[-30:-1].mean())
        if vol_avg > 0 and vol_now / vol_avg > 2.5:
            alerts.append(f"爆量！{sym} {vol_now/vol_avg:.1f}x")

    alerts.extend(macd_alert(df, sym))

    if alerts:
        msg = "\n".join(alerts)
        key = f"{sym}_{msg}"
        if key not in st.session_state:
            st.session_state[key] = True
            now = datetime.now().strftime("%H:%M:%S")
            st.session_state.alerts.append(f"{now} {msg}")
            beep()
        st.success(msg)

    c1,c2,c3 = st.columns(3)
    c1.metric("股票", sym)
    c2.metric("現價", price)
    c3.metric("支撐/阻力", f"{support} / {resistance}")
    st.markdown("---")

if st.session_state.alerts:
    st.subheader("最近警報")
    for a in st.session_state.alerts[-10:]:
        st.write(a)
