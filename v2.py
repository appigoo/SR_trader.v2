import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="終極監控", layout="wide")
st.title("多股票監控面板 — 永不崩潰版")

if "history" not in st.session_state:
    st.session_state.history = []

# 側邊欄
symbols = st.sidebar.text_input("股票代號", "TSLA,AAPL,NVDA").upper().replace(" ","").split(",")
interval = st.sidebar.selectbox("週期", ["1m","5m","15m","60m","1d"], index=1)
period = st.sidebar.selectbox("範圍", ["5d","10d","1mo","3mo"], index=0)
auto_update = st.sidebar.checkbox("自動更新", True)
update_freq = st.sidebar.selectbox("頻率", ["30秒","60秒","3分鐘"], index=1)
sound = st.sidebar.checkbox("聲音", True)

# Telegram
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    tg = True
except:
    tg = False

def send(msg):
    if tg:
        try:
            requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": msg}, timeout=5)
        except: pass

def beep():
    if sound:
        st.markdown("<audio autoplay><source src='https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3'></audio>", 
                    unsafe_allow_html=True)

# 取資料
@st.cache_data(ttl=50)
def get_data(sym):
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
        return df if len(df) >= 10 else None
    except:
        return None

# 永不崩潰的支撐阻力
def get_sr(df):
    if len(df) < 20:
        return float(df["Low"].min()), float(df["High"].max())
    high = pd.to_numeric(df["High"], errors='coerce').values
    low = pd.to_numeric(df["Low"], errors='coerce').values
    close_price = float(df["Close"].iloc[-1])
    window = 5
    res_pts, sup_pts = [], []
    for i in range(window, len(df)-window):
        if np.isnan(high[i]) or np.isnan(low[i]): continue
        if high[i] >= np.nanmax(high[i-window:i+window+1]):
            res_pts.append(high[i])
        if low[i] <= np.nanmin(low[i-window:i+window+1]):
            sup_pts.append(low[i])
    try:
        resistance = max(x for x in res_pts if x < close_price*1.3) if res_pts else float(np.nanmax(high))
    except:
        resistance = float(np.nanmax(high))
    try:
        support = min(x for x in sup_pts if x > close_price*0.7) if sup_pts else float(np.nanmin(low))
    except:
        support = float(np.nanmin(low))
    return float(support), float(resistance)

# MACD 提前預測
def macd_alert(df, sym):
    alerts = []
    close = df["Close"]
    dif = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    dea = dif.ewm(span=9).mean()
    hist = dif - dea
    if len(hist) < 20: return alerts
    accel = dif.diff().diff()
    if len(accel) > 5:
        if accel.iloc[-3] < 0 and accel.iloc[-1] > 0 and hist.iloc[-1] < 0:
            alerts.append(f"MACD 提前翻多！{sym}")
        if accel.iloc[-3] > 0 and accel.iloc[-1] < 0 and hist.iloc[-1] >0:
            alerts.append(f"MACD 提前翻空！{sym}")
    return alerts

# 自動更新
if auto_update:
    st_autorefresh(interval={"30秒":30,"60秒":60,"3分鐘":180}[update_freq]*1000)

if not symbols or symbols == ['']:
    st.stop()

st.write(f"監控：{', '.join(symbols)} | {interval} | {period}")

for sym in symbols:
    df = get_data(sym)
    if df is None:
        st.error(f"{sym} 無資料")
        continue

    support, resistance = get_sr(df)
    price = df["Close"].iloc[-1]

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name=sym))
    fig.add_hline(y=support, line=dict(color="green", dash="dash"))
    fig.add_hline(y=resistance, line=dict(color="red", dash="dash"))
    fig.update_layout(height=480, plot_bgcolor="black", paper_bgcolor="black",
                      font=dict(color="white"), xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # 警報
    alerts = []
    # 突破
    if len(df) >= 3:
        if df["Close"].iloc[-3] <= resistance and price > resistance:
            alerts.append(f"突破阻力！{sym} {price:.2f}")
        if df["Close"].iloc[-3] >= support and price < support:
            alerts.append(f"跌破支撐！{sym} {price:.2f}")
    # 放量
    if len(df) > 30:
        ratio = df["Volume"].iloc[-1] / df["Volume"].iloc[-30:-1].mean()
        if ratio > 2.5:
            alerts.append(f"爆量！{sym} {ratio:.1f}x")
    # MACD
    alerts.extend(macd_alert(df, sym))

    if alerts:
        msg = "\n".join(alerts)
        key = f"{sym}_{hash(msg)}"
        if st.session_state.get(key) != msg:
            st.session_state[key] = msg
            st.session_state.history.append(f"{datetime.now():%H:%M} {msg}")
            send(msg)
            beep()
        st.success(msg)

    c1, c2, c3 = st.columns(3)
    c1.metric("股票", sym)
    c2.metric("現價", f"{price:.2f}")
    c3.metric("支撐/阻力", f"{support:.1f} / {resistance:.1f}")
    st.markdown("---")

if st.session_state.history:
    st.subheader("最近警報")
    for l in st.session_state.history[-10:]:
        st.write(l)
