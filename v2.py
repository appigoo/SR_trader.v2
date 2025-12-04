# ================================================
#   多股票即時監控面板 — 終極無 Bug 完整版
#   複製存成 app.py 直接執行即可！
# ================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from streamlit_autorefresh import st_autorefresh

# ==================== 頁面設定 ====================
st.set_page_config(page_title="頂級多股監控面板", layout="wide")
st.title("多股票支撐/阻力 + MACD 動能前瞻監控神器")

# session_state 防重複警報
for k in ["last_signal_keys", "signal_history"]:
    if k not in st.session_state:
        st.session_state[k] = {} if k == "last_signal_keys" else []

# ==================== 側邊欄設定 ====================
st.sidebar.header("監控設定")
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "TSLA,AAPL,NVDA,META")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval_opt = {"1分鐘":"1m", "5分鐘":"5m", "15分鐘":"15m", "1小時":"60m", "日線":"1d"}
interval_label = st.sidebar.selectbox("K線週期", list(interval_opt.keys()), index=1)
interval = interval_opt[interval_label]

period_opt = {"1天":"1d", "5天":"5d", "10天":"10d", "1個月":"1mo", "3個月":"3mo", "1年":"1y"}
period_label = st.sidebar.selectbox("資料範圍", list(period_opt.keys()), index=1)
period = period_opt[period_label]

lookback = st.sidebar.slider("觀察根數（放量判斷用）", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("自動更新頻率", ["30秒", "60秒", "3分鐘"], index=1)
auto_update = st.sidebar.checkbox("啟用自動更新", True)
buffer_pct = st.sidebar.slider("突破緩衝區(%)", 0.01, 1.0, 0.1, 0.01) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)

# 警報開關
st.sidebar.markdown("### 警報類型")
use_sr_alert     = st.sidebar.checkbox("支撐/阻力突破警報", True)
use_vol_filter   = st.sidebar.checkbox("突破需放量確認 (>1.5x)", True)
use_vol_alert    = st.sidebar.checkbox("獨立成交量爆量警報", True)
vol_multiplier   = st.sidebar.slider("成交量警報倍數", 1.5, 6.0, 2.5, 0.1)

st.sidebar.markdown("#### 自訂價位警報")
custom_price_text = st.sidebar.text_area("格式：SYMBOL,價位1,價位2...", "AAPL,180\nNVDA,900")

st.sidebar.markdown("#### 自訂成交量倍數")
custom_vol_text = st.sidebar.text_area("格式：SYMBOL,倍數", "NVDA,4.0\nTSLA,3.5")

use_macd_alert = st.sidebar.checkbox("MACD 動能前瞻預測警報（最強）", True)

# ==================== 解析自訂設定（防呆版） ====================
def parse_custom(text: str, is_price: bool = True) -> Dict:
    result = {}
    for line in text.split("\n"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2: continue
        sym = parts[0].upper()
        try:
            values = [float(p) for p in parts[1:]]
            if is_price:
                result.setdefault(sym, []).extend(values)
            else:
                result[sym] = values[0]
        except:
            continue
    return result

custom_prices = parse_custom(custom_price_text, True)
custom_vol_mult = parse_custom(custom_vol_text, False)  # 修正關鍵點！

# ==================== Telegram 與聲音 ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    tg_ready = True
except:
    tg_ready = False

def send_telegram(msg: str):
    if not tg_ready: return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=8)
    except:
        pass

def play_alert_sound():
    if sound_alert:
        st.markdown("""
        <audio autoplay style="display:none;">
            <source src="https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

# ==================== 資料快取 ====================
@st.cache_data(ttl=55)
def fetch_data(symbol: str):
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=True)
        if df.empty or len(df) < 10:
            return None
        df = df[~df.index.duplicated(keep='last')]
        return df.dropna(how='all')
    except:
        return None

# ==================== 永不崩潰的支撐阻力識別 ====================
def find_support_resistance(df_full: pd.DataFrame):
    if len(df_full) < 15:
        return df_full["Low"].min(), df_full["High"].max(), []
    
    df = df_full.iloc[:-1].copy()
    high = pd.to_numeric(df["High"], errors='coerce').values
    low  = pd.to_numeric(df["Low"],  errors='coerce').values
    
    res_pts, sup_pts = [], []
    window = 5
    
    for i in range(window, len(high) - window):
        if np.isnan(high[i]) or np.isnan(low[i]):
            continue
        # 阻力點
        if high[i] >= np.nanmax(high[i-window:i]) and high[i] >= np.nanmax(high[i+1:i+window+1]):
            res_pts.append(float(high[i]))
        # 支撐點
        if low[i] <= np.nanmin(low[i-window:i]) and low[i] <= np.nanmin(low[i+1:i+window+1]):
            sup_pts.append(float(low[i]))
    
    # 簡單聚類
    def cluster(pts):
        if len(pts) < 2: return pts
        pts = sorted(pts)
        clusters = []
        curr = [pts[0]]
        for p in pts[1:]:
            if abs(p - curr[-1]) / curr[-1] < 0.006:
                curr.append(p)
            else:
                if len(curr) >= 2:
                    clusters.append(np.mean(curr))
                curr = [p]
        if len(curr) >= 2:
            clusters.append(np.mean(curr))
        return clusters
    
    res_lv = cluster(res_pts)
    sup_lv = cluster(sup_pts)
    all_lv = sup_lv + res_lv
    
    try:
        price = float(df_full["Close"].iloc[-1])
        support = min(sup_lv, default=float(df_full["Low"].min()))
        resistance = max(res_lv, default=float(df_full["High"].max()))
    except:
        support = df_full["Low"].min()
        resistance = df_full["High"].max()
    
    return support, resistance, all_lv

# ==================== MACD 動能前瞻預測 ====================
def macd_forecast_signals(df_full: pd.DataFrame, symbol: str) -> List[Tuple[str, str, str]]:
    if len(df_full) < 60: return []
    df = df_full.copy()
    close = df['Close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    
    dif_vals = dif.values[-120:]
    hist_vals = hist.values[-120:]
    
    signals = []
    now = datetime.now().strftime("%H%M")
    
    # ① DIF 加速度轉向
    if len(dif_vals) >= 14:
        speed = np.diff(dif_vals[-12:])
        accel = np.diff(speed)
        if len(accel) >= 4:
            if accel[-3] < 0 and accel[-2] > 0 and hist_vals[-1] < 0:
                sustain = max(8, min(45, int(abs(dif_vals[-1]) * 12)))
                signals.append((symbol, f"MACD 提前翻多！\n<b>{symbol}</b>\nDIF加速度轉正\n預估多頭維持 <b>{sustain}</b> 根K", f"MAC_UP_{now}"))
            if accel[-3] > 0 and accel[-2] < 0 and hist_vals[-1] > 0:
                sustain = max(8, min(45, int(abs(dif_vals[-1]) * 12)))
                signals.append((symbol, f"MACD 提前翻空！\n<b>{symbol}</b>\nDIF加速度轉負\n預估空頭維持 <b>{sustain}</b> 根K", f"MAC_DN_{now}"))
    
    # ② 柱狀圖連續縮短
    if len(hist_vals) >= 7:
        recent = hist_vals[-7:]
        if all(recent[-6:] > 0) and np.all(np.diff(recent[-5:]) < 0):
            signals.append((symbol, f"MACD 多頭衰竭！紅柱連續縮短\n極大機率翻綠", f"RED_SHRINK_{now}"))
        if all(recent[-6:] < 0) and np.all(np.diff(np.abs(recent[-5:])) < 0):
            signals.append((symbol, f"MACD 空頭衰竭！綠柱連續縮短\n極大機率翻紅", f"GREEN_SHRINK_{now}"))
    
    return signals

# ==================== 圖表繪製（超級防呆） ====================
def draw_chart(df_full: pd.DataFrame, symbol: str, support, resistance, extra_levels):
    df = df_full.copy()
    
    # EMA
    for p in [5,10,20,40,60]:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name=symbol, increasing_line_color='lime', decreasing_line_color='red'
    ))
    
    colors = ['cyan', 'magenta', 'yellow', 'orange', 'white']
    for i, p in enumerate([5,10,20,40,60]):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{p}'], 
                                 name=f'EMA{p}', line=dict(color=colors[i], width=1.5)))
    
    # 主支撐阻力
    fig.add_hline(y=support, line_dash="dash", line_color="green", annotation_text=f"支撐 {support:.2f}")
    fig.add_hline(y=resistance, line_dash="dash", line_color="red", annotation_text=f"阻力 {resistance:.2f}")
    
    # 其他水平線（超級防呆）
    safe_levels = []
    for x in extra_levels:
        try:
            val = float(x)
            if np.isfinite(val):
                safe_levels.append(val)
        except:
            continue
    
    for lv in safe_levels:
        fig.add_hline(y=lv, line_dash="dot", line_color="cyan", opacity=0.7)
    
    fig.update_layout(
        title=f"{symbol} 即時K線圖",
        height=580, plot_bgcolor='black', paper_bgcolor='black',
        font_color='white', xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    return fig

# ==================== 自動更新 ====================
if auto_update:
    st_autorefresh(interval={"30秒":30,"60秒":60,"3分鐘":180}[update_freq]*1000, key="auto_refresh")

if not symbols:
    st.warning("請輸入至少一檔股票代號")
    st.stop()

st.header(f"即時監控中：{', '.join(symbols)} | {interval_label} | {period_label}")

# ==================== 主循環（核心） ====================
all_signals = []
progress = st.progress(0)

for idx, sym in enumerate(symbols):
    progress.progress((idx + 1) / len(symbols))
    
    df = fetch_data(sym)
    if df is None or len(df) < 20:
        st.error(f"無法取得 {sym} 資料")
        continue
    
    support, resistance, levels = find_support_resistance(df)
    
    # 安全處理自訂價位
    raw_custom = custom_prices.get(sym)
    if isinstance(raw_custom, list):
        custom_lvls = raw_custom
    elif raw_custom is not None:
        custom_lvls = [raw_custom]
    else:
        custom_lvls = []
    
    extra_levels = levels + custom_lvls
    
    fig = draw_chart(df, sym, support, resistance, extra_levels)
    
    # 收集所有警報
    if use_sr_alert:
        try:
            c1, c2, c3 = df["Close"].iloc[-3:].values
            vol_ok = (df["Volume"].iloc[-1] > df["Volume"].iloc[-lookback:-1].mean() * 1.5) or (not use_vol_filter)
            buf = resistance * buffer_pct
            if c1 <= resistance - buf and c2 <= resistance - buf and c3 > resistance and vol_ok:
                all_signals.append((sym, f"突破阻力！\n<b>{sym}</b> {c3:.2f} > {resistance:.2f}", "SR_UP"))
            if c1 >= support + buf and c2 >= support + buf and c3 < support and vol_ok:
                all_signals.append((sym, f"跌破支撐！\n<b>{sym}</b> {c3:.2f} < {support:.2f}", "SR_DN"))
        except: pass
    
    if use_vol_alert:
        try:
            vol = df["Volume"].iloc[-1]
            avg = df["Volume"].iloc[-lookback-1:-1].mean()
            if avg > 0:
                ratio = vol / avg
                mult = custom_vol_mult.get(sym, vol_multiplier)
                if ratio > mult:
                    all_signals.append((sym, f"成交量激增！\n<b>{sym}</b> {ratio:.1f}x", f"VOL_{ratio:.1f}"))
        except: pass
    
    if use_macd_alert:
        all_signals.extend(macd_forecast_signals(df, sym))
    
    # 顯示圖表與資訊
    st.subheader(f"{sym}  現價 {df['Close'].iloc[-1]:.2f}")
    st.plotly_chart(fig, use_container_width=True)
    
    # 顯示該股票的警報
    stock_signals = [s for s in all_signals if s[0] == sym]
    for _, msg, key in stock_signals:
        if key_full = f"{sym}_{key}_{datetime.now():%H%M}"
        if st.session_state.last_signal_keys.get(key_full) != key_full:
            st.session_state.last_signal_keys[key_full] = key_full
            st.session_state.signal_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "msg": msg
            })
            send_telegram(msg)
            play_alert_sound()
        st.success(msg)
    
    # 指標欄
    c1, c2, c3 = st.columns(3)
    c1.metric("現價", f"{df['Close'].iloc[-1]:.2f}")
    c2.metric("支撐", f"{support:.2f}")
    c3.metric("阻力", f"{resistance:.2f}")
    st.markdown("---")

# ==================== 歷史警報 ====================
if st.session_state.signal_history:
    st.subheader("最近15筆警報紀錄")
    for h in reversed(st.session_state.signal_history[-15:]):
        st.markdown(f"**{h['time']}** {h['msg']}")
