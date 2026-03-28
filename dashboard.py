import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import os
import plotly.express as px
import plotly.graph_objects as go

# 경로 설정 (상대 경로로 변경하여 이식성 높임)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "commodity_analysis_final.db")

# 지정 업데이트 타겟 티커 5가지 및 테이블 컬럼별 맵핑
TICKERS = ["GC=F", "SI=F", "UUP", "^TNX", "^GSPC"]
TICKER_MAP = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "UUP": "US Dollar Index",
    "^TNX": "10Y TY",
    "^GSPC": "S&P500"
}
TARGET_ASSETS = ['Gold', 'Silver', 'S&P500', 'US Dollar Index', '10Y TY']

# 이벤트 및 기준일 정의
EVENTS = {
    '코로나 팬데믹': '2020-03-11',
    '미국 금리 인상 사이클': '2022-03-16',
    '러시아-우크라이나 전쟁': '2022-02-24',
    '실리콘밸리 은행(SVB) 파산': '2023-03-10',
    '미국-이란 갈등': '2026-02-27'
}

def update_database_from_yfinance():
    """DB 마지막 날짜 이력부터 yfinance에서 신규 데이터를 받아 테이블에 병합(INSERT)"""
    try:
        if not os.path.exists(DB_PATH):
            st.error(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {DB_PATH}")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM wide_prices")
        max_date_str = cursor.fetchone()[0]
        
        if max_date_str is None:
            st.warning("⚠️ DB가 비어있습니다. 초기 데이터가 필요합니다.")
            conn.close()
            return

        max_date = pd.to_datetime(max_date_str).date()
        today = datetime.datetime.today().date()
        next_day = max_date + datetime.timedelta(days=1)
        
        if next_day <= today:
            with st.spinner(f"🚀 최신 데이터를 수집 중입니다... ({next_day} ~ {today})"):
                all_prices = []
                for tk in TICKERS:
                    data = yf.download(tk, start=str(next_day), end=str(today + datetime.timedelta(days=1)), progress=False)
                    if not data.empty:
                        series = data['Close'].copy()
                        if isinstance(series, pd.DataFrame):
                            series = series.iloc[:, 0]
                        series.name = TICKER_MAP[tk]
                        all_prices.append(series)

                if all_prices:
                    prices = pd.concat(all_prices, axis=1)
                    prices = prices.reset_index()
                    prices.rename(columns={prices.columns[0]: 'Date'}, inplace=True)
                    prices['Date'] = pd.to_datetime(prices['Date']).dt.strftime('%Y-%m-%d')
                    prices = prices.ffill()

                    if 'Gold' in prices.columns and 'Silver' in prices.columns:
                        prices['Ratio'] = prices['Gold'] / prices['Silver']
                    
                    ordered_cols = ['Date', 'Gold', 'Silver', 'S&P500', 'US Dollar Index', '10Y TY', 'Ratio']
                    for col in ordered_cols:
                        if col not in prices.columns:
                            prices[col] = None
                    prices = prices[ordered_cols]
                    
                    prices.to_sql('wide_prices', conn, if_exists='append', index=False)
                    st.success(f"🎉 성공! 최신 영업일({len(prices)}일치) 데이터가 DB에 연동되었습니다!")
        conn.close()
    except Exception as e:
        st.error(f"❌ DB 자동 업데이트 중 에러가 발생했습니다: {e}")

@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM wide_prices ORDER BY Date ASC", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.ffill()
    return df

def calculate_metrics(cum_returns):
    """누적 수익률 기반 주요 지표 계산 (Total Return, MDD, Sharpe)"""
    total_return = (cum_returns.iloc[-1] / 100 - 1) * 100
    
    # MDD 계산
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    mdd = drawdown.min() * 100
    
    # Sharpe Ratio (단순화: 무위험 수익률 0 가정)
    daily_rets = cum_returns.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() != 0 else 0
    
    return total_return, mdd, sharpe

def extract_event_window(df, event_date, window_before=30, window_after=60):
    """지정 이벤트 기준 전후 날짜(T-30 ~ T+60) 인덱스를 추출하여 T-30 기준으로 100 정규화"""
    event_idx_list = df[df['Date'] >= event_date].index
    if len(event_idx_list) == 0:
        return None
    event_idx = event_idx_list[0]
    
    start_idx = max(0, event_idx - window_before)
    end_idx = min(len(df) - 1, event_idx + window_after)
    
    window_df = df.iloc[start_idx:end_idx+1].copy()
    window_df['Relative_Day'] = range(-window_before, len(window_df) - window_before)
    
    # 윈도우별 정규화 (최초일 T-30 = 100 기반)
    for col in ['Gold', 'Silver', 'S&P500', 'US Dollar Index']:
        base_val = window_df[col].iloc[0]
        if base_val != 0:
            window_df[col] = (window_df[col] / base_val) * 100
            
    return window_df

# =======================
# 대시보드 설정 및 사이드바
# =======================
st.set_page_config(page_title="Gold-Silver Quant Dashboard", layout="wide")

st.sidebar.title("🛠️ 전략 설정")
st.sidebar.markdown("백테스트 및 분석 파라미터를 조절하세요.")

# 데이터 업데이트 및 로드
update_database_from_yfinance()
full_df = load_data()

if full_df.empty:
    st.error("데이터를 불러올 수 없습니다. DB 파일 경로를 확인하세요.")
    st.stop()

# 사이드바: 날짜 범위 선택
min_date = full_df['Date'].min().date()
max_date = full_df['Date'].max().date()
date_range = st.sidebar.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)

# 선택된 기간으로 데이터 필터링
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = full_df[(full_df['Date'] >= start_date) & (full_df['Date'] <= end_date)].copy()
else:
    df = full_df.copy()

# 사이드바: 위기 시 자산 비중 설정
st.sidebar.subheader("🚨 위기 대응 비중 (Crisis)")
c_gold = st.sidebar.slider("금(Gold) 비중", 0.0, 1.0, 0.4, 0.1)
c_dollar = st.sidebar.slider("달러(USD) 비중", 0.0, 1.0, 0.4, 0.1)
c_sp500 = st.sidebar.slider("주식(S&P500) 비중", 0.0, 1.0, 0.0, 0.1)
c_other = 1.0 - (c_gold + c_dollar + c_sp500)
if c_other < 0:
    st.sidebar.error("비중의 합이 1.0을 초과했습니다!")
    c_other = 0.0
st.sidebar.info(f"기타 자산 비중: {c_other:.1f} (은, 국채)")

# =======================
# 메인 화면 렌더링
# =======================
st.title("📊 다중 자산 폭락 방어형 퀀트(Quant) 대시보드")
st.markdown(f"**분석 기간:** {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")

# 수익률 연산
for col in TARGET_ASSETS:
    df[f'{col}_Ret'] = df[col].pct_change().fillna(0)

# 백테스트 로직 실행
df['SP500_5d_Ret'] = df['S&P500'].pct_change(periods=5).fillna(0)
base_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
ret_cols = [f'{col}_Ret' for col in TARGET_ASSETS]
df['Baseline_Ret'] = df[ret_cols].dot(base_weights)

# Crisis Strategy 적용
crisis_weights = [c_gold, (1.0-(c_gold+c_dollar+c_sp500))/2, c_sp500, c_dollar, (1.0-(c_gold+c_dollar+c_sp500))/2]
is_crisis = df['SP500_5d_Ret'] < -0.05
df['Crisis_Ret'] = np.where(is_crisis, df[ret_cols].dot(crisis_weights), df[ret_cols].dot(base_weights))

df['Baseline_Cum'] = (1 + df['Baseline_Ret']).cumprod() * 100
df['Crisis_Cum'] = (1 + df['Crisis_Ret']).cumprod() * 100

# -----------------------
# 상단 KPI 섹션
# -----------------------
col1, col2, col3, col4 = st.columns(4)

b_ret, b_mdd, b_sharpe = calculate_metrics(df['Baseline_Cum'])
c_ret, c_mdd, c_sharpe = calculate_metrics(df['Crisis_Cum'])

with col1:
    st.metric("누적 수익률 (Crisis)", f"{c_ret:.2f}%", f"{c_ret - b_ret:.2f}% vs Base")
with col2:
    st.metric("최대 낙폭 (MDD)", f"{c_mdd:.2f}%", f"{c_mdd - b_mdd:.2f}%", delta_color="normal")
with col3:
    st.metric("샤프 지수 (Sharpe)", f"{c_sharpe:.2f}", f"{c_sharpe - b_sharpe:.2f}")
with col4:
    crisis_days = is_crisis.sum()
    st.metric("위기 대응 발동 횟수", f"{crisis_days}일", "영업일 기준")

st.write("---")

# 4개의 개별 시각화를 탭 형태로 구성
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 1. 이벤트 흐름 비교", 
    "📈 2. 전략 성과 분석", 
    "📉 3. 자산별 낙폭 추이", 
    "🌡️ 4. 상관관계 매트릭스"
])

# =================
# 탭 1. 이벤트별 차트
# =================
with tab1:
    st.subheader("이벤트 발생일(T=0) 기준 전후 자산 가격 흐름")
    selected_event = st.selectbox("조회할 지정 이벤트:", list(EVENTS.keys()))
    ev_date = EVENTS[selected_event]
    
    w_df = extract_event_window(full_df, ev_date) # 전체 데이터 기준 추출
    
    if w_df is not None:
        fig1 = px.line(
            w_df, x="Relative_Day", 
            y=['Gold', 'Silver', 'S&P500', 'US Dollar Index'],
            title=f"'{selected_event}' 전후 자산 가격 (T-30=100 정규화)",
            labels={'value': '정규화 가격', 'Relative_Day': '경과일(T)'},
            template="plotly_white"
        )
        fig1.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Event Day")
        fig1.update_layout(hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("해당 이벤트의 데이터가 부족합니다.")

# =================
# 탭 2. 전략 성과 분석
# =================
with tab2:
    st.subheader("Baseline vs Crisis 포트폴리오 누적 수익률")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Baseline_Cum'], name='Baseline (Equal Weight)', line=dict(color='gray', dash='dot')))
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Crisis_Cum'], name='Crisis (Dynamic Switching)', line=dict(color='#00CC96', width=3)))
    
    fig2.update_layout(
        title="포트폴리오 누적 성과 비교",
        xaxis_title="날짜", yaxis_title="수익률 (100 기준)",
        template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =================
# 탭 3. 낙폭 추이
# =================
with tab3:
    st.subheader("전고점 대비 하락률 (Drawdown Tracker)")
    dd_df = df.copy()
    for col in ['Gold', 'Silver', 'S&P500', 'US Dollar Index']:
        peak = dd_df[col].cummax()
        dd_df[f'{col}_DD'] = ((dd_df[col] - peak) / peak) * 100
        
    fig3 = px.area(
        dd_df, x="Date", 
        y=['S&P500_DD', 'Gold_DD', 'US Dollar Index_DD'],
        title="자산별 드로우다운 추이",
        labels={'value': '하락률 (%)', 'variable': '자산'},
        template="plotly_white"
    )
    fig3.update_layout(hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

# =================
# 탭 4. 상관관계
# =================
with tab4:
    st.subheader("자산 간 일일 수익률 상관관계")
    corr_matrix = df[ret_cols].corr()
    corr_matrix.columns = TARGET_ASSETS
    corr_matrix.index = TARGET_ASSETS
    
    fig4 = px.imshow(
        corr_matrix, 
        text_auto=".2f",
        color_continuous_scale="RdBu_r", 
        zmin=-1, zmax=1,
        title="수익률 상관관계 히트맵"
    )
    st.plotly_chart(fig4, use_container_width=True)

# 데이터 스냅샷
with st.expander("🔍 데이터 원본 스냅샷 (최근 10일)"):
    st.dataframe(df.tail(10))

