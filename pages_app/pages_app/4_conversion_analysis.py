import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta
from ui.styles import TOSS_CSS

# ──────────────────────────────────────────────
# 1. 디자인 및 페이지 설정
# ──────────────────────────────────────────────
st.markdown(TOSS_CSS, unsafe_allow_html=True)
# 1번 디자인 섹션에 아래 내용을 추가해 보세요
st.markdown("""
    <style>
    /* 달력 내부의 영문 월 이름을 숫자로 강제 변환하려는 시도 (CSS hack) */
    /* 다만, 이 방법은 브라우저마다 렌더링 방식이 달라 완벽하지 않을 수 있습니다. */
    div[data-baseweb="calendar"] button div {
        font-family: 'Pretendard', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)
# ──────────────────────────────────────────────
# 2. 데이터 처리 함수 (Helper Functions)
# ──────────────────────────────────────────────

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_full_history(ticker: str) -> pd.DataFrame:
    """티커 전체 히스토리를 가져와 캐시합니다."""
    if not ticker:
        return pd.DataFrame()
    df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    return df if df is not None else pd.DataFrame()

def _to_series(close_obj):
    """DataFrame에서 종가 Series만 추출합니다."""
    if isinstance(close_obj, pd.DataFrame):
        return close_obj.iloc[:, 0]
    return close_obj

def _compute_common_start(sell: str, buy: str):
    """두 티커의 공통 시작일을 계산합니다."""
    sell_full = fetch_full_history(sell)
    buy_full = fetch_full_history(buy)
    
    if sell_full.empty or buy_full.empty:
        missing = []
        if sell_full.empty: missing.append(sell)
        if buy_full.empty: missing.append(buy)
        raise ValueError(f"데이터를 찾을 수 없음: {', '.join(missing)}")
        
    sell_first = sell_full.index.min().date()
    buy_first = buy_full.index.min().date()
    common_start = max(sell_first, buy_first)
    return common_start, sell_first, buy_first

def build_dataset(sell: str, buy: str, start_d, end_d):
    """선택된 기간에 맞춰 전환비를 계산한 데이터셋을 생성합니다."""
    sell_full = fetch_full_history(sell)
    buy_full = fetch_full_history(buy)
    
    sell_close = _to_series(sell_full['Close'])
    buy_close = _to_series(buy_full['Close'])
    
    df = pd.concat([sell_close, buy_close], axis=1).dropna()
    df.columns = ['Sell_Price', 'Buy_Price']
    
    # 날짜 필터링
    df = df.loc[pd.Timestamp(start_d):pd.Timestamp(end_d)]
    
    if df.empty:
        raise ValueError("해당 기간에 데이터가 존재하지 않습니다.")
        
    df['Conversion_Ratio'] = df['Sell_Price'] / df['Buy_Price']
    
    return {
        'sell_first': sell_full.index.min().date(),
        'buy_first': buy_full.index.min().date(),
        'used_start': df.index.min().date(),
        'used_end': df.index.max().date(),
        'latest_ratio': float(df['Conversion_Ratio'].iloc[-1]),
        'average_ratio': float(df['Conversion_Ratio'].mean()),
        'df': df,
    }

def plot_conversion_chart(df, average_ratio, sell_ticker="", buy_ticker=""):
    """전환비 추이 차트를 생성합니다."""
    fig = go.Figure()
    
    # 전환비 선
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Conversion_Ratio'],
        mode='lines', name='전환비',
        line=dict(color='#3182F6', width=2.2)
    ))
    
    # 평균선
    fig.add_trace(go.Scatter(
        x=df.index, y=[average_ratio] * len(df),
        mode='lines', name=f'평균 ({average_ratio:.4f})',
        line=dict(color='#8B95A1', width=1.5, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(text=f"{sell_ticker} / {buy_ticker} 전환비 추이", font=dict(size=18, color='#191F28')),
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
        hovermode='x unified',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(hoverformat='%Y-%m-%d')  # ✅ 툴팁 날짜 형식 고정
    )
    return fig

# ──────────────────────────────────────────────
# 3. 콜백 및 세션 관리
# ──────────────────────────────────────────────

def _on_ticker_change():
    """티커 입력 시 공통 시작일을 자동으로 찾아 설정하는 핵심 콜백입니다."""
    sell = st.session_state.get("sell_ticker_input", "").strip().upper()
    buy = st.session_state.get("buy_ticker_input", "").strip().upper()
    
    st.session_state["sell_ticker_input"] = sell
    st.session_state["buy_ticker_input"] = buy
    st.session_state["ticker_error"] = None

    if not sell or not buy:
        return

    # 이미 처리한 티커 조합이면 스킵
    if st.session_state.get("_auto_pair") == (sell, buy):
        return

    try:
        common, sell_first, buy_first = _compute_common_start(sell, buy)
        # 자동 날짜 갱신
        st.session_state["start_date"] = common
        st.session_state["_auto_pair"] = (sell, buy)
        st.session_state["auto_info"] = {
            "sell": sell, "buy": buy,
            "sell_first": sell_first, "buy_first": buy_first,
            "common": common,
        }
    except Exception as e:
        st.session_state["ticker_error"] = str(e)
        st.session_state["_auto_pair"] = None
        st.session_state["auto_info"] = None

# ──────────────────────────────────────────────
# 4. 화면 레이아웃 (UI)
# ──────────────────────────────────────────────

st.markdown("# 🔄 매도전환계산기")
st.caption("두 자산 간 가격 전환비(Sell / Buy)의 추이와 평균을 분석합니다.")

# 세션 초기값 설정
st.session_state.setdefault("sell_ticker_input", "TQQQ")
st.session_state.setdefault("buy_ticker_input", "SCHD")
st.session_state.setdefault("start_date", date.today() - timedelta(days=365 * 3))
st.session_state.setdefault("end_date", date.today())
st.session_state.setdefault("_auto_pair", None)
st.session_state.setdefault("ticker_error", None)
st.session_state.setdefault("auto_info", None)
st.session_state.setdefault("last_result", None)

# 최초 실행 시 공통 시작일 계산
if "_initialized" not in st.session_state:
    st.session_state["_initialized"] = True
    _on_ticker_change()

st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)

# 입력 위젯
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    st.text_input("매도 티커 (Sell)", key="sell_ticker_input", on_change=_on_ticker_change)
with c2:
    st.text_input("매수 티커 (Buy)", key="buy_ticker_input", on_change=_on_ticker_change)
with c3:
    # ✅ format="YYYY/MM/DD"로 숫자 중심 UI 제공
    st.date_input("시작일", key="start_date", format="YYYY/MM/DD")
with c4:
    st.date_input("종료일", key="end_date", format="YYYY/MM/DD")

# 메시지 출력
if st.session_state["ticker_error"]:
    st.error(f"⚠️ {st.session_state['ticker_error']}")
elif st.session_state["auto_info"]:
    info = st.session_state["auto_info"]
    later = info["sell"] if info["sell_first"] >= info["buy_first"] else info["buy"]
    st.info(f"📅 공통 시작일 자동 추천: **{info['common']}** (더 늦은 **{later}** 기준)")

run = st.button("분석 실행", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# 5. 분석 로직 및 결과 렌더링
# ──────────────────────────────────────────────

if run:
    sell_ticker = st.session_state["sell_ticker_input"].strip().upper()
    buy_ticker = st.session_state["buy_ticker_input"].strip().upper()
    start_d = st.session_state["start_date"]
    end_d = st.session_state["end_date"]

    if not sell_ticker or not buy_ticker:
        st.warning("티커를 모두 입력해주세요.")
    elif start_d >= end_d:
        st.warning("시작일은 종료일보다 빨라야 합니다.")
    else:
        with st.spinner("데이터 분석 중..."):
            try:
                res_data = build_dataset(sell_ticker, buy_ticker, start_d, end_d)
                st.session_state["last_result"] = {
                    "sell_ticker": sell_ticker,
                    "buy_ticker": buy_ticker,
                    **res_data
                }
            except Exception as e:
                st.error(f"분석 실패: {e}")
                st.session_state["last_result"] = None

# 결과 표시
res = st.session_state.get("last_result")
if res:
    sell_ticker = res["sell_ticker"]
    buy_ticker = res["buy_ticker"]
    df = res["df"]

    st.success(f"✅ 분석 기간: **{res['used_start']} ~ {res['used_end']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{sell_ticker} 시작일", str(res['sell_first']))
    m2.metric(f"{buy_ticker} 시작일", str(res['buy_first']))
    m3.metric("현재 전환비", f"{res['latest_ratio']:.4f}")
    m4.metric("평균 전환비", f"{res['average_ratio']:.4f}", 
              delta=f"{(res['latest_ratio'] - res['average_ratio']):+.4f}")

    # 차트 출력
    fig = plot_conversion_chart(df, res['average_ratio'], sell_ticker, buy_ticker)
    st.plotly_chart(fig, use_container_width=True)

    # 상세 데이터
    with st.expander(f"📋 일자별 상세 데이터 보기 ({res['used_start']} ~ {res['used_end']})"):
        st.dataframe(df.style.format('{:.4f}'), use_container_width=True)