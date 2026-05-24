import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
from io import StringIO
from datetime import date, timedelta
from ui.styles import TOSS_CSS

# ──────────────────────────────────────────────
# 1. 디자인
# ──────────────────────────────────────────────
st.markdown(TOSS_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2. 데이터 헬퍼 (전체 히스토리 캐시) - yfinance 안정성 개선 반영
# ──────────────────────────────────────────────
def _normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("조회 결과가 비어 있습니다.")

    normalized = df.copy()

    if isinstance(normalized.columns, pd.MultiIndex):
        if len(normalized.columns.levels) >= 2:
            normalized.columns = normalized.columns.get_level_values(0)
        else:
            normalized.columns = [col[0] if isinstance(col, tuple) else col for col in normalized.columns]

    if "Close" not in normalized.columns:
        raise ValueError(f"Close 컬럼이 없습니다. 사용 가능 컬럼: {list(normalized.columns)}")

    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized[~normalized.index.isna()]
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.sort_index()

    if normalized.empty:
        raise ValueError("정규화 후 데이터가 비어 있습니다.")

    return normalized


def _fetch_stooq_history(ticker: str) -> pd.DataFrame:
    stooq_symbol = f"{ticker.lower()}.us"
    stooq_url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    response = requests.get(stooq_url, timeout=20)
    response.raise_for_status()
    raw = response.text.strip()
    if not raw:
        raise ValueError("Stooq 응답이 비어 있습니다.")
    df = pd.read_csv(StringIO(raw))
    if df.empty:
        raise ValueError("Stooq CSV가 비어 있습니다.")
    if "Date" not in df.columns:
        raise ValueError(f"Stooq CSV에 Date 컬럼이 없습니다. 컬럼: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    return _normalize_history_frame(df)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_full_history(ticker: str) -> pd.DataFrame:
    if not ticker:
        raise ValueError("티커가 비어 있습니다.")

    errors = []

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", auto_adjust=False, actions=False, raise_errors=True)
        return _normalize_history_frame(df)
    except Exception as e:
        errors.append(f"yfinance.Ticker.history 실패: {e}")

    try:
        df = yf.download(
            ticker,
            period="max",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            timeout=20,
        )
        return _normalize_history_frame(df)
    except Exception as e:
        errors.append(f"yfinance.download 실패: {e}")

    try:
        return _fetch_stooq_history(ticker)
    except Exception as e:
        errors.append(f"Stooq CSV 실패: {e}")

    raise ValueError(f"{ticker} 히스토리 조회 실패 | " + " | ".join(errors))


def _to_series(close_obj):
    if isinstance(close_obj, pd.DataFrame):
        return close_obj.iloc[:, 0]
    return close_obj


def _compute_common_start(sell: str, buy: str):
    try:
        sell_full = fetch_full_history(sell)
    except Exception as e:
        raise ValueError(f"{sell} 데이터 로드 실패: {e}") from e

    try:
        buy_full = fetch_full_history(buy)
    except Exception as e:
        raise ValueError(f"{buy} 데이터 로드 실패: {e}") from e

    sell_first = sell_full.index.min().date()
    buy_first = buy_full.index.min().date()
    return max(sell_first, buy_first), sell_first, buy_first


def build_dataset(sell: str, buy: str, start_d, end_d):
    try:
        sell_full = fetch_full_history(sell)
    except Exception as e:
        raise ValueError(f"{sell} 히스토리 조회 실패: {e}") from e

    try:
        buy_full = fetch_full_history(buy)
    except Exception as e:
        raise ValueError(f"{buy} 히스토리 조회 실패: {e}") from e

    sell_close = _to_series(sell_full['Close'])
    buy_close = _to_series(buy_full['Close'])
    df = pd.concat([sell_close, buy_close], axis=1).dropna()
    df.columns = ['Sell_Price', 'Buy_Price']
    df = df.loc[pd.Timestamp(start_d):pd.Timestamp(end_d)]
    if df.empty:
        raise ValueError("선택 구간에 데이터가 없습니다.")
    df['Conversion_Ratio'] = df['Sell_Price'] / df['Buy_Price']
    return {
        'sell_first': sell_full.index.min().date(),
        'buy_first': buy_full.index.min().date(),
        'used_start': df.index.min().date(),
        'used_end':   df.index.max().date(),
        'latest_ratio':  float(df['Conversion_Ratio'].iloc[-1]),
        'average_ratio': float(df['Conversion_Ratio'].mean()),
        'df': df,
    }


def plot_conversion_chart(df, average_ratio, sell_ticker="", buy_ticker=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Conversion_Ratio'], mode='lines',
                             name='전환비', line=dict(color='#3182F6', width=2.2)))
    fig.add_trace(go.Scatter(x=df.index, y=[average_ratio] * len(df), mode='lines',
                             name=f'평균 ({average_ratio:.4f})',
                             line=dict(color='#8B95A1', width=1.5, dash='dash')))
    
    fig.update_layout(
        title=dict(text=f"{sell_ticker} / {buy_ticker} 전환비 추이", font=dict(size=18, color='#191F28')),
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF', hovermode='x unified',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(hoverformat='%Y-%m-%d')
    )
    return fig


# ──────────────────────────────────────────────
# 3. 콜백: 티커가 바뀐 그 순간 → 공통 시작일 자동 갱신
# ──────────────────────────────────────────────
def _on_ticker_change():
    sell = st.session_state.get("sell_ticker_input", "").strip().upper()
    buy = st.session_state.get("buy_ticker_input", "").strip().upper()
    st.session_state["sell_ticker_input"] = sell
    st.session_state["buy_ticker_input"] = buy
    st.session_state["ticker_error"] = None

    if not sell or not buy:
        return

    if st.session_state.get("_auto_pair") == (sell, buy):
        return

    try:
        common, sell_first, buy_first = _compute_common_start(sell, buy)
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
# 4. 화면 및 입력 위젯
# ──────────────────────────────────────────────
st.markdown("# 🔄 매도전환계산기")
st.caption("두 자산 간 가격 전환비(Sell / Buy)의 추이와 평균을 분석합니다.")

st.session_state.setdefault("sell_ticker_input", "TQQQ")
st.session_state.setdefault("buy_ticker_input", "SCHD")
st.session_state.setdefault("start_date", date.today() - timedelta(days=365 * 3))
st.session_state.setdefault("end_date", date.today())
st.session_state.setdefault("_auto_pair", None)
st.session_state.setdefault("ticker_error", None)
st.session_state.setdefault("auto_info", None)
st.session_state.setdefault("last_result", None)

if "_initialized" not in st.session_state:
    st.session_state["_initialized"] = True
    _on_ticker_change()

st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    st.text_input("매도 티커 (Sell)", key="sell_ticker_input", on_change=_on_ticker_change)
with c2:
    st.text_input("매수 티커 (Buy)", key="buy_ticker_input", on_change=_on_ticker_change)
with c3:
    st.date_input("시작일", key="start_date", format="YYYY/MM/DD")
with c4:
    st.date_input("종료일", key="end_date", format="YYYY/MM/DD")

if st.session_state["ticker_error"]:
    st.error(f"⚠️ {st.session_state['ticker_error']}")
elif st.session_state["auto_info"]:
    info = st.session_state["auto_info"]
    later = info["sell"] if info["sell_first"] >= info["buy_first"] else info["buy"]
    st.info(f"📅 공통 시작일 자동 추천: **{info['common']}** (더 늦은 **{later}** 기준)")

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    run = st.button("분석 실행", type="primary", use_container_width=True)
with btn_col2:
    if st.button("캐시 초기화", use_container_width=True):
        fetch_full_history.clear()
        st.rerun()


# ──────────────────────────────────────────────
# 5. 분석 실행 및 결과 렌더링
# ──────────────────────────────────────────────
if run:
    sell_ticker = st.session_state["sell_ticker_input"].strip().upper()
    buy_ticker = st.session_state["buy_ticker_input"].strip().upper()
    start_d = st.session_state["start_date"]
    end_d = st.session_state["end_date"]

    if not sell_ticker or not buy_ticker:
        st.warning("티커를 모두 입력해주세요.")
        st.stop()
    
    try:
        st.session_state["last_result"] = {
            "sell_ticker": sell_ticker,
            "buy_ticker": buy_ticker,
            **build_dataset(sell_ticker, buy_ticker, start_d, end_d),
        }
    except Exception as e:
        st.session_state["last_result"] = None
        st.error(f"데이터 로드 실패: {e}")

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

    fig = plot_conversion_chart(df, res['average_ratio'], sell_ticker, buy_ticker)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"📋 일자별 상세 데이터 보기 ({res['used_start']} ~ {res['used_end']})"):
        st.dataframe(df.style.format('{:.4f}'), use_container_width=True)
