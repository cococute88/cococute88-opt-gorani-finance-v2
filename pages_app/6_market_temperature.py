import time
import traceback
from io import StringIO

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests

from ui.styles import TOSS_CSS
from logic.market import (
    compute_rsi,
    compute_drawdown_series,
    compute_distance_from_moving_average,
    compute_gorani_market_temperature,
    compute_gorani_market_temperature_v2,
    classify_fear_greed_score,
)

# ──────────────────────────────────────────────
# 1. 디자인
# ──────────────────────────────────────────────
st.markdown(TOSS_CSS, unsafe_allow_html=True)

# 시장온도 기준 종목 및 색상
WATCHLIST = ["QQQ", "SCHD", "SPY"]
TICKER_COLORS = {
    "QQQ": "#3182F6",   # 파랑
    "SCHD": "#00875A",  # 초록
    "SPY": "#FF8B00",   # 주황
}

# 표시 기간 옵션 (라벨 → 거래일 수, None 이면 전체)
RANGE_OPTIONS = {
    "6개월": 126,
    "1년": 252,
    "3년": 756,
    "5년": 1260,
    "전체": None,
}


# ──────────────────────────────────────────────
# 2. 데이터 헬퍼 (4_conversion_analysis.py 패턴 복제)
#    - 기존 페이지를 import 하면 스크립트가 실행되므로, 안전하게 복제하여 사용
# ──────────────────────────────────────────────
def _normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("조회 결과가 비어 있습니다.")

    normalized = df.copy()

    if isinstance(normalized.columns, pd.MultiIndex):
        if len(normalized.columns.levels) >= 2:
            normalized.columns = normalized.columns.get_level_values(0)
        else:
            normalized.columns = [
                col[0] if isinstance(col, tuple) else col for col in normalized.columns
            ]

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
def fetch_close_series(ticker: str) -> pd.Series:
    """티커의 전체 종가(Close) Series 를 반환한다 (yfinance → download → Stooq 폴백)."""
    if not ticker:
        raise ValueError("티커가 비어 있습니다.")

    errors = []

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", auto_adjust=False, actions=False, raise_errors=True)
        normalized = _normalize_history_frame(df)
        return normalized["Close"].astype("float64")
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
        normalized = _normalize_history_frame(df)
        return normalized["Close"].astype("float64")
    except Exception as e:
        errors.append(f"yfinance.download 실패: {e}")

    try:
        normalized = _fetch_stooq_history(ticker)
        return normalized["Close"].astype("float64")
    except Exception as e:
        errors.append(f"Stooq CSV 실패: {e}")

    raise ValueError(f"{ticker} 종가 조회 실패 | " + " | ".join(errors))


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def compute_v2_diagnostics(try_pcr: bool = False) -> dict:
    """고라니 시장온도 v2 진단을 계산한다 (lazy / 캐시 6시간).

    페이지 로드 시 자동 실행되지 않고, 사용자가 'v2 진단 계산하기' 버튼을
    눌렀을 때만 호출되어야 한다. 각 티커 조회는 개별적으로 격리되어 하나가
    실패해도 나머지 계산은 계속 진행되며, 전체 페이지를 죽이지 않는다.

    Args:
        try_pcr: True 이면 ^CPC → ^CPCE 직접 조회를 추가로 시도한다.
                 False(기본)면 ^CPC/^CPCE 는 조회하지 않고, Put/Call 구성요소는
                 VIX/VIX3M proxy 로만 산출한다.

    Returns:
        dict: {
            "result": v2 계산 결과 dict (compute_gorani_market_temperature_v2 반환) 또는 None,
            "ticker_ok": 성공한 티커 list,
            "ticker_errors": {티커: 오류문자열} dict,
            "elapsed": 계산 소요 시간(초, float),
            "error": 전체 예외 traceback 문자열 또는 None,
            "used_pcr_direct": ^CPC/^CPCE 직접 데이터를 실제로 사용했는지(bool),
            "putcall_source": Put/Call 구성요소가 실제 사용한 소스 문자열,
        }
    """
    start = time.perf_counter()
    ticker_ok = []
    ticker_errors = {}
    error = None
    result = None
    used_pcr_direct = False
    putcall_source = None

    # 기본 티커: SPY/^VIX 는 v1 에서 이미 캐시되어 재호출이 캐시 히트로 빠르다.
    # v2 전용 추가 티커는 RSP/HYG/LQD/TLT/^VIX3M 만 조회한다.
    base_tickers = ["SPY", "^VIX", "RSP", "HYG", "LQD", "TLT", "^VIX3M"]
    # ^CPC/^CPCE 직접 조회는 try_pcr=True 일 때만 수행한다 (기본 비활성화).
    pcr_tickers = ["^CPC", "^CPCE"] if try_pcr else []

    raw = {}
    for tk in base_tickers + pcr_tickers:
        try:
            series = fetch_close_series(tk)
            if series is not None and not series.empty:
                raw[tk] = series
                ticker_ok.append(tk)
            else:
                raw[tk] = None
                ticker_errors[tk] = "빈 데이터 반환"
        except Exception as exc:  # noqa: BLE001 - 개별 티커 실패 격리
            raw[tk] = None
            ticker_errors[tk] = str(exc)[:120]

    # Put/Call 직접 데이터: try_pcr=True 일 때만 ^CPC → ^CPCE 순으로 사용.
    pcr_series = None
    if try_pcr:
        pcr_series = raw.get("^CPC")
        if pcr_series is None:
            pcr_series = raw.get("^CPCE")
        used_pcr_direct = pcr_series is not None and not pcr_series.empty

    try:
        result = compute_gorani_market_temperature_v2(
            spy_close=raw.get("SPY"),
            rsp_close=raw.get("RSP"),
            hyg_close=raw.get("HYG"),
            lqd_close=raw.get("LQD"),
            tlt_close=raw.get("TLT"),
            vix_close=raw.get("^VIX"),
            pcr_close=pcr_series,  # None 이면 logic 내부에서 VIX/VIX3M proxy 사용
            vix3m_close=raw.get("^VIX3M"),
            min_components=5,
        )
        comp = (result or {}).get("components") or {}
        putcall = comp.get("Put/Call") or {}
        putcall_source = putcall.get("tickers")
    except Exception:  # noqa: BLE001 - v2 진단은 절대 페이지를 막지 않음
        error = traceback.format_exc()

    elapsed = time.perf_counter() - start
    return {
        "result": result,
        "ticker_ok": ticker_ok,
        "ticker_errors": ticker_errors,
        "elapsed": elapsed,
        "error": error,
        "used_pcr_direct": used_pcr_direct,
        "putcall_source": putcall_source,
    }


def load_watchlist(tickers):
    """워치리스트 종가를 모아 (성공 dict, 실패 dict) 로 반환한다.

    한 종목이 실패해도 나머지는 정상 표시되도록 종목별로 격리한다.
    """
    closes = {}
    failures = {}
    for ticker in tickers:
        try:
            series = fetch_close_series(ticker)
            if series is None or series.empty:
                failures[ticker] = "데이터가 비어 있습니다."
                continue
            closes[ticker] = series
        except Exception as e:  # noqa: BLE001 - 페이지 전체 중단 방지
            failures[ticker] = str(e)
    return closes, failures


def _slice_recent(series: pd.Series, lookback) -> pd.Series:
    if lookback is None:
        return series
    if series is None or series.empty:
        return series
    return series.tail(int(lookback))


def _last_valid(series: pd.Series):
    if series is None:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


# ──────────────────────────────────────────────
# 3. 차트
# ──────────────────────────────────────────────
# 시장 심리 게이지(속도계) 구간 — 앱 톤에 맞춘 연한 색상
GAUGE_STEPS = [
    {"range": [0, 25], "color": "#FDECEA"},    # 극단적 공포 (연빨강)
    {"range": [25, 45], "color": "#FFF3E0"},   # 공포 (연주황)
    {"range": [45, 55], "color": "#F2F4F6"},   # 중립 (연회색)
    {"range": [55, 75], "color": "#E8F3FF"},   # 탐욕 (연파랑)
    {"range": [75, 100], "color": "#E5F6ED"},  # 극단적 탐욕 (연초록)
]


def build_sentiment_gauge(score, title: str, height: int = 300) -> go.Figure:
    """0~100 점수를 속도계(게이지) 형태로 표시하는 Plotly Indicator 를 만든다.

    score 가 None/NaN/비정상이어도 예외 없이 0 으로 처리한 figure 를 반환한다.
    """
    safe = 0.0
    try:
        if score is not None:
            safe = float(score)
            if safe != safe:  # NaN
                safe = 0.0
    except (TypeError, ValueError):
        safe = 0.0
    safe = max(0.0, min(100.0, safe))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=safe,
            number={"font": {"size": 34, "color": "#191F28"}},
            title={"text": title, "font": {"size": 15, "color": "#4E5968"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#8B95A1",
                    "tickvals": [0, 25, 45, 55, 75, 100],
                },
                "bar": {"color": "rgba(25,31,40,0.72)", "thickness": 0.28},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": GAUGE_STEPS,
                "threshold": {
                    "line": {"color": "#191F28", "width": 3},
                    "thickness": 0.75,
                    "value": safe,
                },
            },
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="#FFFFFF",
        font={"color": "#191F28"},
    )
    return fig


def build_rsi_chart(rsi_map: dict) -> go.Figure:
    fig = go.Figure()
    for ticker, rsi_series in rsi_map.items():
        if rsi_series is None or rsi_series.dropna().empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rsi_series.index,
                y=rsi_series.values,
                mode="lines",
                name=ticker,
                line=dict(color=TICKER_COLORS.get(ticker, "#3182F6"), width=2.0),
            )
        )

    # 과매수(70) / 과매도(30) 기준선
    fig.add_hline(y=70, line=dict(color="#D93D44", width=1.2, dash="dash"),
                  annotation_text="과매수 70", annotation_position="top left")
    fig.add_hline(y=30, line=dict(color="#1B64DA", width=1.2, dash="dash"),
                  annotation_text="과매도 30", annotation_position="bottom left")

    fig.update_layout(
        title=dict(text="RSI 14 추이", font=dict(size=18, color="#191F28")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(range=[0, 100], hoverformat=".1f"),
    )
    return fig


def build_drawdown_chart(dd_map: dict) -> go.Figure:
    fig = go.Figure()
    for ticker, dd_series in dd_map.items():
        if dd_series is None or dd_series.dropna().empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=dd_series.index,
                y=dd_series.values,
                mode="lines",
                name=ticker,
                line=dict(color=TICKER_COLORS.get(ticker, "#3182F6"), width=2.0),
            )
        )

    # -10% / -20% / -30% / -40% 기준선 (값은 비율)
    for level in (-0.10, -0.20, -0.30, -0.40):
        fig.add_hline(
            y=level,
            line=dict(color="#8B95A1", width=1.0, dash="dot"),
            annotation_text=f"{level:.0%}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text="고점 대비 하락률 추이", font=dict(size=18, color="#191F28")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(tickformat=".0%", hoverformat=".2%"),
    )
    return fig


# ──────────────────────────────────────────────
# 4. 화면
# ──────────────────────────────────────────────
st.markdown("# 🌡️ 시장온도")
st.caption("QQQ · SCHD · SPY 의 RSI 14 와 고점 대비 하락률로 시장의 과열/침체 상태를 살펴봅니다.")

st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)

ctrl_col, _ = st.columns([1, 3])
with ctrl_col:
    range_label = st.selectbox("표시 기간", list(RANGE_OPTIONS.keys()), index=1)
lookback = RANGE_OPTIONS[range_label]

closes, failures = load_watchlist(WATCHLIST)

# 데이터 로드 결과 안내 (실패해도 페이지는 계속 렌더)
if failures:
    failed_names = ", ".join(failures.keys())
    if not closes:
        st.warning(
            f"⚠️ 시세 데이터를 불러오지 못했습니다 ({failed_names}). "
            "잠시 후 다시 시도하거나, 아래 캐시 초기화 버튼을 눌러주세요."
        )
    else:
        st.info(f"ℹ️ 일부 종목 데이터를 불러오지 못했습니다: {failed_names} (나머지는 정상 표시됩니다.)")

# RSI / 하락률은 전체 종가로 미리 계산 (심리 요약 + 차트에서 함께 사용)
rsi_full = {t: compute_rsi(s, period=14) for t, s in closes.items()}
dd_full = {t: compute_drawdown_series(s) for t, s in closes.items()}


# ──────────────────────────────────────────────
# 4-1. 시장 심리 요약 (고라니 시장온도 / VIX)
# ──────────────────────────────────────────────
vix_value = None
vix_series = None
vix_failed = False
try:
    vix_series = fetch_close_series("^VIX")
    vix_value = _last_valid(vix_series)
    if vix_value is None:
        vix_failed = True
except Exception:  # noqa: BLE001 - VIX 실패는 경고만, 페이지는 유지
    vix_failed = True

gorani = compute_gorani_market_temperature(
    qqq_rsi=_last_valid(rsi_full.get("QQQ")),
    spy_rsi=_last_valid(rsi_full.get("SPY")),
    qqq_drawdown=_last_valid(dd_full.get("QQQ")),
    spy_drawdown=_last_valid(dd_full.get("SPY")),
    spy_ma_distance=compute_distance_from_moving_average(closes.get("SPY"), 200),
    vix_level=vix_value,
)
gorani_score = gorani["score"]

st.markdown("#### 시장 심리 요약")

g_main, g_side = st.columns([1.3, 1])

with g_main:
    if gorani_score is not None:
        st.plotly_chart(
            build_sentiment_gauge(gorani_score, "고라니 시장온도"),
            use_container_width=True,
        )
        st.caption(f"상태: {classify_fear_greed_score(gorani_score)}")
    else:
        st.info("시장 심리 점수를 산출할 데이터가 아직 없습니다.")

    # 게이지 구간 범례 (모바일에서도 안전한 작은 텍스트)
    st.markdown(
        "<div style='font-size:11px; color:#6b7684; text-align:center; margin-top:-8px;'>"
        "극단적 공포 0–25 · 공포 25–45 · 중립 45–55 · 탐욕 55–75 · 극단적 탐욕 75–100"
        "</div>",
        unsafe_allow_html=True,
    )

    # 자체 지표 설명 및 한계 안내
    st.caption(
        "고라니 시장온도는 자체 계산 지표이며, CNN Fear & Greed 공식값과 다를 수 있습니다. "
        "현재 v1은 RSI·고점대비 하락률·VIX 등 가격·변동성 중심 지표입니다."
    )
    st.markdown(
        "<div style='font-size:13px;'>🔗 "
        "<a href='https://www.cnn.com/markets/fear-and-greed' "
        "target='_blank' rel='noopener noreferrer'>"
        "CNN Fear & Greed 공식 페이지에서 확인</a></div>",
        unsafe_allow_html=True,
    )

with g_side:
    # VIX 보조 카드 (메인 판단 지표가 아닌 참고용)
    if vix_value is not None:
        st.metric("현재 VIX", f"{vix_value:.1f}")
        st.caption("참고: VIX가 높을수록 시장 불안 심리가 큼")
    else:
        st.metric("현재 VIX", "N/A")
        st.caption("VIX 조회 실패")

if vix_failed:
    st.warning("⚠️ VIX(^VIX) 조회에 실패했습니다. 나머지 지표는 정상 표시됩니다.")

if gorani["components"]:
    with st.expander("고라니 시장온도 구성요소 보기", expanded=False):
        comp_text = " · ".join(
            f"{name} {value:.0f}" for name, value in gorani["components"].items()
        )
        st.caption(
            f"가용 지표 {len(gorani['components'])}개 평균 · {comp_text}. "
            "RSI↑ · 하락폭↓ · 200일선 위 · VIX↓ 일수록 탐욕(점수↑)으로 해석합니다."
        )
        st.caption(
            "ℹ️ 현재 v1은 가격·변동성 중심이라 모멘텀 장세에서 CNN 공식값보다 높게 나올 수 있습니다. "
            "향후 PCR, 시장폭(RSP/SPY), 신용위험(HYG/LQD), 안전자산 선호(SPY vs TLT)를 반영한 v2로 개선 예정입니다."
        )

st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 4-2. RSI 14 / 고점 대비 하락률 (기존 MVP 유지)
# ──────────────────────────────────────────────
if closes:
    rsi_view = {t: _slice_recent(s, lookback) for t, s in rsi_full.items()}
    dd_view = {t: _slice_recent(s, lookback) for t, s in dd_full.items()}

    # ── 현재 RSI 14 카드 ──
    st.markdown("#### 현재 RSI 14")
    rsi_cols = st.columns(len(WATCHLIST))
    for col, ticker in zip(rsi_cols, WATCHLIST):
        value = _last_valid(rsi_full.get(ticker))
        if value is None:
            col.metric(ticker, "N/A")
        else:
            if value >= 70:
                state = "과매수"
            elif value <= 30:
                state = "과매도"
            else:
                state = "중립"
            col.metric(ticker, f"{value:.1f}", help=f"현재 상태: {state}")

    st.plotly_chart(build_rsi_chart(rsi_view), use_container_width=True)

    # ── 현재 고점 대비 하락률 카드 ──
    st.markdown("#### 현재 고점 대비 하락률")
    dd_cols = st.columns(len(WATCHLIST))
    for col, ticker in zip(dd_cols, WATCHLIST):
        value = _last_valid(dd_full.get(ticker))
        if value is None:
            col.metric(ticker, "N/A")
        else:
            col.metric(ticker, f"{value:.1%}")

    st.plotly_chart(build_drawdown_chart(dd_view), use_container_width=True)

    st.caption(
        "ℹ️ RSI 14 는 Wilder 방식으로 직접 계산하며, 70 이상은 과매수·30 이하는 과매도 신호로 해석합니다. "
        "고점 대비 하락률은 표시 구간 이전을 포함한 전체 고점 기준입니다."
    )


# ──────────────────────────────────────────────
# 4-3. 고라니 시장온도 v2 (진단용) — lazy 계산. 버튼을 눌렀을 때만 계산한다.
#      v2 는 아직 메인 게이지에 연결하지 않는다 (검증 단계).
#      st.expander 는 접혀 있어도 내부 코드가 실행되므로, 무거운 fetch 는
#      반드시 버튼 + st.session_state 조건으로만 호출되도록 격리한다.
# ──────────────────────────────────────────────
with st.expander("🔬 고라니 시장온도 v2 진단 보기", expanded=False):
    # ──────────────────────────────────────────
    # A. 항상 표시되는 블록 (네트워크/계산 이전에 즉시 렌더)
    # ──────────────────────────────────────────
    st.caption("✅ 진단 블록 렌더링됨 (v2 diagnostics UI: lazy-v1)")
    st.caption(
        "v2는 CNN Fear & Greed의 7요소 철학을 참고한 진단용 자체 지표입니다. "
        "현재는 검증 단계이므로 메인 게이지에는 아직 반영하지 않습니다. "
        "아래 버튼을 눌러야만 계산하며, 페이지 로딩 시에는 자동으로 계산하지 않습니다."
    )

    # 실험적 직접 조회 옵션 (기본 False → ^CPC/^CPCE 는 조회하지 않음)
    v2_try_pcr = st.checkbox(
        "실험적으로 ^CPC/^CPCE 직접 조회 시도",
        value=False,
        key="v2_try_pcr",
        help="체크하지 않으면 Put/Call 은 VIX/VIX3M proxy 만 사용합니다 (권장).",
    )

    v2_btn_col1, v2_btn_col2 = st.columns(2)
    v2_run = v2_btn_col1.button(
        "📊 v2 진단 계산하기", use_container_width=True, key="v2_run_btn"
    )
    v2_clear = v2_btn_col2.button(
        "🧹 v2 진단 캐시 초기화", use_container_width=True, key="v2_clear_btn"
    )

    # ──────────────────────────────────────────
    # B. 캐시 초기화 버튼
    # ──────────────────────────────────────────
    if v2_clear:
        compute_v2_diagnostics.clear()
        st.session_state.pop("v2_diag", None)
        st.info("v2 진단 캐시를 초기화했습니다. '계산하기'를 다시 누르면 새로 계산합니다.")

    # ──────────────────────────────────────────
    # C. 계산 버튼 — 오직 이 분기에서만 무거운 fetch/계산이 실행된다.
    # ──────────────────────────────────────────
    if v2_run:
        with st.spinner("v2 진단 계산 중... (신규 티커 조회는 처음엔 느릴 수 있습니다)"):
            st.session_state["v2_diag"] = compute_v2_diagnostics(try_pcr=v2_try_pcr)

    diag = st.session_state.get("v2_diag")

    # ──────────────────────────────────────────
    # D. 결과 렌더 (계산 전이면 안내만, 계산 후이면 상세 표시)
    # ──────────────────────────────────────────
    if diag is None:
        st.info("아직 계산하지 않았습니다. 버튼을 누르면 v2 진단을 계산합니다.")
    else:
        result = diag.get("result") if isinstance(diag.get("result"), dict) else {}
        comp_data = result.get("components") or {}
        v2_score = result.get("score")
        v2_avail = result.get("available_components", 0)
        v2_min = result.get("min_components", 5)
        v2_ticker_ok = diag.get("ticker_ok") or []
        v2_ticker_errors = diag.get("ticker_errors") or {}
        v2_error = diag.get("error")
        elapsed = diag.get("elapsed")

        if v2_error is not None:
            v2_status = "error"
        else:
            v2_status = result.get("status", "error")

        # 계산 시간 / Put/Call 소스 / 직접 조회 사용 여부
        elapsed_text = f"{elapsed:.2f}초" if isinstance(elapsed, (int, float)) else "N/A"
        st.caption(
            f"⏱ 계산 시간: {elapsed_text} · "
            f"Put/Call 소스: {diag.get('putcall_source') or 'N/A'} · "
            f"^CPC/^CPCE 직접 사용: {'예' if diag.get('used_pcr_direct') else '아니오'}"
        )

        # v1 vs v2 메트릭 (v2 는 진단용 — 메인 게이지에 연결하지 않음)
        col_v1, col_v2, col_info = st.columns(3)
        col_v1.metric(
            "현재 메인 v1", f"{gorani_score:.1f}" if gorani_score is not None else "N/A"
        )
        if v2_status == "ok" and v2_score is not None:
            col_v2.metric("진단용 v2", f"{v2_score:.1f}", help=f"상태: {result.get('label')}")
        elif v2_status == "insufficient_data":
            col_v2.metric("진단용 v2", "데이터 부족")
        else:
            col_v2.metric("진단용 v2", "오류")
        col_info.metric("유효 구성요소", f"{v2_avail} / 7", help=f"최소 필요: {v2_min}개")

        st.caption(f"v2 상태: **{v2_status}** · 유효 {v2_avail}/{v2_min}개 이상 필요")

        # 실패 원인 안내
        if v2_status == "insufficient_data":
            st.info(f"유효 구성요소가 {v2_min}개 미만({v2_avail}개)이라 v2 점수를 산출하지 못했습니다.")
        elif v2_status == "error":
            st.warning("v2 계산 중 예외가 발생했습니다. 아래 구성요소 표/Traceback을 확인하세요.")

        # 구성요소 7행 표 (comp_data 비어도 N/A 7행)
        comp_rows = []
        for name in ("Momentum", "Price Strength", "Breadth", "Put/Call",
                     "Junk Bond", "Volatility", "Safe Haven"):
            info = comp_data.get(name, {}) if isinstance(comp_data, dict) else {}
            raw_val = info.get("raw")
            score_val = info.get("score")
            comp_rows.append({
                "구성요소": name,
                "사용 티커": info.get("tickers", ""),
                "raw 최신값": f"{raw_val:.4f}" if isinstance(raw_val, (int, float)) else "N/A",
                "score (0~100)": f"{score_val:.1f}" if isinstance(score_val, (int, float)) else "N/A",
                "상태": info.get("status", "na"),
                "note": "" if info else "데이터 없음",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.caption(
            "각 구성요소는 과거 252거래일 대비 rolling percentile rank(0~100)로 점수화합니다. "
            "공포 방향 지표(Put/Call, Volatility)는 100−분위로 반전합니다."
        )

        # 티커 조회 결과
        st.caption(
            f"성공 티커: {', '.join(v2_ticker_ok) if v2_ticker_ok else '없음'}"
        )
        if v2_ticker_errors:
            failed_tks = ", ".join(f"{tk}({err})" for tk, err in v2_ticker_errors.items())
            st.caption(f"⚠️ 티커 조회 실패: {failed_tks}")

        # 오류 Traceback (있을 때)
        if v2_error:
            st.markdown("**오류 Traceback (마지막 800자)**")
            st.code(v2_error[-800:])

        # 원시 v2 result dict 보기
        with st.expander("🔧 원시 v2 result 보기", expanded=False):
            if not result:
                st.write("v2 result 가 비어 있습니다.")
            else:
                try:
                    st.json({
                        "score": v2_score,
                        "label": result.get("label"),
                        "status": result.get("status"),
                        "available_components": v2_avail,
                        "min_components": v2_min,
                        "elapsed": elapsed,
                        "used_pcr_direct": diag.get("used_pcr_direct"),
                        "putcall_source": diag.get("putcall_source"),
                        "ticker_ok": v2_ticker_ok,
                        "ticker_errors": v2_ticker_errors,
                        "components": {
                            k: {kk: (vv if isinstance(vv, (int, float, type(None))) else str(vv))
                                for kk, vv in v.items()}
                            for k, v in comp_data.items()
                        } if comp_data else {},
                    })
                except Exception as _je:  # noqa: BLE001
                    st.write(f"원시 dict 표시 실패: {_je}")
                    st.write(result)

        # 배포 반영 확인용 버전 마커 (하단)
        st.caption("v2 diagnostics UI: lazy-v1")


# 캐시 초기화 (시세/심리 데이터가 일시적으로 비어 있을 때 수동 갱신용)
if st.button("🔄 시세 캐시 초기화", use_container_width=True):
    fetch_close_series.clear()
    st.rerun()


# ──────────────────────────────────────────────
# 5. 시장온도 참고 시트 (구글 스프레드시트 임베드) — 탭 최하단
#    - 구글 시트를 iframe 으로 "보기"만 한다 (Google API/secrets/pandas 미사용).
#    - 시트가 로딩되지 않아도 위 게이지/RSI/하락률/v2 진단 화면은 영향을 받지 않는다.
# ──────────────────────────────────────────────
st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)
st.markdown("### 📊 시장온도 참고 시트")

# 구글 시트 '웹에 게시 → 삽입(Embed)' URL.
# HTML 원본의 &amp; 는 Python URL 문자열에서 일반 & 로 사용한다.
sheet_url = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRQsjM2Yp05NyPTnXEeUuHrO8oiOJhuRmtDqIFQHOrsAGNnxVHDvs8eg0_qS-6CR5mnAG29v02j-fJ7/"
    "pubhtml?gid=331043462&single=true&widget=true&headers=false"
)

st.caption(
    "구글 시트가 보이지 않으면 아래 '새 탭에서 열기' 링크로 열거나, "
    "시트의 '웹에 게시(Publish to web)' 설정을 확인해주세요."
)

# iframe 임베드 (높이 800, 스크롤 허용). 외부 시트 로딩 실패는 iframe 내부 문제로
# 한정되며 Streamlit 앱 전체를 중단시키지 않는다.
components.iframe(sheet_url, height=800, scrolling=True)

# 새 탭에서 열기 링크 (iframe 이 막혀도 사용자가 직접 확인할 수 있도록 제공)
st.markdown(
    f"<div style='font-size:13px; margin-top:8px;'>🔗 "
    f"<a href='{sheet_url}' target='_blank' rel='noopener noreferrer'>"
    "새 탭에서 시장온도 참고 시트 열기</a></div>",
    unsafe_allow_html=True,
)
