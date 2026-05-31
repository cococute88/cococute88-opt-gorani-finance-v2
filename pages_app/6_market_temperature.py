import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
from io import StringIO

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
# 4-1. 시장 심리 요약 (고라니 시장온도 v2 / VIX)
# ──────────────────────────────────────────────
# v2 구성요소에 필요한 추가 종목 조회 (실패 시 해당 구성요소만 제외)
_v2_tickers = {"RSP": None, "HYG": None, "LQD": None, "TLT": None,
               "^VIX": None, "^CPC": None, "^VIX3M": None}
for _tk in _v2_tickers:
    try:
        _v2_tickers[_tk] = fetch_close_series(_tk)
    except Exception:  # noqa: BLE001
        pass

vix_series = _v2_tickers.get("^VIX")
vix_value = _last_valid(vix_series)
vix_failed = vix_value is None

# v2 계산
gorani_v2 = compute_gorani_market_temperature_v2(
    spy_close=closes.get("SPY"),
    rsp_close=_v2_tickers.get("RSP"),
    hyg_close=_v2_tickers.get("HYG"),
    lqd_close=_v2_tickers.get("LQD"),
    tlt_close=_v2_tickers.get("TLT"),
    vix_close=vix_series,
    pcr_close=_v2_tickers.get("^CPC"),
    vix3m_close=_v2_tickers.get("^VIX3M"),
    k=1.0,
    min_components=4,
)

# v1 fallback (v2 산출 불가 시 사용)
gorani_v1 = compute_gorani_market_temperature(
    qqq_rsi=_last_valid(rsi_full.get("QQQ")),
    spy_rsi=_last_valid(rsi_full.get("SPY")),
    qqq_drawdown=_last_valid(dd_full.get("QQQ")),
    spy_drawdown=_last_valid(dd_full.get("SPY")),
    spy_ma_distance=compute_distance_from_moving_average(closes.get("SPY"), 200),
    vix_level=vix_value,
)

# 메인 점수 결정
if gorani_v2["ok"] and gorani_v2["score"] is not None:
    main_score = gorani_v2["score"]
    main_version = "v2"
    main_components = gorani_v2["components"]
    main_count = gorani_v2["available_count"]
else:
    main_score = gorani_v1["score"]
    main_version = "v1"
    main_components = gorani_v1.get("components", {})
    main_count = len([v for v in main_components.values() if v is not None])

st.markdown("#### 시장 심리 요약")

g_main, g_side = st.columns([1.3, 1])

with g_main:
    if main_score is not None:
        st.plotly_chart(
            build_sentiment_gauge(main_score, "고라니 시장온도"),
            use_container_width=True,
        )
        st.caption(f"상태: {classify_fear_greed_score(main_score)}")
    else:
        st.info("시장 심리 점수를 산출할 데이터가 아직 없습니다.")

    # 게이지 구간 범례
    st.markdown(
        "<div style='font-size:11px; color:#6b7684; text-align:center; margin-top:-8px;'>"
        "극단적 공포 0–25 · 공포 25–45 · 중립 45–55 · 탐욕 55–75 · 극단적 탐욕 75–100"
        "</div>",
        unsafe_allow_html=True,
    )

    # 자체 지표 설명
    st.caption(
        "고라니 시장온도 v2는 CNN Fear & Greed의 7요소 철학을 참고한 자체 계산 지표입니다. "
        "CNN 공식값과 완전히 일치하지 않을 수 있습니다."
    )
    st.markdown(
        "<div style='font-size:13px;'>🔗 "
        "<a href='https://www.cnn.com/markets/fear-and-greed' "
        "target='_blank' rel='noopener noreferrer'>"
        "CNN Fear & Greed 공식 페이지에서 확인</a></div>",
        unsafe_allow_html=True,
    )

with g_side:
    # VIX 보조 카드
    if vix_value is not None:
        st.metric("현재 VIX", f"{vix_value:.1f}")
        st.caption("참고: VIX가 높을수록 시장 불안 심리가 큼")
    else:
        st.metric("현재 VIX", "N/A")
        st.caption("VIX 조회 실패")

if vix_failed:
    st.warning("⚠️ VIX(^VIX) 조회에 실패했습니다. 나머지 지표는 정상 표시됩니다.")

if main_version == "v1":
    st.info(
        "ℹ️ v2 구성요소가 부족하여 v1 fallback을 표시하고 있습니다. "
        "(v1은 RSI·하락률·VIX 기반으로, 상승장에서 높게 나올 수 있습니다.)"
    )

# 구성요소 expander
with st.expander(f"고라니 시장온도 {main_version} 구성요소 ({main_count}개 사용)", expanded=False):
    if main_version == "v2":
        v2_labels = ["Momentum", "Price Strength", "Breadth", "Put/Call",
                     "Junk Bond", "Volatility", "Safe Haven"]
        for name in v2_labels:
            score = main_components.get(name)
            if score is not None:
                st.caption(f"• {name}: {score:.0f}")
            else:
                st.caption(f"• {name}: N/A (데이터 없음)")
        st.caption(
            "각 구성요소는 252일 rolling z-score를 sigmoid로 0~100 점수화합니다. "
            "z=0(과거 평균)이면 50점, 탐욕 방향일수록 100에 가깝습니다."
        )
    else:
        comp_text = " · ".join(
            f"{name} {value:.0f}" for name, value in main_components.items()
            if value is not None
        )
        st.caption(f"v1 구성요소: {comp_text}")
        st.caption("RSI↑ · 하락폭↓ · 200일선 위 · VIX↓ 일수록 탐욕(점수↑)")

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

# 캐시 초기화 (시세/심리 데이터가 일시적으로 비어 있을 때 수동 갱신용)
if st.button("🔄 시세 캐시 초기화", use_container_width=True):
    fetch_close_series.clear()
    st.rerun()
