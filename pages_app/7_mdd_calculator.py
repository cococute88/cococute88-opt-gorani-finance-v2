import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
from io import StringIO
from datetime import date, timedelta

from ui.styles import TOSS_CSS
from logic.market import (
    compute_drawdown_series,
    compute_mdd_details,
    align_and_convert_to_krw,
)

# ──────────────────────────────────────────────
# 1. 디자인
# ──────────────────────────────────────────────
st.markdown(TOSS_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2. 데이터 헬퍼 (4·6번 페이지와 동일한 fallback 패턴을 안전하게 복제)
#    - 페이지 파일을 import 하면 스크립트가 실행되므로 직접 import 하지 않는다.
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
def fetch_usdkrw_series() -> pd.Series:
    """USD/KRW(1달러당 원화) 환율 종가 Series 를 반환한다.

    KRW=X → USDKRW=X 순으로 시도하고, '1달러당 원화' 값이 통상 범위(대략
    1,000~2,000원대)를 크게 벗어나면 비정상으로 보고 실패 처리한다.
    (반전/이상치 방지로 700~3,000 허용) 모두 실패 시 예외를 던진다.
    """
    errors = []
    for symbol in ("KRW=X", "USDKRW=X"):
        try:
            series = fetch_close_series(symbol).dropna()
        except Exception as e:
            errors.append(f"{symbol}: {e}")
            continue
        if series.empty:
            errors.append(f"{symbol}: 데이터 없음")
            continue
        latest = float(series.iloc[-1])
        if not (700.0 <= latest <= 3000.0):
            errors.append(f"{symbol}: 비정상 환율값 {latest:.4f}")
            continue
        return series

    raise ValueError("USD/KRW 환율 조회 실패 | " + " | ".join(errors))


# ──────────────────────────────────────────────
# 3. 표시 헬퍼
# ──────────────────────────────────────────────
def _fmt_price(value):
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def _fmt_krw(value):
    if value is None:
        return "N/A"
    return f"₩{value:,.0f}"


def _fmt_pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _fmt_date(value):
    if value is None:
        return "N/A"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return "N/A"


def _price_at(series: pd.Series, label):
    if series is None or label is None:
        return None
    try:
        value = series.loc[label]
    except Exception:
        return None
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────
# 4. 차트
# ──────────────────────────────────────────────
def build_price_chart(window: pd.Series, details: dict, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=window.index, y=window.values, mode="lines",
            name=f"{ticker} 종가", line=dict(color="#3182F6", width=2.0),
        )
    )

    peak_date = details.get("peak_date")
    trough_date = details.get("trough_date")
    recovery_date = details.get("recovery_date")

    if peak_date is not None:
        fig.add_trace(
            go.Scatter(
                x=[peak_date], y=[_price_at(window, peak_date)], mode="markers",
                name="MDD 고점", marker=dict(color="#00875A", size=11, symbol="triangle-up"),
            )
        )
    if trough_date is not None:
        fig.add_trace(
            go.Scatter(
                x=[trough_date], y=[_price_at(window, trough_date)], mode="markers",
                name="MDD 저점", marker=dict(color="#D93D44", size=11, symbol="triangle-down"),
            )
        )
    if recovery_date is not None:
        fig.add_trace(
            go.Scatter(
                x=[recovery_date], y=[_price_at(window, recovery_date)], mode="markers",
                name="회복일", marker=dict(color="#FF8B00", size=11, symbol="circle"),
            )
        )

    fig.update_layout(
        title=dict(text=f"{ticker} 달러 기준 가격", font=dict(size=18, color="#191F28")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(hoverformat="$,.2f"),
    )
    return fig


def build_drawdown_chart(dd_series: pd.Series, details: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_series.index, y=dd_series.values, mode="lines",
            name="Drawdown", line=dict(color="#3182F6", width=2.0),
            fill="tozeroy", fillcolor="rgba(49,130,246,0.08)",
        )
    )

    # 0 기준선
    fig.add_hline(y=0, line=dict(color="#4E5968", width=1.0))
    # -10% / -20% / -30% / -40% 기준선
    for level in (-0.10, -0.20, -0.30, -0.40):
        fig.add_hline(
            y=level,
            line=dict(color="#8B95A1", width=1.0, dash="dot"),
            annotation_text=f"{level:.0%}",
            annotation_position="bottom right",
        )

    # 최대 MDD 지점 표시
    trough_date = details.get("trough_date")
    mdd = details.get("mdd")
    if trough_date is not None and mdd is not None:
        fig.add_trace(
            go.Scatter(
                x=[trough_date], y=[mdd], mode="markers",
                name=f"최대 MDD ({mdd * 100:.2f}%)",
                marker=dict(color="#D93D44", size=12, symbol="x"),
            )
        )

    fig.update_layout(
        title=dict(text="고점 대비 하락률 (Drawdown / MDD)", font=dict(size=18, color="#191F28")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(tickformat=".0%", hoverformat=".2%"),
    )
    return fig


def build_dd_compare_chart(usd_dd: pd.Series, krw_dd: pd.Series) -> go.Figure:
    fig = go.Figure()
    if usd_dd is not None and not usd_dd.dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=usd_dd.index, y=usd_dd.values, mode="lines",
                name="달러 기준", line=dict(color="#3182F6", width=2.0),
            )
        )
    if krw_dd is not None and not krw_dd.dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=krw_dd.index, y=krw_dd.values, mode="lines",
                name="원화 기준", line=dict(color="#FF8B00", width=2.0),
            )
        )

    fig.add_hline(y=0, line=dict(color="#4E5968", width=1.0))
    for level in (-0.10, -0.20, -0.30, -0.40):
        fig.add_hline(
            y=level,
            line=dict(color="#8B95A1", width=1.0, dash="dot"),
            annotation_text=f"{level:.0%}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text="달러 vs 원화 Drawdown 비교", font=dict(size=18, color="#191F28")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(tickformat=".0%", hoverformat=".2%"),
    )
    return fig


# ──────────────────────────────────────────────
# 5. 화면 및 입력
# ──────────────────────────────────────────────
st.markdown("# 📉 MDD 계산기")
st.caption("MDD는 선택한 기간 동안 고점 대비 최대 하락률을 계산합니다. 달러 기준과 원화 환산 기준을 함께 확인할 수 있습니다.")

st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)

st.session_state.setdefault("mdd_ticker_input", "QQQ")
st.session_state.setdefault("mdd_start_date", date.today() - timedelta(days=365 * 5))
st.session_state.setdefault("mdd_end_date", date.today())

c1, c2, c3 = st.columns([1.4, 1, 1])
with c1:
    ticker_raw = st.text_input("티커", key="mdd_ticker_input")
with c2:
    start_d = st.date_input("시작일", key="mdd_start_date", format="YYYY/MM/DD")
with c3:
    end_d = st.date_input("종료일", key="mdd_end_date", format="YYYY/MM/DD")

ticker = (ticker_raw or "").strip().upper()

# 입력 방어 처리 (앱이 죽지 않도록)
valid_input = True
if not ticker:
    st.info("ℹ️ 분석할 티커를 입력해주세요. (예: QQQ, SMH, SPY)")
    valid_input = False
if start_d > end_d:
    st.error("⚠️ 시작일이 종료일보다 늦습니다. 날짜를 다시 선택해주세요.")
    valid_input = False

if valid_input:
    # ── 데이터 조회 (실패해도 페이지는 유지) ──
    full_close = None
    try:
        full_close = fetch_close_series(ticker)
    except Exception as e:
        st.warning(
            f"⚠️ '{ticker}' 시세를 불러오지 못했습니다. 티커 철자나 네트워크를 확인해주세요. "
            "(아래 캐시 초기화 후 재시도 가능)"
        )
        st.caption(f"상세: {e}")

    if full_close is not None and not full_close.empty:
        window = full_close.loc[pd.Timestamp(start_d):pd.Timestamp(end_d)]

        if window.empty:
            st.warning("⚠️ 선택한 기간에 해당하는 데이터가 없습니다. 기간을 더 넓게 조정해주세요.")
        else:
            details = compute_mdd_details(window)
            dd_series = compute_drawdown_series(window)

            used_start = _fmt_date(window.index.min())
            used_end = _fmt_date(window.index.max())
            st.success(f"✅ 분석 기간: **{used_start} ~ {used_end}**  ·  데이터 {len(window)}일")

            # ── 결과 카드 (2행 x 3) ──
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("현재가", _fmt_price(details["current_price"]))
            r1c2.metric("기간 내 최고가", _fmt_price(details["period_high"]))
            r1c3.metric("현재 고점대비 하락률", _fmt_pct(details["current_drawdown"]))

            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.metric("최대 MDD", _fmt_pct(details["mdd"]))
            r2c2.metric("MDD 고점일", _fmt_date(details["peak_date"]),
                        help=f"고점가 {_fmt_price(details['peak_price'])}")
            if details["recovered"]:
                recovery_help = "저점 이후 고점가 이상으로 회복한 날"
                r2c3.metric("MDD 저점일 → 회복일",
                            f"{_fmt_date(details['trough_date'])} → {_fmt_date(details['recovery_date'])}",
                            help=recovery_help)
            else:
                r2c3.metric("MDD 저점일 (회복 여부)",
                            f"{_fmt_date(details['trough_date'])} · 미회복",
                            help=f"저점가 {_fmt_price(details['trough_price'])}")

            # 회복 소요기간 안내
            if details["recovered"] and details["peak_date"] is not None and details["recovery_date"] is not None:
                try:
                    days = (pd.Timestamp(details["recovery_date"]) - pd.Timestamp(details["peak_date"])).days
                    st.caption(f"⏱️ 고점({_fmt_date(details['peak_date'])})에서 회복까지 약 **{days}일** 소요되었습니다.")
                except Exception:
                    pass
            elif details["mdd"] is not None and not details["recovered"]:
                st.caption("⏱️ 아직 직전 고점가를 회복하지 못했습니다. (미회복)")

            # ── 그래프 ──
            st.plotly_chart(build_price_chart(window, details, ticker), use_container_width=True)
            st.plotly_chart(build_drawdown_chart(dd_series, details), use_container_width=True)

            st.caption(
                "ℹ️ MDD(최대낙폭) = 기간 내 고점 대비 최대 하락률입니다. "
                "기준선은 -10% / -20% / -30% / -40% 입니다. 아래에 원화 기준 비교를 함께 제공합니다."
            )

            # ──────────────────────────────────────────
            # 원화(KRW) 기준 MDD — 환율 실패와 무관하게 위 달러 결과는 항상 유지
            # ──────────────────────────────────────────
            st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)
            st.markdown("### 🇰🇷 원화 기준 MDD (한국 투자자 체감)")
            st.caption(
                "원화 기준 MDD는 달러 종가에 USD/KRW 환율을 곱해 계산한 한국 투자자 체감 기준입니다. "
                "환율 효과 때문에 달러 기준 MDD와 다를 수 있으며, 공식 금융사이트의 달러 기준 MDD와 다를 수 있습니다."
            )

            fx_full = None
            fx_failed = False
            fx_error = ""
            try:
                fx_full = fetch_usdkrw_series()
            except Exception as e:
                fx_failed = True
                fx_error = str(e)

            if fx_failed or fx_full is None or fx_full.empty:
                st.warning(
                    "⚠️ USD/KRW 환율을 불러오지 못해 원화 기준 분석을 생략합니다. "
                    "위 달러 기준 결과/그래프는 그대로 유효합니다."
                )
                if fx_error:
                    st.caption(f"상세: {fx_error}")
            else:
                krw_window, aligned_rate = align_and_convert_to_krw(window, fx_full)

                if krw_window.empty or len(krw_window) < 2:
                    st.warning(
                        "⚠️ 선택 구간에 유효한 환율 데이터가 부족하여 원화 기준 분석을 생략합니다. "
                        "기간을 조정하거나 잠시 후 다시 시도해주세요. (달러 기준 결과는 위에 유지됩니다.)"
                    )
                else:
                    krw_details = compute_mdd_details(krw_window)
                    krw_dd = compute_drawdown_series(krw_window)

                    # 적용 환율 정보 카드
                    latest_rate = _price_at(aligned_rate, aligned_rate.index.max())
                    rate_date = _fmt_date(aligned_rate.index.max())
                    fx1, fx2 = st.columns(2)
                    fx1.metric(
                        "적용 USD/KRW (최신)",
                        f"{latest_rate:,.2f}원" if latest_rate is not None else "N/A",
                        help=f"환율 기준일: {rate_date}",
                    )
                    fx2.metric(
                        "원화 환산 데이터", f"{len(krw_window)}일",
                        help="달러 거래일에 환율을 reindex 후 ffill 정렬하여 환산했습니다.",
                    )

                    # 달러/원화 비교 표
                    compare_df = pd.DataFrame(
                        {
                            "달러 기준": [
                                _fmt_price(details["current_price"]),
                                _fmt_price(details["period_high"]),
                                _fmt_pct(details["current_drawdown"]),
                                _fmt_pct(details["mdd"]),
                                _fmt_date(details["peak_date"]),
                                _fmt_date(details["trough_date"]),
                                _fmt_date(details["recovery_date"]) if details["recovered"] else "미회복",
                            ],
                            "원화 기준": [
                                _fmt_krw(krw_details["current_price"]),
                                _fmt_krw(krw_details["period_high"]),
                                _fmt_pct(krw_details["current_drawdown"]),
                                _fmt_pct(krw_details["mdd"]),
                                _fmt_date(krw_details["peak_date"]),
                                _fmt_date(krw_details["trough_date"]),
                                _fmt_date(krw_details["recovery_date"]) if krw_details["recovered"] else "미회복",
                            ],
                        },
                        index=[
                            "현재가", "기간 내 최고가", "현재 고점대비 하락률",
                            "최대 MDD", "MDD 고점일", "MDD 저점일", "회복일",
                        ],
                    )
                    st.table(compare_df)

                    # 달러/원화 drawdown 비교차트 (가격은 단위가 달라 중첩하지 않음)
                    st.plotly_chart(
                        build_dd_compare_chart(dd_series, krw_dd),
                        use_container_width=True,
                    )

                    st.caption(
                        "ℹ️ 비교 핵심은 drawdown 입니다. 하락장에서 원화가 약세(환율 상승)면 원화 기준 낙폭이 "
                        "달러 기준보다 작아질 수 있고, 원화가 강세면 더 커질 수 있습니다."
                    )

# 캐시 초기화 (시세/환율이 일시적으로 비어 있을 때 수동 갱신용)
if st.button("🔄 시세 캐시 초기화", use_container_width=True):
    fetch_close_series.clear()
    fetch_usdkrw_series.clear()
    st.rerun()
