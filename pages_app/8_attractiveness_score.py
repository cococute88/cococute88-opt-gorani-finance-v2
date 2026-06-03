import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from ui.styles import TOSS_CSS


TICKER = "SCHD"
CACHE_TTL_SECONDS = 60 * 60 * 12
TARGET_YIELDS = [0.035, 0.037, 0.038]
PERIOD_OPTIONS = {
    "1M": pd.DateOffset(months=1),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
    "10Y": pd.DateOffset(years=10),
}
DEFAULT_PERIOD = "5Y"
ORANGE = "#f2994a"


# ──────────────────────────────────────────────
# 1. Formatting helpers
# ──────────────────────────────────────────────
def _format_currency(value) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "-"
    return f"${float(value):,.2f}"


def _format_percent(value, digits: int = 2) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "-"
    return f"{float(value):,.{digits}f}%"


# ──────────────────────────────────────────────
# 2. Data loading and calculation
# ──────────────────────────────────────────────
def _to_naive_normalized_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.to_datetime(index, errors="coerce")
    if getattr(dt_index, "tz", None) is not None:
        dt_index = dt_index.tz_localize(None)
    return dt_index.normalize()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_schd_history() -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Fetch SCHD adjusted price and dividend history from yfinance only."""
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=365 * 11 + 10)

    history = yf.Ticker(TICKER).history(
        start=start,
        end=end,
        auto_adjust=True,
        actions=True,
        repair=True,
        timeout=20,
    )

    if history is None or history.empty:
        raise ValueError("SCHD yfinance 조회 결과가 비어 있습니다.")

    data = history.copy()
    data.index = _to_naive_normalized_index(data.index)
    data = data[~data.index.isna()]
    data = data[~data.index.duplicated(keep="last")].sort_index()

    if "Close" not in data.columns:
        raise ValueError("SCHD 가격 데이터에 Close 컬럼이 없습니다.")

    price_df = pd.DataFrame(index=data.index)
    price_df["price"] = pd.to_numeric(data["Close"], errors="coerce")
    price_df = price_df.dropna(subset=["price"])
    price_df = price_df[price_df["price"] > 0]

    if price_df.empty:
        raise ValueError("SCHD 유효 가격 데이터가 없습니다.")

    if "Dividends" in data.columns:
        dividend_series = pd.to_numeric(data["Dividends"], errors="coerce").fillna(0)
    else:
        dividend_series = pd.Series(0.0, index=data.index)

    dividends_df = dividend_series[dividend_series > 0].to_frame("dividend")
    dividends_df.index.name = "date"
    price_df.index.name = "date"
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return price_df.astype("float64"), dividends_df.astype("float64"), fetched_at


def _calculate_ttm_dividend(price_index: pd.DatetimeIndex, dividends_df: pd.DataFrame) -> pd.Series:
    if dividends_df is None or dividends_df.empty:
        return pd.Series(np.nan, index=price_index, dtype="float64")

    dividends = dividends_df.copy().sort_index()
    dividends["dividend"] = pd.to_numeric(dividends["dividend"], errors="coerce")
    dividends = dividends.dropna(subset=["dividend"])
    dividends = dividends[dividends["dividend"] > 0]

    if dividends.empty:
        return pd.Series(np.nan, index=price_index, dtype="float64")

    div_dates = dividends.index.to_numpy(dtype="datetime64[ns]")
    div_values = dividends["dividend"].to_numpy(dtype="float64")
    cumulative = np.concatenate([[0.0], np.cumsum(div_values)])

    ttm_values = []
    for current_date in price_index:
        current_np = np.datetime64(current_date)
        start_np = np.datetime64(current_date - pd.Timedelta(days=365))
        left = np.searchsorted(div_dates, start_np, side="right")
        right = np.searchsorted(div_dates, current_np, side="right")
        ttm_values.append(cumulative[right] - cumulative[left])

    return pd.Series(ttm_values, index=price_index, dtype="float64")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def calculate_schd_dividend_yield() -> dict:
    price_df, dividends_df, fetched_at = fetch_schd_history()
    metrics = price_df.copy()
    metrics["ttm_dividend"] = _calculate_ttm_dividend(metrics.index, dividends_df)
    metrics["ttm_yield"] = np.where(
        (metrics["price"] > 0) & (metrics["ttm_dividend"] > 0),
        metrics["ttm_dividend"] / metrics["price"] * 100,
        np.nan,
    )
    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    valid = metrics.dropna(subset=["price", "ttm_dividend", "ttm_yield"])
    if valid.empty:
        raise ValueError("SCHD TTM 배당률을 계산할 수 없습니다.")

    latest_date = valid.index.max()
    latest = valid.loc[latest_date]
    current_price = float(latest["price"])
    recent_12m_dividend = float(latest["ttm_dividend"])
    current_ttm_yield = float(latest["ttm_yield"])

    if dividends_df is not None and not dividends_df.empty:
        latest_dividend = dividends_df.sort_index()["dividend"].dropna().iloc[-1]
        recent_quarter_dividend = float(latest_dividend) if latest_dividend > 0 else np.nan
    else:
        recent_quarter_dividend = np.nan

    five_year_start = latest_date - pd.DateOffset(years=5)
    five_year_yields = valid.loc[valid.index >= five_year_start, "ttm_yield"].dropna()
    five_year_average_yield = float(five_year_yields.mean()) if not five_year_yields.empty else np.nan

    target_rows = []
    for target_yield in TARGET_YIELDS:
        ttm_buy_price = recent_12m_dividend / target_yield if recent_12m_dividend > 0 else np.nan
        quarter_buy_price = (
            recent_quarter_dividend * 4 / target_yield
            if recent_quarter_dividend is not None and recent_quarter_dividend > 0
            else np.nan
        )
        drawdown = (ttm_buy_price / current_price - 1) * 100 if current_price > 0 and ttm_buy_price > 0 else np.nan
        target_rows.append(
            {
                "목표 배당률": f"{target_yield * 100:.1f}%",
                "TTM 기준 매수가": _format_currency(ttm_buy_price),
                "최근 분기×4 기준 매수가": _format_currency(quarter_buy_price),
                "현재가 대비 하락률": _format_percent(drawdown, 1),
            }
        )

    return {
        "metrics": metrics,
        "dividends": dividends_df,
        "latest_date": latest_date,
        "current_price": current_price,
        "current_ttm_yield": current_ttm_yield,
        "five_year_average_yield": five_year_average_yield,
        "recent_12m_dividend": recent_12m_dividend,
        "recent_quarter_dividend": recent_quarter_dividend,
        "target_table": pd.DataFrame(target_rows),
        "fetched_at": fetched_at,
    }


# ──────────────────────────────────────────────
# 3. UI builders
# ──────────────────────────────────────────────
def apply_schd_styles() -> None:
    st.markdown(TOSS_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            .schd-card-grid {
                display: grid;
                grid-template-columns: repeat(5, minmax(145px, 1fr));
                gap: 12px;
                margin: 18px 0 20px;
            }
            .schd-card {
                background: #ffffff;
                border: 1px solid #edf0f3;
                border-radius: 18px;
                padding: 16px 15px;
                box-shadow: 0 8px 22px rgba(25, 31, 40, 0.05);
            }
            .schd-card-label {
                color: #6b7684;
                font-size: 13px;
                font-weight: 650;
                margin-bottom: 7px;
                word-break: keep-all;
            }
            .schd-card-value {
                color: #191f28;
                font-size: 24px;
                font-weight: 800;
                letter-spacing: -0.02em;
            }
            .schd-card-sub {
                color: #8b95a1;
                font-size: 12px;
                margin-top: 6px;
            }
            .schd-judgement {
                background: #fff8ef;
                border: 1px solid #ffe0bd;
                border-radius: 16px;
                padding: 16px 18px;
                color: #4e5968;
                line-height: 1.65;
                margin: 14px 0 22px;
            }
            .schd-judgement b { color: #b85b00; }
            @media screen and (max-width: 1100px) {
                .schd-card-grid { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
            }
            @media screen and (max-width: 720px) {
                .schd-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .schd-card { padding: 14px 13px; }
                .schd-card-value { font-size: 20px; }
            }
            @media screen and (max-width: 420px) {
                .schd-card-grid { grid-template-columns: 1fr; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(data: dict) -> None:
    cards = [
        ("현재 SCHD 가격", _format_currency(data["current_price"]), f"기준일 {data['latest_date']:%Y-%m-%d}"),
        ("현재 TTM 배당률", _format_percent(data["current_ttm_yield"]), "최근 12개월 배당금 ÷ 현재가"),
        ("5년 평균 배당률", _format_percent(data["five_year_average_yield"]), "일별 TTM 배당률 평균"),
        ("최근 12개월 배당금", _format_currency(data["recent_12m_dividend"]), "최신일 기준 365일 합계"),
        ("최근 분기 배당금", _format_currency(data["recent_quarter_dividend"]), "가장 최근 1회 배당"),
    ]

    html = ['<div class="schd-card-grid">']
    for label, value, sub in cards:
        html.append(
            f"""
            <div class="schd-card">
                <div class="schd-card-label">{label}</div>
                <div class="schd-card-value">{value}</div>
                <div class="schd-card-sub">{sub}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _filter_period(metrics: pd.DataFrame, latest_date: pd.Timestamp, selected_period: str) -> pd.DataFrame:
    offset = PERIOD_OPTIONS.get(selected_period, PERIOD_OPTIONS[DEFAULT_PERIOD])
    start_date = latest_date - offset
    return metrics.loc[metrics.index >= start_date].copy()


def build_yield_chart(chart_df: pd.DataFrame, data: dict) -> go.Figure:
    hover_data = np.column_stack(
        [
            chart_df["ttm_dividend"].to_numpy(dtype="float64"),
            chart_df["price"].to_numpy(dtype="float64"),
        ]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["ttm_yield"],
            mode="lines",
            name="TTM 배당률",
            line=dict(color=ORANGE, width=3),
            customdata=hover_data,
            hovertemplate=(
                "날짜: %{x|%Y-%m-%d}<br>"
                "TTM 배당률: %{y:.2f}%<br>"
                "TTM 배당금: $%{customdata[0]:.2f}<br>"
                "가격: $%{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )

    reference_lines = [
        (data["current_ttm_yield"], "현재 배당률", "#f2994a"),
        (data["five_year_average_yield"], "5년 평균", "#3182f6"),
        (3.5, "3.5%", "#8b95a1"),
        (3.7, "3.7%", "#6b7684"),
        (3.8, "3.8%", "#4e5968"),
    ]
    for y_value, label, color in reference_lines:
        if y_value is None or pd.isna(y_value) or not np.isfinite(y_value):
            continue
        fig.add_hline(
            y=float(y_value),
            line_dash="dot",
            line_color=color,
            line_width=1.2,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color=color,
        )

    fig.update_layout(
        title="SCHD Dividend Yield TTM",
        height=470,
        margin=dict(l=20, r=20, t=58, b=32),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        xaxis_title="날짜",
        yaxis_title="배당률",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#191f28"),
    )
    fig.update_yaxes(ticksuffix="%", rangemode="tozero", gridcolor="#f2f4f6")
    fig.update_xaxes(gridcolor="#f8f9fa")
    return fig


def render_judgement(data: dict) -> None:
    current_yield = data["current_ttm_yield"]
    five_year_average = data["five_year_average_yield"]

    if pd.isna(five_year_average) or not np.isfinite(five_year_average):
        average_message = "5년 평균 배당률을 계산하기 위한 데이터가 충분하지 않습니다."
    elif current_yield > five_year_average:
        average_message = "현재 SCHD 배당률은 5년 평균보다 높습니다."
    elif current_yield < five_year_average:
        average_message = "현재 SCHD 배당률은 5년 평균보다 낮습니다."
    else:
        average_message = "현재 SCHD 배당률은 5년 평균과 같습니다."

    if current_yield >= 3.7:
        buy_message = "현재 배당률이 3.7% 이상입니다. SCHD 적극 매수 검토 구간입니다."
    elif current_yield >= 3.5:
        buy_message = "현재 배당률이 3.5% 이상입니다. SCHD 진입 가능 구간입니다."
    else:
        buy_message = "현재 배당률이 3.5% 미만입니다. 배당률 기준으로는 아직 보수적 접근 구간입니다."

    st.markdown(
        f"""
        <div class="schd-judgement">
            <div><b>5년 평균 비교</b> · {average_message}</div>
            <div><b>매수 판단</b> · {buy_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 4. Main page
# ──────────────────────────────────────────────
def main() -> None:
    apply_schd_styles()

    st.markdown("# 📈 SCHD 배당률 매수 판단")
    st.caption("yfinance의 SCHD 가격 데이터와 배당 이력만으로 TTM 배당률을 직접 계산합니다.")

    try:
        with st.spinner("SCHD 가격·배당 데이터를 불러오는 중입니다..."):
            data = calculate_schd_dividend_yield()
    except Exception as exc:
        st.error("SCHD 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        st.caption(f"오류 정보: {exc}")
        return

    if data["current_price"] <= 0 or data["recent_12m_dividend"] <= 0:
        st.warning("SCHD 가격 또는 최근 12개월 배당금이 0 이하로 계산되어 일부 지표가 제한될 수 있습니다.")

    if data["current_ttm_yield"] < 1 or data["current_ttm_yield"] > 8:
        st.warning("최신 TTM 배당률이 일반적인 점검 범위(1%~8%) 밖입니다. 가격·배당 조정 기준을 확인해 주세요.")

    render_metric_cards(data)
    render_judgement(data)

    selected_period = st.radio(
        "조회 기간",
        options=list(PERIOD_OPTIONS.keys()),
        index=list(PERIOD_OPTIONS.keys()).index(DEFAULT_PERIOD),
        horizontal=True,
        key="schd_yield_period",
    )

    chart_df = _filter_period(data["metrics"].dropna(subset=["ttm_yield"]), data["latest_date"], selected_period)
    if chart_df.empty:
        st.info("선택한 기간에 표시할 SCHD 배당률 데이터가 없습니다.")
    else:
        st.plotly_chart(build_yield_chart(chart_df, data), use_container_width=True)

    st.markdown("## 목표가 표")
    st.dataframe(data["target_table"], use_container_width=True, hide_index=True)

    with st.expander("계산 기준 보기"):
        st.markdown(
            """
            - 현재 SCHD 가격: yfinance 조정 종가(`Close`, `auto_adjust=True`)의 가장 최근 거래일 값
            - 최근 12개월 배당금: 최신 가격일 기준 과거 365일 동안 지급된 SCHD 배당금 합계
            - 각 날짜의 TTM 배당률: 해당 날짜 기준 과거 365일 배당금 합계 ÷ 해당 날짜 가격 × 100
            - 5년 평균 배당률: 최근 5년 구간의 일별 TTM 배당률 평균
            - TTM 기준 매수가: 최근 12개월 배당금 ÷ 목표 배당률
            - 최근 분기×4 기준 매수가: 최근 분기 배당금 × 4 ÷ 목표 배당률
            """
        )
        st.caption(f"데이터 소스: yfinance · 티커: {TICKER} · 캐시 TTL: {CACHE_TTL_SECONDS:,}초 · 조회 시각: {data['fetched_at']}")
        if st.button("🔄 SCHD 배당률 캐시 초기화", use_container_width=True):
            fetch_schd_history.clear()
            calculate_schd_dividend_yield.clear()
            st.rerun()


if not os.environ.get("GORANI_SKIP_PAGE_RENDER"):
    main()
