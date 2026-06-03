import importlib.util
import os
from datetime import datetime, timedelta, timezone

import streamlit as st


def _find_missing_required_packages() -> list[str]:
    """Return missing runtime packages before importing data/chart dependencies."""
    required_packages = ["pandas", "numpy", "plotly", "yfinance"]
    return [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]


_MISSING_REQUIRED_PACKAGES = _find_missing_required_packages()
if _MISSING_REQUIRED_PACKAGES:
    st.error(f"필수 패키지 누락: {', '.join(_MISSING_REQUIRED_PACKAGES)}")
    st.stop()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def _normalize_dividends_to_close_basis(data: pd.DataFrame) -> pd.DataFrame:
    """Return dividend events on the same split-adjusted basis as Yahoo Close.

    Yahoo's historical ``Close`` is the regular close series adjusted for stock
    splits, but not for dividend reinvestment.  That is the right denominator
    for a TTM yield chart.  Dividend events are normally
    returned on the same split-adjusted per-share basis, but some yfinance/Yahoo
    responses around ETF splits can contain pre-split cash amounts.  If a
    pre-split dividend is implausibly larger than the first post-split dividend,
    divide only the older dividends by the split ratio so numerator and
    denominator stay on the same per-share basis.
    """
    if "Dividends" not in data.columns:
        return pd.DataFrame(columns=["dividend"])

    dividends = pd.to_numeric(data["Dividends"], errors="coerce").fillna(0).to_frame("dividend")
    dividends = dividends[dividends["dividend"] > 0].copy()
    if dividends.empty or "Stock Splits" not in data.columns:
        return dividends.astype("float64")

    splits = pd.to_numeric(data["Stock Splits"], errors="coerce").fillna(0)
    splits = splits[splits > 0].sort_index()
    if splits.empty:
        return dividends.astype("float64")

    adjusted = dividends.copy()
    for split_date, split_ratio in splits.items():
        if split_ratio <= 1:
            continue

        before = adjusted.loc[adjusted.index < split_date, "dividend"].tail(4)
        after = adjusted.loc[adjusted.index > split_date, "dividend"].head(4)
        if before.empty or after.empty:
            continue

        before_median = float(before.median())
        after_median = float(after.median())
        if after_median <= 0:
            continue

        # If dividends are already split-adjusted, before/after cash amounts
        # should be broadly comparable.  If they are raw pre-split amounts, the
        # ratio will be close to the stock split ratio (e.g. SCHD's 3-for-1).
        if before_median / after_median > max(1.8, float(split_ratio) * 0.65):
            adjusted.loc[adjusted.index < split_date, "dividend"] = (
                adjusted.loc[adjusted.index < split_date, "dividend"] / float(split_ratio)
            )

    return adjusted.astype("float64")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_schd_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Fetch SCHD split-adjusted close and dividend history from yfinance only."""
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=365 * 11 + 10)

    history = yf.Ticker(TICKER).history(
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
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

    dividends_df = _normalize_dividends_to_close_basis(data)
    dividends_df.index.name = "date"
    price_df.index.name = "date"
    actions_df = data[[col for col in ["Dividends", "Stock Splits"] if col in data.columns]].copy()
    actions_df.index.name = "date"
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return price_df.astype("float64"), dividends_df.astype("float64"), actions_df, fetched_at


def _calculate_latest_four_dividend_sum(price_index: pd.DatetimeIndex, dividends_df: pd.DataFrame) -> pd.Series:
    """Return each price date's sum of the most recent four dividend events.

    A 365-day trailing window can briefly include five regular quarterly
    dividends around an ex-dividend date when the new dividend and the prior
    year's same-quarter dividend both fall inside the lookback window.  Counting
    exactly the latest four events as of each price date keeps the TTM dividend
    on a regular quarterly cadence and removes one-day dividend-date needles.
    """
    if dividends_df is None or dividends_df.empty:
        return pd.Series(np.nan, index=price_index, dtype="float64")

    dividends = dividends_df.copy().sort_index()
    dividends.index = _to_naive_normalized_index(dividends.index)
    dividends["dividend"] = pd.to_numeric(dividends["dividend"], errors="coerce")
    dividends = dividends.dropna(subset=["dividend"])
    dividends = dividends[dividends["dividend"] > 0]

    if len(dividends) < 4:
        return pd.Series(np.nan, index=price_index, dtype="float64")

    div_dates = dividends.index.to_numpy(dtype="datetime64[ns]")
    div_values = dividends["dividend"].to_numpy(dtype="float64")
    cumulative = np.concatenate([[0.0], np.cumsum(div_values)])

    price_dates = pd.DatetimeIndex(price_index)
    price_dates_np = _to_naive_normalized_index(price_dates).to_numpy(dtype="datetime64[ns]")
    right_edges = np.searchsorted(div_dates, price_dates_np, side="right")

    ttm_values = np.full(len(price_dates), np.nan, dtype="float64")
    enough_dividends = right_edges >= 4
    rights = right_edges[enough_dividends]
    ttm_values[enough_dividends] = cumulative[rights] - cumulative[rights - 4]

    return pd.Series(ttm_values, index=price_index, dtype="float64")


def _build_spike_diagnostics(metrics: pd.DataFrame, dividends_df: pd.DataFrame) -> pd.DataFrame:
    """Build a small diagnostic table for dates that could create yield spikes."""
    if metrics.empty:
        return pd.DataFrame()

    diagnostic = metrics.copy()
    diagnostic["yield_abs_change"] = diagnostic["ttm_yield"].diff().abs()
    diagnostic["price_pct_change"] = diagnostic["price"].pct_change().abs() * 100

    dividend_dates = set(dividends_df.index.normalize()) if dividends_df is not None and not dividends_df.empty else set()
    diagnostic["dividend_event"] = diagnostic.index.normalize().isin(dividend_dates)
    candidates = diagnostic[
        (diagnostic["yield_abs_change"] >= 0.35)
        | (diagnostic["price_pct_change"] >= 8)
        | diagnostic["dividend_event"]
    ].tail(24)
    return candidates[
        ["price", "ttm_dividend", "ttm_yield", "ttm_yield_raw", "yield_abs_change", "price_pct_change", "dividend_event"]
    ]


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def calculate_schd_dividend_yield() -> dict:
    price_df, dividends_df, actions_df, fetched_at = fetch_schd_history()
    metrics = price_df.copy()
    metrics["ttm_dividend"] = _calculate_latest_four_dividend_sum(metrics.index, dividends_df)
    metrics["ttm_yield"] = np.where(
        (metrics["price"] > 0) & (metrics["ttm_dividend"] > 0),
        metrics["ttm_dividend"] / metrics["price"] * 100,
        np.nan,
    )
    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    # Data-error guardrail: a split/adjustment mismatch usually appears as a
    # one-day needle far outside SCHD's practical yield range.  Keep normal
    # market movements intact, but exclude extreme observations from charting
    # and averages so one bad Yahoo event does not dominate the visual.
    metrics["ttm_yield_raw"] = metrics["ttm_yield"]
    outlier_mask = metrics["ttm_yield"].notna() & ((metrics["ttm_yield"] < 1.0) | (metrics["ttm_yield"] > 8.0))
    metrics.loc[outlier_mask, "ttm_yield"] = np.nan

    valid = metrics.dropna(subset=["price", "ttm_dividend", "ttm_yield"])
    if valid.empty:
        raise ValueError("SCHD TTM 배당률을 계산할 수 없습니다.")

    latest_date = valid.index.max()
    latest = valid.loc[latest_date]
    current_price = float(latest["price"])
    latest_four_dividend = float(latest["ttm_dividend"])
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
        ttm_buy_price = latest_four_dividend / target_yield if latest_four_dividend > 0 else np.nan
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
        "actions": actions_df,
        "spike_diagnostics": _build_spike_diagnostics(metrics, dividends_df),
        "latest_date": latest_date,
        "current_price": current_price,
        "current_ttm_yield": current_ttm_yield,
        "five_year_average_yield": five_year_average_yield,
        "latest_four_dividend": latest_four_dividend,
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(data: dict) -> None:
    cards = [
        ("현재 SCHD 가격", _format_currency(data["current_price"]), f"기준일 {data['latest_date']:%Y-%m-%d}"),
        ("현재 TTM 배당률", _format_percent(data["current_ttm_yield"]), "최근 4회 배당금 ÷ 현재가"),
        ("5년 평균 배당률", _format_percent(data["five_year_average_yield"]), "일별 TTM 배당률 평균"),
        ("최근 4회 배당금", _format_currency(data["latest_four_dividend"]), "최신일 기준 최근 4개 배당 합계"),
        ("최근 분기 배당금", _format_currency(data["recent_quarter_dividend"]), "가장 최근 1회 배당"),
    ]

    columns = st.columns(len(cards))
    for column, (label, value, sub) in zip(columns, cards, strict=True):
        column.metric(label=label, value=value, help=sub, border=True)


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
        (data["current_ttm_yield"], "현재 배당률", "#f2994a", "solid", 1.8),
        (data["five_year_average_yield"], "5년 평균", "#3182f6", "dash", 1.5),
        (3.5, "3.5%", "#98a2b3", "dot", 1.0),
        (3.7, "3.7%", "#667085", "dot", 1.0),
        (3.8, "3.8%", "#475467", "dot", 1.0),
    ]
    for y_value, label, color, dash, width in reference_lines:
        if y_value is None or pd.isna(y_value) or not np.isfinite(y_value):
            continue
        fig.add_trace(
            go.Scatter(
                x=[chart_df.index.min(), chart_df.index.max()],
                y=[float(y_value), float(y_value)],
                mode="lines",
                name=f"{label} {_format_percent(y_value)}",
                line=dict(color=color, width=width, dash=dash),
                opacity=0.78,
                hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
            )
        )

    y_candidates = [chart_df["ttm_yield"].dropna()]
    y_candidates.extend(pd.Series([line[0]]) for line in reference_lines if line[0] is not None and np.isfinite(line[0]))
    y_values = pd.concat(y_candidates, ignore_index=True).dropna()
    y_min = float(y_values.min()) if not y_values.empty else 3.0
    y_max = float(y_values.max()) if not y_values.empty else 4.5
    y_range = [min(3.0, y_min - 0.15), max(4.5, y_max + 0.15)]

    fig.update_layout(
        title="SCHD Dividend Yield TTM",
        height=470,
        margin=dict(l=24, r=24, t=62, b=36),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        xaxis_title="날짜",
        yaxis_title="배당률",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#344054", size=11),
        ),
        font=dict(color="#191f28"),
    )
    fig.update_yaxes(ticksuffix="%", range=y_range, gridcolor="#edf2f7", zeroline=False)
    fig.update_xaxes(gridcolor="#f8f9fa", rangeslider_visible=False)
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
    except ModuleNotFoundError as exc:
        missing_package = exc.name or str(exc)
        st.error(f"필수 패키지 누락: {missing_package}")
        return
    except ImportError as exc:
        st.error(f"필수 패키지 누락: {exc}")
        return
    except Exception as exc:
        st.error("SCHD 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        st.caption(f"오류 정보: {exc}")
        return

    if data["current_price"] <= 0 or data["latest_four_dividend"] <= 0:
        st.warning("SCHD 가격 또는 최근 4회 배당금이 0 이하로 계산되어 일부 지표가 제한될 수 있습니다.")

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
            - 현재 SCHD 가격: yfinance 일반 종가(`Close`, `auto_adjust=False`, split-adjusted)의 가장 최근 거래일 값
            - 최근 4회 배당금: 최신 가격일 기준 가장 최근 4개 split-adjusted SCHD 배당금 합계
            - 각 날짜의 TTM 배당률: 해당 날짜까지 발생한 SCHD 배당 이벤트 중 가장 최근 4회 split-adjusted 배당금 합계 ÷ 같은 기준의 종가 × 100
            - 5년 평균 배당률: 최근 5년 구간의 일별 TTM 배당률 평균
            - TTM 기준 매수가: 최근 4회 배당금 ÷ 목표 배당률
            - 최근 분기×4 기준 매수가: 최근 분기 배당금 × 4 ÷ 목표 배당률
            - 데이터 오류 방어: 가격/배당 split 기준 불일치로 판단되는 1% 미만 또는 8% 초과 TTM 배당률은 차트와 평균 계산에서 제외
            """
        )
        st.caption(f"데이터 소스: yfinance · 티커: {TICKER} · 캐시 TTL: {CACHE_TTL_SECONDS:,}초 · 조회 시각: {data['fetched_at']}")
        diagnostics = data.get("spike_diagnostics")
        if diagnostics is not None and not diagnostics.empty:
            st.markdown("#### 스파이크 점검용 최근 후보 데이터")
            st.dataframe(diagnostics, use_container_width=True)
        if st.button("🔄 SCHD 배당률 캐시 초기화", use_container_width=True):
            fetch_schd_history.clear()
            calculate_schd_dividend_yield.clear()
            st.rerun()


if not os.environ.get("GORANI_SKIP_PAGE_RENDER"):
    main()
