import html
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
TARGET_YIELDS = [0.035, 0.036, 0.037, 0.038]
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
    price_df["Close"] = price_df["price"]
    if "High" in data.columns:
        price_df["High"] = pd.to_numeric(data["High"], errors="coerce")
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


def _calculate_52w_high_drawdown(
    price_df: pd.DataFrame,
    current_price: float,
    latest_date: pd.Timestamp,
) -> tuple[float, float]:
    """Return the latest 52-week high and current-price drawdown from it."""
    if price_df is None or price_df.empty:
        return np.nan, np.nan
    if (
        current_price is None
        or pd.isna(current_price)
        or not np.isfinite(current_price)
        or current_price <= 0
    ):
        return np.nan, np.nan
    if latest_date is None or pd.isna(latest_date):
        return np.nan, np.nan

    one_year_ago = latest_date - pd.Timedelta(days=365)
    last_52w_df = price_df.loc[price_df.index >= one_year_ago]
    if last_52w_df.empty:
        return np.nan, np.nan

    high_col = "High" if "High" in last_52w_df.columns and not last_52w_df["High"].dropna().empty else "Close"
    high_52w = np.nan
    for candidate_col in [high_col, "Close"]:
        if candidate_col not in last_52w_df.columns:
            continue
        candidate_high = pd.to_numeric(last_52w_df[candidate_col], errors="coerce").dropna().max()
        if (
            candidate_high is not None
            and not pd.isna(candidate_high)
            and np.isfinite(candidate_high)
            and candidate_high > 0
        ):
            high_52w = float(candidate_high)
            break

    if pd.isna(high_52w) or not np.isfinite(high_52w) or high_52w <= 0:
        return np.nan, np.nan

    drawdown_from_52w_high = current_price / high_52w - 1
    return float(high_52w), float(drawdown_from_52w_high)


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
    high_52w, drawdown_from_52w_high = _calculate_52w_high_drawdown(price_df, current_price, latest_date)

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
        "high_52w": high_52w,
        "drawdown_from_52w_high": drawdown_from_52w_high,
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
            .schd-yield-card {
                background: #ffffff;
                border: 1px solid #eaecf0;
                border-radius: 16px;
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
                min-height: 106px;
                padding: 12px 12px 11px;
                margin-bottom: 6px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                gap: 4px;
            }
            .schd-yield-card-label {
                color: #667085;
                font-size: 0.82rem;
                font-weight: 700;
                line-height: 1.18;
                min-height: 1rem;
                margin: 0;
            }
            .schd-yield-card-value {
                color: #191f28;
                font-size: clamp(1.32rem, 2.05vw, 1.76rem);
                font-weight: 800;
                letter-spacing: -0.035em;
                line-height: 1.04;
                margin: 0;
                word-break: keep-all;
            }
            .schd-yield-card-subtext {
                color: #667085;
                font-size: 0.72rem;
                font-weight: 600;
                line-height: 1.2;
                min-height: 1.55rem;
                margin: 1px 0 0;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                word-break: keep-all;
            }
            .schd-yield-target-summary {
                display: flex;
                align-items: center;
                justify-content: center;
                flex-wrap: wrap;
                gap: 0 4px;
                line-height: 1.2;
            }
            .schd-yield-target-summary span {
                white-space: nowrap;
            }
            .schd-yield-card-subtext:empty::after {
                content: "\\00a0";
            }
            @media (max-width: 640px) {
                .schd-yield-card {
                    min-height: 104px;
                    padding: 12px 12px 10px;
                }
                .schd-yield-card-value {
                    font-size: 1.48rem;
                }
                .schd-yield-card-subtext {
                    font-size: 0.71rem;
                    min-height: 1.45rem;
                }
            }
            @media (prefers-color-scheme: dark) {
                .schd-yield-card {
                    background: #ffffff;
                    border-color: #eaecf0;
                    box-shadow: none;
                }
                .schd-yield-card-label,
                .schd-yield-card-subtext {
                    color: #667085;
                }
            }
            .schd-reference-link {
                display: block;
                margin-top: 6px;
                color: #98a2b3;
                font-size: 0.84rem;
            }
            .schd-reference-link a { color: #667085; text-decoration: none; }
            .schd-reference-link a:hover { color: #344054; text-decoration: underline; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_ttm_yield_card_style(ttm_yield: float) -> tuple[str, str, str, str]:
    if ttm_yield is None or pd.isna(ttm_yield) or not np.isfinite(ttm_yield):
        return "#F2F4F7", "#344054", "#D0D5DD", "계산 대기"
    if ttm_yield < 3.4:
        return "#FDECEC", "#DC2626", "#FCA5A5", "🟥 비싸요"
    if ttm_yield < 3.5:
        return "#FFF3E0", "#EA580C", "#FDBA74", "🟧 진입고려"
    if ttm_yield < 3.6:
        return "#FEF9C3", "#CA8A04", "#FDE68A", "🟨 진입OK"
    if ttm_yield < 3.7:
        return "#ECFCCB", "#65A30D", "#BEF264", "🟩 매수GO"
    if ttm_yield < 3.8:
        return "#DCFCE7", "#16A34A", "#86EFAC", "💚 매수가자"
    return "#D1FAE5", "#047857", "#6EE7B7", "💚 강함"


def _render_metric_card(
    column,
    label: str,
    value: str,
    subtext: str = "",
    bg_color: str = "#ffffff",
    value_color: str = "#191f28",
    border_color: str = "#eaecf0",
    subtext_is_html: bool = False,
) -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_subtext = str(subtext) if subtext_is_html else html.escape(str(subtext))
    column.markdown(
        f'''
        <div class="schd-yield-card" style="background: {bg_color}; border-color: {border_color};">
            <div class="schd-yield-card-label">{safe_label}</div>
            <div class="schd-yield-card-value" style="color: {value_color};">{safe_value}</div>
            <div class="schd-yield-card-subtext">{safe_subtext}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _build_52w_drawdown_price_label(drawdown_from_52w_high: float) -> str:
    if (
        drawdown_from_52w_high is None
        or pd.isna(drawdown_from_52w_high)
        or not np.isfinite(drawdown_from_52w_high)
    ):
        return "현재 SCHD 가격"
    return f"현재가 = 52H {drawdown_from_52w_high * 100:.1f}%"


def _build_target_price_summary(target_table: pd.DataFrame) -> str:
    if target_table is None or target_table.empty:
        return ""
    if not {"목표 배당률", "TTM 기준 매수가"}.issubset(target_table.columns):
        return ""

    target_colors = {
        "3.5%": "#CA8A04",
        "3.6%": "#65A30D",
        "3.7%": "#16A34A",
    }
    summary_parts = []
    for target_label, color in target_colors.items():
        matched = target_table.loc[target_table["목표 배당률"] == target_label, "TTM 기준 매수가"]
        target_price = "-" if matched.empty else matched.iloc[0]
        if target_price is None or pd.isna(target_price) or str(target_price).strip() in {"", "-"}:
            summary_text = f"{target_label} -"
        else:
            summary_text = f"{target_label} {target_price}"
        summary_parts.append(f'<span style="color:{color}">{html.escape(summary_text)}</span>')

    return '<span class="schd-yield-target-summary">' + " · ".join(summary_parts) + "</span>"


def render_metric_cards(data: dict) -> None:
    background_color, value_color, border_color, status = _get_ttm_yield_card_style(data["current_ttm_yield"])
    price_summary = _build_target_price_summary(data["target_table"])
    cards = [
        {
            "label": "현재 TTM 배당률",
            "value": _format_percent(data["current_ttm_yield"]),
            "subtext": f"현재는 · {status}",
            "bg_color": background_color,
            "value_color": value_color,
            "border_color": border_color,
        },
        {
            "label": _build_52w_drawdown_price_label(data.get("drawdown_from_52w_high")),
            "value": _format_currency(data["current_price"]),
            "subtext": price_summary or f"기준일 {data['latest_date']:%Y-%m-%d}",
            "bg_color": "#F3F6FA",
            "border_color": "#D8E1EC",
            "subtext_is_html": bool(price_summary),
        },
        {
            "label": "5년 평균 배당률",
            "value": _format_percent(data["five_year_average_yield"]),
            "subtext": "일별 TTM 배당률 평균",
        },
        {
            "label": "최근 4회 배당금",
            "value": _format_currency(data["latest_four_dividend"]),
            "subtext": "최신일 기준 최근 4개 배당 합계",
        },
        {
            "label": "최근 분기 배당금",
            "value": _format_currency(data["recent_quarter_dividend"]),
            "subtext": "가장 최근 1회 배당",
        },
    ]

    columns = st.columns(5, gap="small")
    for column, card in zip(columns, cards, strict=True):
        _render_metric_card(column, **card)


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

    five_year_average_yield = data["five_year_average_yield"]
    reference_lines = [
        (
            five_year_average_yield,
            f"5년 평균 {_format_percent(five_year_average_yield)}",
            "#3182f6",
            "dash",
            1.5,
        ),
        (3.5, "3.5%", "#98a2b3", "dot", 1.0),
        (3.6, "3.6%", "#7c8798", "dot", 1.0),
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
                name=label,
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



# ──────────────────────────────────────────────
# 4. Main page
# ──────────────────────────────────────────────
def main() -> None:
    apply_schd_styles()

    st.markdown("# 📈 SCHD 배당률 매수 판단")
    st.caption("판별: 배당률 3.4% 미만(🟥), 3.5% (진입고려🟡-7%선), 3.6% (진입OK🟢), 3.7% 이상(매수GOOD🐸)")

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
    st.markdown(
        '<span class="schd-reference-link">참고: '
        '<a href="https://seekingalpha.com/symbol/SCHD/dividends/yield" target="_blank" rel="noopener noreferrer">'
        'Seeking Alpha SCHD Dividend Yield 페이지 바로가기</a></span>',
        unsafe_allow_html=True,
    )

    with st.expander("계산 기준 보기"):
        st.markdown(
            """
            - 현재 SCHD 가격: yfinance 일반 종가(`Close`, `auto_adjust=False`, split-adjusted)의 가장 최근 거래일 값
            - 최근 4회 배당금: 최신 가격일 기준 가장 최근 4개 split-adjusted SCHD 배당금 합계
            - 각 날짜의 TTM 배당률: 해당 날짜까지 발생한 SCHD 배당 이벤트 중 가장 최근 4회 split-adjusted 배당금 합계 ÷ 같은 기준의 종가 × 100
            - 5년 평균 배당률: 최근 5년 구간의 일별 TTM 배당률 평균
            - 목표 배당률 및 기준선: 3.5%, 3.6%, 3.7%, 3.8%
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


if not os.environ.get("GORANI_SKIP_PAGE_RENDER"):
    main()
