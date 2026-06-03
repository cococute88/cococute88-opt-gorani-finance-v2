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
            .schd-yield-card {
                background: #ffffff;
                border: 1px solid #eaecf0;
                border-radius: 18px;
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
                min-height: 134px;
                padding: 16px 14px 14px;
                margin-bottom: 10px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                gap: 7px;
            }
            .schd-yield-card-label {
                color: #667085;
                font-size: 0.86rem;
                font-weight: 700;
                line-height: 1.25;
                min-height: 1.1rem;
            }
            .schd-yield-card-value {
                color: #191f28;
                font-size: clamp(1.42rem, 2.3vw, 1.92rem);
                font-weight: 800;
                letter-spacing: -0.035em;
                line-height: 1.12;
                word-break: keep-all;
            }
            .schd-yield-card-subtext {
                color: #667085;
                font-size: 0.76rem;
                font-weight: 600;
                line-height: 1.32;
                min-height: 2.05rem;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                word-break: keep-all;
            }
            .schd-yield-card-subtext:empty::after {
                content: "\\00a0";
            }
            @media (max-width: 640px) {
                .schd-yield-card {
                    min-height: 122px;
                    padding: 15px 14px 13px;
                }
                .schd-yield-card-value {
                    font-size: 1.65rem;
                }
                .schd-yield-card-subtext {
                    min-height: 1.8rem;
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
    sub
