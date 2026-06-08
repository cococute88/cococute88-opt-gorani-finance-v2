"""Virtual performance analytics for the asset tracker page.

This module implements a defensive, yfinance-based backtest for the latest
asset tracker snapshot.  The asset tracker stores monthly KRW valuation by tag,
not transaction quantities, so quantities and initial capital are inferred from
current valuation and market prices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from logic.market import align_and_convert_to_krw
from logic.tracker import get_asset_type

KOSPI_TICKER = "^KS11"
SP500_TICKER = "^GSPC"
QQQ_TICKER = "QQQ"
USDKRW_TICKERS = ("KRW=X", "USDKRW=X")

US_TICKERS = {
    "AAPL", "AMZN", "BRK", "BRKB", "BULZ", "DGRW", "DGRO", "DIA", "FNGU",
    "GOOGL", "GOOG", "IVV", "JNJ", "KO", "META", "MSFT", "NVDA", "PG", "QLD",
    "QQQ", "QQQM", "SCHD", "SOXL", "SPLG", "SPY", "SSO", "TQQQ", "TECL", "UPRO",
    "VTI", "VTV", "VUG", "VYM",
}

CASH_KEYWORDS = ("현금", "예금", "적금", "예수금", "원화", "달러", "usd", "cma", "rp", "mmf", "기타")


@dataclass
class TrackerAssetInput:
    tag: str
    amount_krw: float
    ticker: str
    currency: str


@dataclass
class TrackerPerformanceResult:
    latest_key: str = ""
    latest_date: date | None = None
    requested_start_date: date | None = None
    effective_start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = 0.0
    current_portfolio_value: float = 0.0
    included_assets: list[str] = field(default_factory=list)
    excluded_assets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    chart: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    cards: dict[str, dict[str, float]] = field(default_factory=dict)


def month_key_to_date(month_key: str, today: date | None = None) -> date:
    """Return the usable snapshot date for a YYYY-MM key.

    Monthly tracker data has no day component, so the month-end is used unless it
    would be in the future; in that case today's date is used.
    """
    today = today or date.today()
    period_end = pd.Period(month_key, freq="M").to_timestamp("M").date()
    return min(period_end, today)


def default_start_date_for_latest(month_key: str, today: date | None = None) -> date:
    latest = month_key_to_date(month_key, today=today)
    return (pd.Timestamp(latest) - pd.DateOffset(years=2)).date()


def _normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("조회 결과가 비어 있습니다.")
    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        close_cols = [
            col for col in normalized.columns
            if any(str(part).lower() == "close" for part in col)
        ]
        if not close_cols:
            raise ValueError("Close 컬럼이 없습니다.")
        close = normalized.loc[:, close_cols]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        normalized = pd.DataFrame({"Close": close})
    else:
        if "Close" not in normalized.columns:
            raise ValueError("Close 컬럼이 없습니다.")
        close = normalized["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        normalized = pd.DataFrame({"Close": close})
    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized[~normalized.index.isna()]
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    if normalized.empty:
        raise ValueError("정규화 후 데이터가 비어 있습니다.")
    return normalized


def _fetch_stooq_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    if ticker.startswith("^") or "=" in ticker:
        raise ValueError("Stooq 폴백 미지원 티커입니다.")
    stooq_url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d&d1={start.replace('-', '')}&d2={end.replace('-', '')}"
    response = requests.get(stooq_url, timeout=20)
    response.raise_for_status()
    raw = response.text.strip()
    if not raw:
        raise ValueError("Stooq 응답이 비어 있습니다.")
    df = pd.read_csv(StringIO(raw))
    if df.empty or "Date" not in df.columns:
        raise ValueError("Stooq CSV가 비어 있습니다.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return _normalize_history_frame(df.dropna(subset=["Date"]).set_index("Date"))


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_close_series(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch close prices defensively with yfinance and a Stooq fallback for US symbols."""
    errors: list[str] = []
    if not ticker:
        return pd.Series(dtype="float64")

    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=False, raise_errors=True)
        normalized = _normalize_history_frame(df)
        return pd.to_numeric(normalized["Close"], errors="coerce").dropna().astype("float64")
    except Exception as exc:
        errors.append(f"Ticker.history 실패: {exc}")

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, actions=False, progress=False, threads=False, timeout=20)
        normalized = _normalize_history_frame(df)
        return pd.to_numeric(normalized["Close"], errors="coerce").dropna().astype("float64")
    except Exception as exc:
        errors.append(f"download 실패: {exc}")

    try:
        normalized = _fetch_stooq_history(ticker, start, end)
        return pd.to_numeric(normalized["Close"], errors="coerce").dropna().astype("float64")
    except Exception as exc:
        errors.append(f"Stooq 실패: {exc}")

    return pd.Series(dtype="float64", name="; ".join(errors))


def _fetch_with_korean_fallback(ticker: str, start: str, end: str) -> tuple[pd.Series, str]:
    series = fetch_close_series(ticker, start, end)
    used = ticker
    if series.empty and ticker.endswith(".KS"):
        alt = ticker.removesuffix(".KS") + ".KQ"
        alt_series = fetch_close_series(alt, start, end)
        if not alt_series.empty:
            return alt_series, alt
    return series, used


def _fetch_fx_series(start: str, end: str) -> pd.Series:
    for ticker in USDKRW_TICKERS:
        series = fetch_close_series(ticker, start, end).dropna()
        if not series.empty and 700.0 <= float(series.iloc[-1]) <= 3000.0:
            return series
    return pd.Series(dtype="float64")


def _asof(series: pd.Series, day: date | pd.Timestamp, direction: str = "backward") -> tuple[float | None, pd.Timestamp | None]:
    if series is None or series.empty:
        return None, None
    ts = pd.Timestamp(day).tz_localize(None).normalize()
    if direction == "forward":
        values = series.loc[series.index >= ts]
        if values.empty:
            return None, None
        idx = values.index[0]
    else:
        values = series.loc[series.index <= ts]
        if values.empty:
            return None, None
        idx = values.index[-1]
    value = float(values.loc[idx])
    return (value, idx) if value > 0 else (None, None)


def _standardize_tag(tag: Any) -> str:
    return str(tag or "").strip().upper().replace(" ", "")


def map_tracker_tag_to_asset(tag: str, amount_krw: float) -> TrackerAssetInput | None:
    """Map an asset-tracker tag to a free market-data ticker when safe.

    Tags that look like cash buckets or unmapped grouped assets are intentionally
    excluded because the tracker stores KRW value only, not enough information to
    infer a reliable price series.
    """
    raw = str(tag or "").strip()
    normalized = _standardize_tag(raw)
    lower = raw.lower()
    if not raw or any(keyword in lower for keyword in CASH_KEYWORDS):
        return None
    if "비트코인" in raw or normalized in {"BTC", "BITCOIN"}:
        return TrackerAssetInput(raw, float(amount_krw), "BTC-KRW", "KRW")
    if normalized in US_TICKERS or get_asset_type(normalized) in {"nasdaq", "spy", "dividend", "leverage"}:
        ticker = "BRK-B" if normalized in {"BRK", "BRKB"} else normalized
        return TrackerAssetInput(raw, float(amount_krw), ticker, "USD")
    if normalized.isdigit() and len(normalized) == 6:
        return TrackerAssetInput(raw, float(amount_krw), f"{normalized}.KS", "KRW")
    return None


def _to_krw_close(asset: TrackerAssetInput, start: str, end: str, fx_series: pd.Series) -> tuple[pd.Series, str]:
    close, used_ticker = _fetch_with_korean_fallback(asset.ticker, start, end)
    if close.empty:
        return pd.Series(dtype="float64"), used_ticker
    if asset.currency == "USD":
        krw_close, _ = align_and_convert_to_krw(close, fx_series)
        return krw_close, used_ticker
    return close, used_ticker


def _scale_index_to_initial(close: pd.Series, start_day: date, initial_capital: float) -> tuple[pd.Series, float | None]:
    start_price, start_idx = _asof(close, start_day, direction="forward")
    if start_price is None or start_idx is None or initial_capital <= 0:
        return pd.Series(dtype="float64"), None
    scaled = close.loc[close.index >= start_idx] / start_price * initial_capital
    current_price, _ = _asof(close, close.index.max(), direction="backward")
    current_value = (current_price / start_price * initial_capital) if current_price else None
    return scaled.dropna(), current_value


def _month_end_frequency() -> str:
    # pandas 3.x prefers ME; older pandas versions only support M.
    try:
        pd.date_range("2024-01-01", periods=1, freq="ME")
        return "ME"
    except ValueError:
        return "M"


def _monthly_profit_frame(portfolio: pd.Series, end_day: date | None = None) -> pd.DataFrame:
    """Build the 12-month monthly P/L frame from the same KRW portfolio series used in the top chart."""
    if portfolio.empty:
        return pd.DataFrame()

    monthly_source = pd.to_numeric(portfolio.copy(), errors="coerce").dropna()
    monthly_source.index = pd.to_datetime(monthly_source.index, errors="coerce")
    monthly_source = monthly_source[~monthly_source.index.isna()]
    monthly_source.index = monthly_source.index.tz_localize(None) if monthly_source.index.tz is not None else monthly_source.index
    monthly_source.index = monthly_source.index.normalize()
    monthly_source = monthly_source[~monthly_source.index.duplicated(keep="last")].sort_index()

    if end_day is not None:
        monthly_source = monthly_source.loc[monthly_source.index <= pd.Timestamp(end_day).normalize()]
    if monthly_source.empty:
        return pd.DataFrame()

    monthly_value = monthly_source.resample(_month_end_frequency()).last().dropna().sort_index()
    if len(monthly_value) < 2:
        return pd.DataFrame()

    monthly = pd.DataFrame({"portfolio": monthly_value})
    monthly["profit"] = monthly["portfolio"].diff()
    monthly = monthly.dropna(subset=["profit"]).tail(12).copy()
    monthly["month_end"] = monthly.index
    monthly["display_label"] = monthly["month_end"].dt.strftime("%y.%m")
    # Plotly may infer labels such as 25.10 as numeric 25.1.  Keep the visible
    # label unchanged while forcing the current page traces to stay categorical.
    monthly["label"] = "\u200b" + monthly["display_label"]
    return monthly


def build_tracker_performance(asset_data: dict[str, dict[str, Any]], start_day: date, today: date | None = None) -> TrackerPerformanceResult:
    today = today or date.today()
    result = TrackerPerformanceResult(requested_start_date=start_day)
    valid_keys = sorted(k for k, v in asset_data.items() if isinstance(v, dict) and v)
    if not valid_keys:
        result.warnings.append("저장된 포트폴리오 스냅샷이 없어 성과 분석을 표시할 수 없습니다.")
        return result

    latest_key = valid_keys[-1]
    latest_snapshot = asset_data[latest_key]
    end_day = month_key_to_date(latest_key, today=today)
    result.latest_key = latest_key
    result.latest_date = end_day
    result.end_date = end_day

    fetch_start = (pd.Timestamp(start_day) - pd.Timedelta(days=14)).date().isoformat()
    fetch_end = (pd.Timestamp(end_day) + pd.Timedelta(days=3)).date().isoformat()
    fx_series = _fetch_fx_series(fetch_start, fetch_end)
    if fx_series.empty:
        result.warnings.append("USD/KRW 환율 데이터를 가져오지 못해 USD 자산과 S&P 500/QQQ 비교 일부가 제외될 수 있습니다.")

    priced_assets: list[tuple[str, pd.Series, float]] = []
    effective_starts: list[pd.Timestamp] = []
    latest_total = sum(float(v or 0) for v in latest_snapshot.values())

    for tag, amount in latest_snapshot.items():
        amount_krw = float(amount or 0)
        asset = map_tracker_tag_to_asset(tag, amount_krw)
        if asset is None:
            result.excluded_assets.append(str(tag))
            continue
        close_krw, used_ticker = _to_krw_close(asset, fetch_start, fetch_end, fx_series)
        if close_krw.empty:
            result.excluded_assets.append(f"{tag}({asset.ticker})")
            continue
        current_price, _ = _asof(close_krw, end_day, direction="backward")
        _, start_idx = _asof(close_krw, start_day, direction="forward")
        if current_price is None or start_idx is None:
            result.excluded_assets.append(f"{tag}({used_ticker})")
            continue
        quantity = amount_krw / current_price
        priced_assets.append((str(tag), close_krw, quantity))
        effective_starts.append(start_idx)
        result.included_assets.append(f"{tag}({used_ticker})")

    if effective_starts:
        result.effective_start_date = max(effective_starts).date()

    portfolio_parts: list[pd.Series] = []
    if result.effective_start_date is not None:
        for tag, close_krw, quantity in priced_assets:
            start_price, _ = _asof(close_krw, result.effective_start_date, direction="backward")
            if start_price is None:
                result.excluded_assets.append(tag)
                continue
            result.initial_capital += quantity * start_price
            asset_value = close_krw.loc[close_krw.index >= pd.Timestamp(result.effective_start_date)].ffill() * quantity
            if not asset_value.empty:
                portfolio_parts.append(asset_value.rename(tag))

    if not portfolio_parts or result.initial_capital <= 0:
        result.current_portfolio_value = latest_total
        result.warnings.append("가격 데이터가 있는 보유자산이 부족해 가상 성과 그래프를 만들 수 없습니다.")
        return result

    combined = pd.concat(portfolio_parts, axis=1).sort_index().ffill()
    combined = combined.loc[combined.index >= pd.Timestamp(result.effective_start_date)]
    combined = combined.loc[combined.index <= pd.Timestamp(end_day).normalize()]
    portfolio = combined.sum(axis=1, min_count=1).dropna().sort_index()
    current_value, _ = _asof(portfolio, end_day, direction="backward")
    result.current_portfolio_value = float(current_value) if current_value is not None else latest_total

    capital_line = pd.Series(result.initial_capital, index=portfolio.index, name="initial_capital")
    chart = pd.DataFrame({"portfolio": portfolio, "initial_capital": capital_line})

    kospi = fetch_close_series(KOSPI_TICKER, fetch_start, fetch_end)
    sp500_usd = fetch_close_series(SP500_TICKER, fetch_start, fetch_end)
    qqq_usd = fetch_close_series(QQQ_TICKER, fetch_start, fetch_end)
    sp500_krw, _ = align_and_convert_to_krw(sp500_usd, fx_series)
    qqq_krw, _ = align_and_convert_to_krw(qqq_usd, fx_series)

    benchmark_specs = {
        "kospi": kospi,
        "sp500": sp500_krw,
        "qqq": qqq_krw,
    }
    current_values: dict[str, float] = {}
    for key, series in benchmark_specs.items():
        scaled, current_value = _scale_index_to_initial(series, result.effective_start_date or start_day, result.initial_capital)
        if scaled.empty or current_value is None:
            result.warnings.append(f"{key.upper()} 비교지표 가격 데이터가 부족해 그래프에서 제외했습니다.")
            continue
        chart[key] = scaled.reindex(chart.index).ffill()
        current_values[key] = float(chart[key].dropna().iloc[-1]) if not chart[key].dropna().empty else float(current_value)

    result.chart = chart.dropna(subset=["portfolio"]).sort_index()
    result.monthly = _monthly_profit_frame(result.chart["portfolio"], end_day=result.end_date)
    result.cards = {
        "initial_capital": {"value": result.initial_capital, "return": 0.0},
        "portfolio": {"value": result.current_portfolio_value, "return": result.current_portfolio_value / result.initial_capital - 1.0},
    }
    for key, value in current_values.items():
        result.cards[key] = {"value": value, "return": value / result.initial_capital - 1.0}
    return result
