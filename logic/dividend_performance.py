"""Performance analytics helpers for the dividend ledger page.

The ledger's existing ``transaction["date"]`` field is treated as the purchase
(or sell) date for all performance calculations. This module keeps yfinance and
calculation details out of the Streamlit page as much as possible while staying
fully defensive: missing market/FX data should reduce chart completeness, not
break page rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf

from logic.dividend_ledger import normalize_transactions, summarize_holdings, to_float

KOSPI_TICKER = "^KS11"
SP500_TICKER = "^GSPC"
USDKRW_TICKER = "USDKRW=X"


@dataclass
class PerformanceResult:
    """Container returned to the Streamlit page for rendering."""

    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    yearly: pd.DataFrame = field(default_factory=pd.DataFrame)
    kpis: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    available_years: list[int] = field(default_factory=list)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close_series(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch daily closes from yfinance, returning an empty Series on failure."""
    if not ticker:
        return pd.Series(dtype="float64")
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    except Exception:
        return pd.Series(dtype="float64")
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype="float64")
    series = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if series.empty:
        return pd.Series(dtype="float64")
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    return series[~series.index.duplicated(keep="last")].sort_index()


def _safe_date(value: Any) -> date | None:
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _asof(series: pd.Series | None, day: date | pd.Timestamp) -> float | None:
    """Return the latest known value on/before day."""
    if series is None or series.empty:
        return None
    ts = pd.Timestamp(day).tz_localize(None).normalize()
    values = series.loc[series.index <= ts]
    if values.empty:
        return None
    value = to_float(values.iloc[-1], 0.0)
    return value if value > 0 else None


def _month_ends(start_day: date, end_day: date) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_day).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_day).to_period("M").to_timestamp("M")
    try:
        months = pd.date_range(start=start, end=end, freq="ME")
    except ValueError:
        months = pd.date_range(start=start, end=end, freq="M")
    return list(months) if len(months) else [end]


def _history_window(transactions: list[dict[str, Any]], today: date) -> tuple[str, str]:
    valid_dates = [_safe_date(tx.get("date")) for tx in transactions]
    valid_dates = [d for d in valid_dates if d is not None]
    start_day = min(valid_dates) if valid_dates else today
    start_day = start_day - timedelta(days=10)
    end_day = today + timedelta(days=2)
    return start_day.isoformat(), end_day.isoformat()


def _ticker_price_series(fetch_ticker: str, start: str, end: str) -> tuple[pd.Series, str]:
    """Fetch one symbol, trying KQ when a Korean .KS lookup is empty."""
    series = fetch_close_series(fetch_ticker, start, end)
    used = fetch_ticker
    if series.empty and fetch_ticker.endswith(".KS"):
        alt = fetch_ticker.removesuffix(".KS") + ".KQ"
        alt_series = fetch_close_series(alt, start, end)
        if not alt_series.empty:
            series = alt_series
            used = alt
    return series, used


def _transaction_fx(tx: dict[str, Any], fx_series: pd.Series, warnings: list[str]) -> float | None:
    if tx.get("currency") != "USD" and tx.get("asset_class") != "US":
        return 1.0
    saved_fx = to_float(tx.get("exchange_rate"), 0.0)
    if saved_fx > 0:
        return saved_fx
    tx_day = _safe_date(tx.get("date"))
    fx = _asof(fx_series, tx_day) if tx_day else None
    if fx is not None:
        return fx
    warnings.append(f"{tx.get('date')} {tx.get('ticker')} 거래의 USD/KRW 환율을 찾지 못해 해당 현금흐름을 제외했습니다.")
    return None


def _cashflows_krw(transactions: list[dict[str, Any]], fx_series: pd.Series, warnings: list[str]) -> pd.DataFrame:
    rows = []
    for tx in transactions:
        tx_day = _safe_date(tx.get("date"))
        if tx_day is None:
            continue
        fx = _transaction_fx(tx, fx_series, warnings)
        if fx is None:
            continue
        amount = to_float(tx.get("quantity"), 0.0) * to_float(tx.get("price"), 0.0) * fx
        if amount <= 0:
            continue
        signed = amount if tx.get("side") == "BUY" else -amount
        rows.append({"date": pd.Timestamp(tx_day), "month": pd.Timestamp(tx_day).to_period("M").to_timestamp("M"), "amount_krw": signed})
    return pd.DataFrame(rows, columns=["date", "month", "amount_krw"])


def _portfolio_values(
    transactions: list[dict[str, Any]],
    months: list[pd.Timestamp],
    price_series: dict[str, pd.Series],
    fx_series: pd.Series,
    warnings: list[str],
) -> list[float]:
    values: list[float] = []
    missing_reported: set[str] = set()
    for month_end in months:
        active = [tx for tx in transactions if (_safe_date(tx.get("date")) or date.max) <= month_end.date()]
        month_holdings = summarize_holdings(active)
        total = 0.0
        if month_holdings.empty:
            values.append(0.0)
            continue
        for row in month_holdings.to_dict("records"):
            qty = to_float(row.get("quantity"), 0.0)
            if qty <= 0:
                continue
            fetch_ticker = str(row.get("fetch_ticker") or row.get("ticker") or "")
            close = _asof(price_series.get(fetch_ticker), month_end)
            if close is None:
                key = f"{fetch_ticker}:price"
                if key not in missing_reported:
                    warnings.append(f"{fetch_ticker} 월말 가격이 부족한 구간은 평가액에서 제외했습니다.")
                    missing_reported.add(key)
                continue
            fx = 1.0
            if row.get("currency") == "USD" or row.get("asset_class") == "US":
                fx = _asof(fx_series, month_end) or to_float(row.get("last_exchange_rate"), 0.0)
                if fx <= 0:
                    key = f"{fetch_ticker}:fx"
                    if key not in missing_reported:
                        warnings.append(f"{fetch_ticker} 월말 USD/KRW 환율이 없어 해당 월 평가액에서 제외했습니다.")
                        missing_reported.add(key)
                    continue
            total += qty * close * fx
        values.append(total)
    return values


def _benchmark_values(
    cashflows: pd.DataFrame,
    months: list[pd.Timestamp],
    close_series: pd.Series,
    fx_series: pd.Series,
    is_usd_index: bool,
    label: str,
    warnings: list[str],
) -> list[float]:
    values: list[float] = []
    units = 0.0
    flows = cashflows.sort_values("date").to_dict("records") if not cashflows.empty else []
    flow_idx = 0
    missing_flow_reported = False
    missing_eval_reported = False
    for month_end in months:
        while flow_idx < len(flows) and flows[flow_idx]["date"] <= month_end:
            flow = flows[flow_idx]
            close = _asof(close_series, flow["date"])
            fx = _asof(fx_series, flow["date"]) if is_usd_index else 1.0
            if close and fx:
                invest_amount = flow["amount_krw"] / fx if is_usd_index else flow["amount_krw"]
                units += invest_amount / close
            elif not missing_flow_reported:
                warnings.append(f"{label} 현금흐름 적용일 가격/환율 일부가 없어 해당 거래를 벤치마크 계산에서 제외했습니다.")
                missing_flow_reported = True
            flow_idx += 1
        month_close = _asof(close_series, month_end)
        month_fx = _asof(fx_series, month_end) if is_usd_index else 1.0
        if month_close and month_fx:
            values.append(units * month_close * month_fx)
        else:
            if not missing_eval_reported:
                warnings.append(f"{label} 월말 가격/환율 일부가 없어 직전 계산 가능 값으로 표시했습니다.")
                missing_eval_reported = True
            values.append(values[-1] if values else 0.0)
    return values


def build_performance_result(
    transactions: list[dict[str, Any]] | None,
    priced_holdings: pd.DataFrame | None = None,
    today: date | None = None,
) -> PerformanceResult:
    """Build monthly portfolio, benchmark, and P/L datasets in KRW."""
    today = today or date.today()
    txs = normalize_transactions(transactions or [])
    result = PerformanceResult()
    if not txs:
        result.warnings.append("거래 내역이 없어 성과 분석 그래프를 표시할 데이터가 없습니다.")
        return result

    start, end = _history_window(txs, today)
    months = _month_ends(min((_safe_date(tx.get("date")) or today) for tx in txs), today)
    fx_series = fetch_close_series(USDKRW_TICKER, start, end)
    if fx_series.empty:
        result.warnings.append("USD/KRW 환율 조회에 실패했습니다. 저장된 거래 환율이 없는 USD 현금흐름과 월말 USD 평가는 제외됩니다.")

    tickers = sorted({str(tx.get("fetch_ticker") or "") for tx in txs if tx.get("fetch_ticker")})
    price_series: dict[str, pd.Series] = {}
    for ticker in tickers:
        series, used = _ticker_price_series(ticker, start, end)
        if series.empty:
            result.warnings.append(f"{ticker} 가격 데이터를 조회하지 못했습니다. 해당 종목의 월말 평가액은 가능한 구간만 반영됩니다.")
        price_series[ticker] = series
        if used != ticker:
            result.warnings.append(f"{ticker} 가격이 비어 있어 {used} 데이터를 사용했습니다.")

    kospi_series = fetch_close_series(KOSPI_TICKER, start, end)
    sp500_series = fetch_close_series(SP500_TICKER, start, end)
    if kospi_series.empty:
        result.warnings.append("KOSPI 가격 데이터를 조회하지 못해 KOSPI 비교선이 0 또는 직전값으로 표시될 수 있습니다.")
    if sp500_series.empty:
        result.warnings.append("S&P 500 가격 데이터를 조회하지 못해 S&P 500 비교선이 0 또는 직전값으로 표시될 수 있습니다.")

    cashflows = _cashflows_krw(txs, fx_series, result.warnings)
    cashflow_by_month = cashflows.groupby("month")["amount_krw"].sum() if not cashflows.empty else pd.Series(dtype="float64")
    portfolio_values = _portfolio_values(txs, months, price_series, fx_series, result.warnings)
    kospi_values = _benchmark_values(cashflows, months, kospi_series, fx_series, False, "KOSPI", result.warnings)
    sp500_values = _benchmark_values(cashflows, months, sp500_series, fx_series, True, "S&P 500", result.warnings)

    monthly = pd.DataFrame({"month_end": months})
    monthly["month"] = monthly["month_end"].dt.strftime("%Y-%m")
    monthly["year"] = monthly["month_end"].dt.year
    monthly["month_num"] = monthly["month_end"].dt.month
    monthly["display_month"] = monthly.apply(
        lambda row: f"{int(row['year']) % 100}/{int(row['month_num'])}", axis=1
    )
    monthly["net_investment_krw"] = monthly["month_end"].map(lambda m: to_float(cashflow_by_month.get(m, 0.0), 0.0))
    monthly["cumulative_deposit_krw"] = monthly["net_investment_krw"].cumsum()
    monthly["portfolio_value_krw"] = portfolio_values
    monthly["kospi_value_krw"] = kospi_values
    monthly["sp500_value_krw"] = sp500_values

    latest_deposit = to_float(monthly["cumulative_deposit_krw"].iloc[-1], 0.0) if not monthly.empty else 0.0
    current_portfolio = None
    if priced_holdings is not None and not priced_holdings.empty and "current_value_krw" in priced_holdings.columns:
        current_values = pd.to_numeric(priced_holdings["current_value_krw"], errors="coerce").dropna()
        if not current_values.empty:
            current_portfolio = to_float(current_values.sum(), 0.0)
    if current_portfolio is None and not monthly.empty:
        current_portfolio = to_float(monthly["portfolio_value_krw"].iloc[-1], 0.0)
    if current_portfolio is not None and not monthly.empty:
        monthly.loc[monthly.index[-1], "portfolio_value_krw"] = current_portfolio
    previous_value = monthly["portfolio_value_krw"].shift(1).fillna(0.0)
    monthly["monthly_profit_krw"] = monthly["portfolio_value_krw"] - previous_value - monthly["net_investment_krw"]

    yearly = monthly.groupby("year", as_index=False).agg(
        annual_profit_krw=("monthly_profit_krw", "sum"),
        ending_assets_krw=("portfolio_value_krw", "last"),
    )

    def _return(value: float) -> float:
        return (value / latest_deposit - 1.0) * 100.0 if latest_deposit > 0 and value > 0 else 0.0

    result.monthly = monthly
    result.yearly = yearly
    result.available_years = sorted(monthly["year"].dropna().astype(int).unique().tolist())
    result.kpis = {
        "cumulative_deposit_krw": latest_deposit,
        "portfolio_value_krw": to_float(current_portfolio, 0.0),
        "portfolio_return_pct": _return(to_float(current_portfolio, 0.0)),
        "kospi_value_krw": to_float(monthly["kospi_value_krw"].iloc[-1], 0.0) if not monthly.empty else 0.0,
        "kospi_return_pct": _return(to_float(monthly["kospi_value_krw"].iloc[-1], 0.0)) if not monthly.empty else 0.0,
        "sp500_value_krw": to_float(monthly["sp500_value_krw"].iloc[-1], 0.0) if not monthly.empty else 0.0,
        "sp500_return_pct": _return(to_float(monthly["sp500_value_krw"].iloc[-1], 0.0)) if not monthly.empty else 0.0,
    }
    # Keep warning output compact and deterministic.
    result.warnings = list(dict.fromkeys(result.warnings))[:8]
    return result
