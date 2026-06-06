"""Dividend ledger calculation helpers for the Streamlit dividend ledger page.

The functions in this module intentionally avoid Streamlit and Firebase so the
page can keep persistence/UI concerns separate from reusable calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


DEFAULT_TAX_RATES = {
    "US": 0.15,
    "KR": 0.154,
    "COIN": 0.0,
}

CURRENCY_BY_ASSET_CLASS = {
    "US": "USD",
    "KR": "KRW",
    "COIN": "KRW",
}

ASSET_CLASS_LABELS = {
    "US": "미국 주식/ETF",
    "KR": "국내 주식/ETF",
    "COIN": "코인",
}


@dataclass(frozen=True)
class TickerInfo:
    """Normalized ticker metadata used by price/dividend fetchers."""

    asset_class: str
    input_ticker: str
    display_ticker: str
    fetch_ticker: str
    currency: str


def to_float(value: Any, default: float = 0.0) -> float:
    """Convert arbitrary user/Firebase values to a finite float."""
    if value is None or isinstance(value, bool):
        return default
    try:
        result = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default
    if pd.isna(result):
        return default
    return result


def normalize_asset_class(asset_class: Any) -> str:
    """Return one of US/KR/COIN, accepting Korean UI labels as input."""
    raw = str(asset_class or "US").strip().upper()
    if "코인" in raw or raw in {"COIN", "CRYPTO"}:
        return "COIN"
    if "국내" in raw or raw in {"KR", "KOREA", "KOR"}:
        return "KR"
    return "US"


def normalize_ticker(ticker: Any, asset_class: Any = "US") -> TickerInfo:
    """Normalize user input for display and yfinance fetching.

    Important project rule: coin BTC is always treated as BTC-KRW, never BTC-USD.
    Korean numeric tickers are fetched via the Yahoo ``.KS`` suffix while the UI
    keeps the original six-digit code such as ``069500``.
    """
    cls = normalize_asset_class(asset_class)
    raw = str(ticker or "").strip().upper().replace(" ", "")

    if cls == "COIN":
        base = raw.replace("-USD", "").replace("-KRW", "") or "BTC"
        if base == "BTC":
            display = "BTC-KRW"
            fetch = "BTC-KRW"
        else:
            display = f"{base}-KRW" if "-" not in base else base
            fetch = display
        return TickerInfo(cls, raw, display, fetch, "KRW")

    if cls == "KR":
        display = raw.removesuffix(".KS").removesuffix(".KQ")
        fetch = raw if raw.endswith((".KS", ".KQ")) else f"{display}.KS"
        return TickerInfo(cls, raw, display, fetch, "KRW")

    display = raw
    return TickerInfo(cls, raw, display, display, "USD")


def normalize_transaction(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Clean one transaction row. Invalid/empty rows return None."""
    info = normalize_ticker(raw.get("ticker"), raw.get("asset_class", "US"))
    if not info.display_ticker:
        return None

    side = str(raw.get("side") or "BUY").strip().upper()
    if "매도" in side or side == "SELL":
        side = "SELL"
    else:
        side = "BUY"

    # ``date`` is the 기준일 used as the 매수일/매도일 for performance charts,
    # not a separate registration timestamp. Missing or malformed legacy values
    # fall back to today to keep existing Firebase rows loadable.
    tx_date = raw.get("date") or date.today().isoformat()
    if isinstance(tx_date, (datetime, date)):
        tx_date = tx_date.isoformat()[:10]
    else:
        try:
            tx_date = pd.to_datetime(tx_date).date().isoformat()
        except Exception:
            tx_date = date.today().isoformat()

    quantity = abs(to_float(raw.get("quantity")))
    price = abs(to_float(raw.get("price")))
    exchange_rate = to_float(raw.get("exchange_rate"), 0.0)
    if quantity <= 0:
        return None

    return {
        "id": str(raw.get("id") or f"{tx_date}-{info.display_ticker}-{datetime.now().timestamp()}"),
        "date": tx_date,
        "asset_class": info.asset_class,
        "ticker": info.display_ticker,
        "fetch_ticker": info.fetch_ticker,
        "name": str(raw.get("name") or "").strip(),
        "side": side,
        "quantity": quantity,
        "price": price,
        "currency": info.currency,
        "exchange_rate": exchange_rate,
        "memo": str(raw.get("memo") or "").strip(),
    }


def normalize_transactions(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize and date-sort all valid transaction rows."""
    cleaned = []
    for row in rows or []:
        if isinstance(row, dict):
            item = normalize_transaction(row)
            if item:
                cleaned.append(item)
    return sorted(cleaned, key=lambda x: (x["date"], x["ticker"], x["id"]))


def summarize_holdings(transactions: list[dict[str, Any]]) -> pd.DataFrame:
    """Compute current quantity, average cost, and last trade fallback per ticker."""
    txs = normalize_transactions(transactions)
    positions: dict[tuple[str, str], dict[str, Any]] = {}

    for tx in txs:
        key = (tx["asset_class"], tx["ticker"])
        pos = positions.setdefault(
            key,
            {
                "asset_class": tx["asset_class"],
                "ticker": tx["ticker"],
                "fetch_ticker": tx["fetch_ticker"],
                "name": tx.get("name", ""),
                "currency": tx["currency"],
                "quantity": 0.0,
                "cost_amount": 0.0,
                "last_trade_price": 0.0,
                "last_exchange_rate": 0.0,
            },
        )
        qty = tx["quantity"]
        price = tx["price"]
        if tx["side"] == "BUY":
            pos["quantity"] += qty
            pos["cost_amount"] += qty * price
        else:
            sell_qty = min(qty, pos["quantity"])
            avg = pos["cost_amount"] / pos["quantity"] if pos["quantity"] > 0 else 0.0
            pos["quantity"] -= sell_qty
            pos["cost_amount"] -= avg * sell_qty
        if price > 0:
            pos["last_trade_price"] = price
        if tx.get("exchange_rate", 0) > 0:
            pos["last_exchange_rate"] = tx["exchange_rate"]
        if tx.get("name"):
            pos["name"] = tx["name"]

    rows = []
    for pos in positions.values():
        if pos["quantity"] <= 1e-12:
            continue
        avg_cost = pos["cost_amount"] / pos["quantity"] if pos["quantity"] else 0.0
        rows.append({**pos, "avg_cost": avg_cost})
    columns = [
        "asset_class", "ticker", "fetch_ticker", "name", "currency", "quantity",
        "avg_cost", "cost_amount", "last_trade_price", "last_exchange_rate",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["asset_class", "ticker"]) if rows else pd.DataFrame(columns=columns)


def build_price_map(
    holdings: pd.DataFrame,
    fetched_prices: dict[str, float] | None = None,
    usdkrw: float | None = None,
) -> pd.DataFrame:
    """Attach current prices to holdings using last trade price as API fallback.

    A failed current-price lookup must not become 0 KRW. If fetched price is
    missing/invalid, the holding's last transaction unit price is used first,
    then average cost is used as a final calculable fallback.
    """
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    fetched_prices = fetched_prices or {}
    rows = []
    for row in holdings.to_dict("records"):
        fetch_ticker = row.get("fetch_ticker") or row.get("ticker")
        fetched = to_float(fetched_prices.get(fetch_ticker), 0.0)
        last_trade = to_float(row.get("last_trade_price"), 0.0)
        avg_cost = to_float(row.get("avg_cost"), 0.0)
        fallback = last_trade if last_trade > 0 else avg_cost
        current_price = fetched if fetched > 0 else fallback
        price_source = "current" if fetched > 0 else ("last_trade" if last_trade > 0 else "avg_cost")
        currency = row.get("currency")
        current_usdkrw = to_float(usdkrw, 0.0)
        fx = 1.0 if currency == "KRW" else current_usdkrw or to_float(row.get("last_exchange_rate"), 0.0)
        current_value = current_price * to_float(row.get("quantity")) if current_price > 0 else None
        current_value_krw = current_value * fx if current_value is not None and fx > 0 else None
        if current_value is None:
            current_value_usd = None
        elif currency == "USD":
            current_value_usd = current_value
        elif current_usdkrw > 0:
            current_value_usd = current_value / current_usdkrw
        else:
            current_value_usd = None
        rows.append({
            **row,
            "current_price": current_price,
            "price_source": price_source,
            "fx_rate": fx,
            "current_value": current_value,
            "current_value_krw": current_value_krw,
            "current_value_usd": current_value_usd,
        })
    return pd.DataFrame(rows)


def estimate_monthly_dividends(
    holdings: pd.DataFrame,
    dividend_history: dict[str, pd.Series] | None = None,
    usdkrw: float | None = None,
    tax_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate next-12-month dividends from trailing per-share dividends.

    The output is intentionally labelled as an estimate, not a confirmed
    dividend schedule. Historical monthly per-share payouts are shifted into the
    coming 12 calendar months. Missing dividend histories produce 0 estimates.
    """
    dividend_history = dividend_history or {}
    tax_rates = {**DEFAULT_TAX_RATES, **(tax_rates or {})}
    month_numbers = list(range(1, 13))
    result = pd.DataFrame({"month": [str(m) for m in month_numbers], "gross_krw": 0.0, "net_krw": 0.0})
    if holdings is None or holdings.empty:
        return result

    for row in holdings.to_dict("records"):
        qty = to_float(row.get("quantity"))
        if qty <= 0:
            continue
        series = dividend_history.get(row.get("fetch_ticker"))
        if series is None or len(series) == 0:
            continue
        divs = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
        if divs.empty:
            continue
        divs.index = pd.to_datetime(divs.index, errors="coerce")
        divs = divs[~divs.index.isna()].sort_index().tail(24)
        if divs.empty:
            continue
        monthly = divs.groupby(divs.index.month).sum()
        fx = 1.0 if row.get("currency") == "KRW" else to_float(usdkrw, 0.0) or to_float(row.get("last_exchange_rate"), 0.0)
        if fx <= 0:
            continue
        tax_rate = tax_rates.get(row.get("asset_class"), 0.0)
        for idx, month_number in enumerate(month_numbers):
            per_share = to_float(monthly.get(month_number), 0.0)
            gross = per_share * qty * fx
            result.loc[idx, "gross_krw"] += gross
            result.loc[idx, "net_krw"] += gross * (1.0 - tax_rate)
    return result


def normalize_targets(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Clean target-share rows used for progress cards."""
    cleaned = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        info = normalize_ticker(raw.get("ticker"), raw.get("asset_class", "US"))
        target_qty = to_float(raw.get("target_quantity"), 0.0)
        if info.display_ticker and target_qty > 0:
            cleaned.append({"asset_class": info.asset_class, "ticker": info.display_ticker, "fetch_ticker": info.fetch_ticker, "target_quantity": target_qty})
    return cleaned


def _target_price_usd(
    target: dict[str, Any],
    enriched_holdings: pd.DataFrame,
    usdkrw: float | None = None,
    fetched_prices: dict[str, float] | None = None,
) -> float:
    """Return the target symbol's current unit price in USD when calculable."""
    fetched_prices = fetched_prices or {}
    asset_class = normalize_asset_class(target.get("asset_class"))
    fetch_ticker = target.get("fetch_ticker") or normalize_ticker(target.get("ticker"), asset_class).fetch_ticker
    display_ticker = target.get("ticker") or normalize_ticker(fetch_ticker, asset_class).display_ticker

    price = to_float(fetched_prices.get(fetch_ticker), 0.0)
    if price <= 0 and enriched_holdings is not None and not enriched_holdings.empty:
        matches = enriched_holdings[
            (enriched_holdings.get("asset_class") == asset_class)
            & (enriched_holdings.get("ticker") == display_ticker)
        ]
        if not matches.empty:
            row = matches.iloc[0]
            price = to_float(row.get("current_price"), 0.0)
            if price <= 0:
                price = to_float(row.get("last_trade_price"), 0.0) or to_float(row.get("avg_cost"), 0.0)
    if price <= 0:
        return 0.0
    if asset_class == "US":
        return price
    fx = to_float(usdkrw, 0.0)
    return price / fx if fx > 0 else 0.0


def compute_goal_achievement(
    goal: dict[str, Any] | None,
    enriched_holdings: pd.DataFrame,
    usdkrw: float | None = None,
    fetched_prices: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute goal achievement using total portfolio USD value.

    The achievement denominator is the target symbol's target quantity converted
    to a USD goal amount at the current target-symbol price. The numerator is
    the sum of every registered holding's USD valuation. KR/COIN holdings require
    a current USDKRW rate for this USD-standard calculation.
    """
    targets = normalize_targets([goal] if goal else [])
    if not targets:
        return {"ok": False, "error": "저장된 목표 수량이 없습니다"}
    target = targets[0]
    target_qty = to_float(target.get("target_quantity"), 0.0)
    if target_qty <= 0:
        return {"ok": False, "error": "저장된 목표 수량이 없습니다"}

    holdings_records = [] if enriched_holdings is None or enriched_holdings.empty else enriched_holdings.to_dict("records")
    fx = to_float(usdkrw, 0.0)
    needs_fx = target["asset_class"] != "US" or any(normalize_asset_class(row.get("asset_class")) != "US" for row in holdings_records)
    if needs_fx and fx <= 0:
        return {"ok": False, "error": "현재 USD 환율 조회 불가"}

    target_price_usd = _target_price_usd(target, enriched_holdings, usdkrw, fetched_prices)
    if target_price_usd <= 0:
        return {"ok": False, "error": "목표 종목 현재가 조회 불가"}

    portfolio_amount_usd = 0.0
    actual_target_qty = 0.0
    for row in holdings_records:
        qty = to_float(row.get("quantity"), 0.0)
        if normalize_asset_class(row.get("asset_class")) == target["asset_class"] and row.get("ticker") == target["ticker"]:
            actual_target_qty += qty
        value_usd = row.get("current_value_usd")
        if pd.notna(value_usd):
            portfolio_amount_usd += to_float(value_usd, 0.0)

    target_amount_usd = target_price_usd * target_qty
    equivalent_target_qty = portfolio_amount_usd / target_price_usd if target_price_usd > 0 else 0.0
    return {
        "ok": True,
        "target_symbol": target["ticker"],
        "target_asset_class": target["asset_class"],
        "target_quantity": target_qty,
        "target_price_usd": target_price_usd,
        "target_amount_usd": target_amount_usd,
        "portfolio_amount_usd": portfolio_amount_usd,
        "achievement_pct": min(portfolio_amount_usd / target_amount_usd * 100.0, 100.0) if target_amount_usd > 0 else 0.0,
        "actual_target_symbol_quantity": actual_target_qty,
        "equivalent_target_quantity": equivalent_target_qty,
        "remaining_actual_quantity": max(target_qty - actual_target_qty, 0.0),
        "remaining_equivalent_quantity": max(target_qty - equivalent_target_qty, 0.0),
        "remaining_amount_usd": max(target_amount_usd - portfolio_amount_usd, 0.0),
    }


def calculate_target_progress(holdings: pd.DataFrame, targets: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Return target quantity progress per ticker."""
    targets = normalize_targets(targets)
    if not targets:
        return pd.DataFrame(columns=["asset_class", "ticker", "target_quantity", "quantity", "progress_pct", "remaining_quantity"])
    qty_map = {}
    if holdings is not None and not holdings.empty:
        for row in holdings.to_dict("records"):
            qty_map[(row.get("asset_class"), row.get("ticker"))] = to_float(row.get("quantity"))
    rows = []
    for target in targets:
        qty = qty_map.get((target["asset_class"], target["ticker"]), 0.0)
        target_qty = target["target_quantity"]
        rows.append({**target, "quantity": qty, "progress_pct": min(qty / target_qty * 100.0, 100.0), "remaining_quantity": max(target_qty - qty, 0.0)})
    return pd.DataFrame(rows)
