from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from core.firebase import load_data, save_data
from core.sync import _safe_uid
from logic.dividend_ledger import (
    ASSET_CLASS_LABELS,
    build_price_map,
    calculate_target_progress,
    estimate_monthly_dividends,
    normalize_targets,
    normalize_ticker,
    normalize_transactions,
    summarize_holdings,
    to_float,
)
from ui.styles import TOSS_CSS


LEDGER_PATH = "dividend_ledger"
CACHE_TTL_SECONDS = 60 * 60
KST = timezone(timedelta(hours=9))
ASSET_CLASS_OPTIONS = ["US", "KR", "COIN"]
SIDE_OPTIONS = ["BUY", "SELL"]
DEFAULT_LEDGER = {"transactions": [], "targets": [], "settings": {"display_basis": "net"}}


st.markdown(TOSS_CSS, unsafe_allow_html=True)
st.markdown("# 💵 배당금가계부")
st.caption("거래 기록을 기준으로 보유 수량과 월별 예상 배당금을 계산합니다. 기본 표시는 세후 기준입니다.")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_quote_metadata(fetch_tickers: tuple[str, ...]) -> dict[str, dict[str, object]]:
    """Fetch price/name metadata. Korean .KS symbols retry with .KQ.

    Returned keys always match the original requested fetch ticker so downstream
    price fallback can continue using the normalized ledger symbol.
    """
    result: dict[str, dict[str, object]] = {}
    for ticker in fetch_tickers:
        if not ticker:
            continue
        candidates = [ticker]
        if ticker.endswith(".KS"):
            candidates.append(f"{ticker[:-3]}.KQ")

        best_name = ""
        best_price = 0.0
        used_symbol = ticker
        for candidate in candidates:
            try:
                tk = yf.Ticker(candidate)
            except Exception:
                continue

            name = ""
            try:
                info = tk.get_info() or {}
                name = str(info.get("shortName") or info.get("longName") or info.get("symbol") or "").strip()
            except Exception:
                name = ""

            price = 0.0
            try:
                fast_info = getattr(tk, "fast_info", {}) or {}
                price = to_float(fast_info.get("last_price"), 0.0)
            except Exception:
                price = 0.0
            if price <= 0:
                try:
                    hist = tk.history(period="5d", auto_adjust=False)
                    if hist is not None and not hist.empty and "Close" in hist.columns:
                        close = hist["Close"].dropna()
                        if not close.empty:
                            price = to_float(close.iloc[-1], 0.0)
                except Exception:
                    price = 0.0

            if not best_name and name:
                best_name = name
            if price > 0:
                best_price = price
                used_symbol = candidate
                if best_name:
                    break

        result[ticker] = {
            "name": best_name or ticker,
            "price": best_price,
            "used_symbol": used_symbol,
        }
    return result


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_usdkrw_rate() -> float | None:
    hist = yf.Ticker("USDKRW=X").history(period="5d", auto_adjust=False)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    close = hist["Close"].dropna()
    if close.empty:
        return None
    rate = to_float(close.iloc[-1], 0.0)
    return rate if rate > 0 else None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_dividend_histories(fetch_tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    histories: dict[str, pd.Series] = {}
    for ticker in fetch_tickers:
        if not ticker or ticker.endswith("-KRW"):
            continue
        candidates = [ticker]
        if ticker.endswith(".KS"):
            candidates.append(f"{ticker[:-3]}.KQ")
        for candidate in candidates:
            try:
                divs = yf.Ticker(candidate).dividends
            except Exception:
                divs = None
            if divs is not None and not divs.empty:
                histories[ticker] = pd.to_numeric(divs, errors="coerce").dropna().tail(24)
                break
    return histories


def _ledger_state_key() -> str:
    uid = st.session_state.get("user", {}).get("uid", "anonymous")
    return f"dividend_ledger_loaded_{_safe_uid(uid)}"


def load_ledger() -> dict:
    if "user" not in st.session_state:
        return DEFAULT_LEDGER.copy()
    uid = _safe_uid(st.session_state["user"].get("uid", ""))
    raw = load_data(uid, LEDGER_PATH) or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "transactions": normalize_transactions(raw.get("transactions", [])),
        "targets": normalize_targets(raw.get("targets", [])),
        "settings": raw.get("settings", {}) if isinstance(raw.get("settings", {}), dict) else {},
        "_last_sync": raw.get("_last_sync"),
    }


def save_ledger(ledger: dict) -> None:
    if "user" not in st.session_state:
        st.warning("로그인 정보가 없어 저장할 수 없습니다.")
        return
    uid = _safe_uid(st.session_state["user"].get("uid", ""))
    payload = {
        "transactions": normalize_transactions(ledger.get("transactions", [])),
        "targets": normalize_targets(ledger.get("targets", [])),
        "settings": ledger.get("settings", {}),
        "_last_sync": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_data(uid, LEDGER_PATH, payload)
    st.session_state["dividend_ledger"] = payload


def fmt_won(value) -> str:
    value = to_float(value, 0.0)
    return f"{int(round(value)):,}원"


def fmt_money(value, currency: str) -> str:
    value = to_float(value, 0.0)
    if currency == "USD":
        return f"${value:,.2f}"
    return f"{int(round(value)):,}원"


def fmt_quantity(value) -> str:
    value = to_float(value, 0.0)
    return f"{value:,.6f}".rstrip("0").rstrip(".")


def metric_card(label: str, value: str, sub: str = "", accent: bool = False) -> None:
    cls = "toss-metric accent" if accent else "toss-metric"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def first_target_summary(progress: pd.DataFrame) -> tuple[str, str]:
    if progress is None or progress.empty:
        return "-", "목표 수량을 저장해 주세요"
    row = progress.sort_values("progress_pct", ascending=False).iloc[0]
    pct = to_float(row.get("progress_pct"), 0.0)
    return f"{pct:.1f}%", f"{row['ticker']} {fmt_quantity(row['quantity'])} / {fmt_quantity(row['target_quantity'])}주"


if _ledger_state_key() not in st.session_state:
    st.session_state["dividend_ledger"] = load_ledger()
    st.session_state[_ledger_state_key()] = True

ledger = st.session_state.get("dividend_ledger", DEFAULT_LEDGER.copy())
ledger["transactions"] = normalize_transactions(ledger.get("transactions", []))
ledger["targets"] = normalize_targets(ledger.get("targets", []))
ledger.setdefault("settings", {})

basis_label = st.radio(
    "배당 표시 기준",
    ["세후", "세전"],
    index=0 if ledger["settings"].get("display_basis", "net") != "gross" else 1,
    horizontal=True,
    help="월별 예상 배당금 그래프와 요약 카드 표시 기준입니다. 기본값은 세후입니다.",
)
ledger["settings"]["display_basis"] = "gross" if basis_label == "세전" else "net"

with st.form("dividend_ledger_add_target", clear_on_submit=True):
    t1, t2, t3, t4 = st.columns([1, 1.4, 1, 1])
    target_class = t1.selectbox("목표 자산", ASSET_CLASS_OPTIONS, format_func=lambda x: ASSET_CLASS_LABELS[x], key="target_class")
    target_ticker = t2.text_input("목표 티커", value="SCHD", key="target_ticker")
    target_qty = t3.number_input("목표 수량", min_value=0.0, value=3300.0, step=1.0, key="target_qty")
    if t4.form_submit_button("목표 저장", use_container_width=True):
        info = normalize_ticker(target_ticker, target_class)
        ledger["targets"] = [t for t in ledger["targets"] if not (t["asset_class"] == info.asset_class and t["ticker"] == info.display_ticker)]
        ledger["targets"].append({"asset_class": info.asset_class, "ticker": info.display_ticker, "target_quantity": target_qty})
        save_ledger(ledger)
        st.toast("목표가 저장되었습니다.", icon="✅")
        st.rerun()

holdings = summarize_holdings(ledger["transactions"])
fetch_tickers = tuple(sorted(set(holdings["fetch_ticker"].dropna().tolist()))) if not holdings.empty else tuple()
quote_error = None
fx_error = None
try:
    quote_metadata = fetch_quote_metadata(fetch_tickers)
except Exception as exc:
    quote_metadata = {}
    quote_error = str(exc)
fetched_prices = {ticker: to_float(data.get("price"), 0.0) for ticker, data in quote_metadata.items()}
try:
    usdkrw = fetch_usdkrw_rate()
except Exception as exc:
    usdkrw = None
    fx_error = str(exc)
priced_holdings = build_price_map(holdings, fetched_prices, usdkrw)
try:
    dividend_histories = fetch_dividend_histories(fetch_tickers)
except Exception:
    dividend_histories = {}
monthly = estimate_monthly_dividends(priced_holdings, dividend_histories, usdkrw)
value_col = "gross_krw" if ledger["settings"].get("display_basis") == "gross" else "net_krw"
basis_text = "세전" if value_col == "gross_krw" else "세후"
progress = calculate_target_progress(holdings, ledger["targets"])
target_value, target_sub = first_target_summary(progress)

if quote_error:
    st.warning(f"현재가/종목명 조회 일부 실패: 가격은 마지막 거래 단가를 fallback으로 사용합니다. ({quote_error})")
if fx_error or usdkrw is None:
    st.info("환율 조회 실패 시 임의 고정 환율을 쓰지 않습니다. USD 종목 원화 환산은 환율 조회가 가능할 때만 계산됩니다.")
else:
    st.caption(f"환율은 USD 종목 원화 환산 계산에만 사용됩니다. USD/KRW {usdkrw:,.2f}")

st.markdown("### 📌 요약")
m1, m2, m3, m4 = st.columns(4)
convertible_values = pd.to_numeric(priced_holdings.get("current_value_krw", pd.Series(dtype="float64")), errors="coerce").dropna() if not priced_holdings.empty else pd.Series(dtype="float64")
monthly_total = to_float(monthly[value_col].sum(), 0.0) if not monthly.empty else 0.0
with m1:
    metric_card("평가금액(환산 가능분)", fmt_won(convertible_values.sum()), "USD 환율 없으면 해당 종목 제외", True)
with m2:
    metric_card(f"연간 예상 배당({basis_text})", fmt_won(monthly_total), "최근 배당 이력 기반 추정")
with m3:
    metric_card(f"월평균 예상 배당({basis_text})", fmt_won(monthly_total / 12 if monthly_total else 0), "월별 예상 배당금 기준")
with m4:
    metric_card("목표 달성률", target_value, target_sub)

st.markdown("### 📈 월별 예상 배당금")
chart_df = monthly.copy()
chart_df["월"] = [str(i) for i in range(1, 13)]
chart_df["예상 배당금"] = chart_df[value_col]
fig = px.bar(chart_df, x="월", y="예상 배당금", text="예상 배당금", hover_data={"월": True, "예상 배당금": ":,.0f"})
fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", marker_color="#3182F6")
fig.update_layout(
    height=360,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis_title="원",
    xaxis_title="월",
    xaxis={"type": "category", "categoryorder": "array", "categoryarray": [str(i) for i in range(1, 13)]},
)
st.plotly_chart(fig, use_container_width=True)
st.caption("배당 이력이 제공되는 종목의 과거 월별 배당을 보유 수량에 적용한 추정치입니다. 실제 지급액과 지급월은 달라질 수 있습니다.")

st.markdown("### 🎯 목표 수량")
if progress.empty:
    st.caption("저장된 목표 수량이 없습니다. 예: SCHD 목표 3300주")
else:
    for row in progress.to_dict("records"):
        pct = to_float(row["progress_pct"], 0.0)
        st.progress(
            min(pct / 100.0, 1.0),
            text=(
                f"{row['ticker']} {fmt_quantity(row['quantity'])} / {fmt_quantity(row['target_quantity'])}주"
                f" · 남은 수량 {fmt_quantity(row['remaining_quantity'])}주 · 달성률 {pct:.1f}%"
            ),
        )

st.markdown("### ✍️ 거래 입력")
with st.form("dividend_ledger_add_transaction", clear_on_submit=True):
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.2, 1, 1, 1])
    asset_class = c1.selectbox("자산 구분", ASSET_CLASS_OPTIONS, format_func=lambda x: ASSET_CLASS_LABELS[x])
    ticker = c2.text_input("티커", value="SCHD", help="US: SCHD/TQQQ/MSFT · KR: 069500/458730 · COIN: BTC")
    side = c3.selectbox("구분", SIDE_OPTIONS, format_func=lambda x: "매수" if x == "BUY" else "매도")
    quantity = c4.number_input("수량", min_value=0.0, value=0.0, step=1.0, format="%.6f")
    price = c5.number_input("거래 단가", min_value=0.0, value=0.0, step=1.0, format="%.4f")

    submitted = st.form_submit_button("➕ 거래 추가", type="primary", use_container_width=True)
    if submitted:
        info = normalize_ticker(ticker, asset_class)
        item = {
            "id": uuid4().hex,
            "date": datetime.now(KST).date().isoformat(),
            "asset_class": asset_class,
            "ticker": info.display_ticker,
            "side": side,
            "quantity": quantity,
            "price": price,
        }
        normalized = normalize_transactions([item])
        if not normalized:
            st.error("티커와 수량을 확인해 주세요.")
        else:
            ledger["transactions"].extend(normalized)
            ledger["transactions"] = normalize_transactions(ledger["transactions"])
            save_ledger(ledger)
            st.toast("거래가 저장되었습니다.", icon="✅")
            st.rerun()

st.markdown("### 📒 보유 현황")
if priced_holdings.empty:
    st.info("아직 거래 내역이 없습니다.")
else:
    display_holdings = priced_holdings.copy()
    display_holdings["자산"] = display_holdings["asset_class"].map(ASSET_CLASS_LABELS)
    display_holdings["종목명"] = [
        str(quote_metadata.get(fetch_ticker, {}).get("name") or ticker)
        for fetch_ticker, ticker in zip(display_holdings["fetch_ticker"], display_holdings["ticker"])
    ]
    display_holdings["수량"] = display_holdings["quantity"].apply(fmt_quantity)
    display_holdings["평균단가"] = [fmt_money(v, c) for v, c in zip(display_holdings["avg_cost"], display_holdings["currency"])]
    display_holdings["현재가"] = [fmt_money(v, c) for v, c in zip(display_holdings["current_price"], display_holdings["currency"])]
    display_holdings["가격출처"] = display_holdings["price_source"].map({"current": "현재가", "last_trade": "마지막 거래 단가"})
    display_holdings["평가금액(KRW)"] = display_holdings["current_value_krw"].apply(lambda x: fmt_won(x) if pd.notna(x) else "환율 필요")
    st.dataframe(
        display_holdings[["자산", "ticker", "종목명", "수량", "평균단가", "현재가", "가격출처", "평가금액(KRW)"]],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### 💼 거래 내역 관리")
if ledger["transactions"]:
    tx_df = pd.DataFrame(ledger["transactions"])
    tx_df["delete"] = False
    edited = st.data_editor(
        tx_df[["delete", "date", "asset_class", "ticker", "side", "quantity", "price", "id"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "delete": st.column_config.CheckboxColumn("삭제"),
            "date": st.column_config.TextColumn("거래일", disabled=True),
            "asset_class": st.column_config.SelectboxColumn("자산 구분", options=ASSET_CLASS_OPTIONS),
            "side": st.column_config.SelectboxColumn("구분", options=SIDE_OPTIONS),
            "id": st.column_config.TextColumn("id", disabled=True),
        },
    )
    c_save, c_delete = st.columns(2)
    if c_save.button("💾 거래 내역 수정 저장", use_container_width=True):
        rows = edited.drop(columns=["delete"]).to_dict("records")
        ledger["transactions"] = normalize_transactions(rows)
        save_ledger(ledger)
        st.toast("거래 내역을 저장했습니다.", icon="✅")
        st.rerun()
    if c_delete.button("🗑️ 선택 거래 삭제", use_container_width=True):
        keep = edited[edited["delete"] != True].drop(columns=["delete"]).to_dict("records")  # noqa: E712
        ledger["transactions"] = normalize_transactions(keep)
        save_ledger(ledger)
        st.toast("선택한 거래를 삭제했습니다.", icon="✅")
        st.rerun()
else:
    st.caption("거래 내역이 입력되면 여기에서 수정/삭제할 수 있습니다.")

last_sync = ledger.get("_last_sync")
if last_sync:
    st.caption(f"마지막 저장: {last_sync}")
