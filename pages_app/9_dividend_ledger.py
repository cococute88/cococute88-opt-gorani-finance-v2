from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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
st.caption("거래 기록을 기준으로 보유 수량과 월별 예상 배당 추이를 계산합니다. 기본 표시는 세후 기준입니다.")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_current_prices(fetch_tickers: tuple[str, ...]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker in fetch_tickers:
        if not ticker:
            continue
        price = 0.0
        tk = yf.Ticker(ticker)
        fast_info = getattr(tk, "fast_info", {}) or {}
        try:
            price = to_float(fast_info.get("last_price"), 0.0)
        except Exception:
            price = 0.0
        if price <= 0:
            hist = tk.history(period="5d", auto_adjust=False)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                price = to_float(hist["Close"].dropna().iloc[-1], 0.0)
        if price > 0:
            prices[ticker] = price
    return prices


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_usdkrw_rate() -> float | None:
    hist = yf.Ticker("USDKRW=X").history(period="5d", auto_adjust=False)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    rate = to_float(hist["Close"].dropna().iloc[-1], 0.0)
    return rate if rate > 0 else None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_dividend_histories(fetch_tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    histories: dict[str, pd.Series] = {}
    for ticker in fetch_tickers:
        if not ticker or ticker.endswith("-KRW"):
            continue
        divs = yf.Ticker(ticker).dividends
        if divs is not None and not divs.empty:
            histories[ticker] = pd.to_numeric(divs, errors="coerce").dropna().tail(24)
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
    help="월별 예상 배당 그래프와 요약 카드 표시 기준입니다. 기본값은 세후입니다.",
)
ledger["settings"]["display_basis"] = "gross" if basis_label == "세전" else "net"

st.markdown("### ✍️ 거래 입력")
with st.form("dividend_ledger_add_transaction", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([1.1, 1, 1.2, 1])
    tx_date = c1.date_input("거래일", value=date.today())
    asset_class = c2.selectbox("자산 구분", ASSET_CLASS_OPTIONS, format_func=lambda x: ASSET_CLASS_LABELS[x])
    ticker = c3.text_input("티커", value="SCHD", help="국내 예: 069500 / 코인 BTC는 BTC-KRW로 저장됩니다.")
    side = c4.selectbox("구분", SIDE_OPTIONS, format_func=lambda x: "매수" if x == "BUY" else "매도")

    c5, c6, c7, c8 = st.columns([1, 1, 1, 1.4])
    quantity = c5.number_input("수량", min_value=0.0, value=0.0, step=1.0, format="%.6f")
    price = c6.number_input("거래 단가", min_value=0.0, value=0.0, step=1.0, format="%.4f")
    exchange_rate = c7.number_input("적용 환율", min_value=0.0, value=0.0, step=1.0, help="USD 거래면 입력 권장. 환율 조회 실패 시 이 값을 fallback으로 씁니다.")
    name = c8.text_input("종목명/메모", value="")

    submitted = st.form_submit_button("➕ 거래 추가", type="primary", use_container_width=True)
    if submitted:
        info = normalize_ticker(ticker, asset_class)
        item = {
            "id": uuid4().hex,
            "date": tx_date.isoformat(),
            "asset_class": asset_class,
            "ticker": info.display_ticker,
            "name": name,
            "side": side,
            "quantity": quantity,
            "price": price,
            "exchange_rate": exchange_rate,
            "memo": name,
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

holdings = summarize_holdings(ledger["transactions"])
fetch_tickers = tuple(sorted(set(holdings["fetch_ticker"].dropna().tolist()))) if not holdings.empty else tuple()
price_error = None
fx_error = None
try:
    fetched_prices = fetch_current_prices(fetch_tickers)
except Exception as exc:
    fetched_prices = {}
    price_error = str(exc)
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

if price_error:
    st.warning(f"현재가 조회 일부 실패: 마지막 거래 단가를 fallback으로 사용합니다. ({price_error})")
if fx_error or usdkrw is None:
    st.info("환율 조회 실패 시 임의 고정 환율을 쓰지 않고, 거래 입력 환율이 있는 USD 종목만 원화 환산합니다.")

st.markdown("### 📌 요약")
m1, m2, m3, m4 = st.columns(4)
convertible_values = pd.to_numeric(priced_holdings.get("current_value_krw", pd.Series(dtype="float64")), errors="coerce").dropna() if not priced_holdings.empty else pd.Series(dtype="float64")
monthly_total = to_float(monthly[value_col].sum(), 0.0) if not monthly.empty else 0.0
with m1:
    metric_card("평가금액(환산 가능분)", fmt_won(convertible_values.sum()), "USD 환율 없으면 해당 종목 제외", True)
with m2:
    metric_card(f"연간 예상 배당({basis_text})", fmt_won(monthly_total), "최근 배당 이력 기반 추정")
with m3:
    metric_card(f"월 평균 예상 배당({basis_text})", fmt_won(monthly_total / 12 if monthly_total else 0), "월별 예상 배당 추이 기준")
with m4:
    metric_card("USD/KRW", f"{usdkrw:,.2f}" if usdkrw else "-", "조회 실패 시 고정값 미사용")

st.markdown("### 📈 월별 예상 배당 추이")
chart_df = monthly.copy()
chart_df["예상 배당"] = chart_df[value_col]
fig = px.bar(chart_df, x="month", y="예상 배당", text="예상 배당")
fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", marker_color="#3182F6")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="KRW", xaxis_title="월")
st.plotly_chart(fig, use_container_width=True)
st.caption("배당 이력이 제공되는 종목의 과거 월별 배당을 보유 수량에 적용한 추정치입니다. 실제 지급액과 지급월은 달라질 수 있습니다.")

st.markdown("### 🎯 목표 수량")
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

progress = calculate_target_progress(holdings, ledger["targets"])
if progress.empty:
    st.caption("저장된 목표 수량이 없습니다. 예: SCHD 목표 3300주")
else:
    for row in progress.to_dict("records"):
        st.progress(min(to_float(row["progress_pct"]) / 100.0, 1.0), text=f"{row['ticker']} {row['quantity']:,.4g} / {row['target_quantity']:,.4g}주 · 남은 수량 {row['remaining_quantity']:,.4g}주")

st.markdown("### 📒 보유 현황")
if priced_holdings.empty:
    st.info("아직 거래 내역이 없습니다.")
else:
    display_holdings = priced_holdings.copy()
    display_holdings["자산"] = display_holdings["asset_class"].map(ASSET_CLASS_LABELS)
    display_holdings["수량"] = display_holdings["quantity"]
    display_holdings["평균단가"] = [fmt_money(v, c) for v, c in zip(display_holdings["avg_cost"], display_holdings["currency"])]
    display_holdings["현재가"] = [fmt_money(v, c) for v, c in zip(display_holdings["current_price"], display_holdings["currency"])]
    display_holdings["가격출처"] = display_holdings["price_source"].map({"current": "현재가", "last_trade": "마지막 거래 단가"})
    display_holdings["평가금액(KRW)"] = display_holdings["current_value_krw"].apply(lambda x: fmt_won(x) if pd.notna(x) else "환율 필요")
    st.dataframe(
        display_holdings[["자산", "ticker", "name", "수량", "평균단가", "현재가", "가격출처", "평가금액(KRW)"]],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### 🗂 거래 내역 관리")
if ledger["transactions"]:
    tx_df = pd.DataFrame(ledger["transactions"])
    tx_df["delete"] = False
    edited = st.data_editor(
        tx_df[["delete", "date", "asset_class", "ticker", "name", "side", "quantity", "price", "exchange_rate", "memo", "id"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "delete": st.column_config.CheckboxColumn("삭제"),
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
