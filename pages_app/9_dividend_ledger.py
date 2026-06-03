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
    compute_goal_achievement,
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
DEFAULT_LEDGER = {"transactions": [], "targets": [], "settings": {"display_basis": "net"}}


st.markdown(TOSS_CSS, unsafe_allow_html=True)
st.markdown("# 💵 배당금가계부")
st.caption("거래 기록을 기준으로 보유 수량과 월별 예상 배당금을 계산합니다. 기본 표시는 세후 기준입니다.")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_current_prices(fetch_tickers: tuple[str, ...]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker in fetch_tickers:
        if not ticker:
            continue
        price = 0.0
        lookup_tickers = [ticker]
        if ticker.endswith(".KS"):
            lookup_tickers.append(ticker.removesuffix(".KS") + ".KQ")
        for lookup_ticker in lookup_tickers:
            tk = yf.Ticker(lookup_ticker)
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
                break
    return prices


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_symbol_names(fetch_tickers: tuple[str, ...]) -> dict[str, str]:
    names: dict[str, str] = {}
    for ticker in fetch_tickers:
        if not ticker:
            continue
        lookup_tickers = [ticker]
        if ticker.endswith(".KS"):
            lookup_tickers.append(ticker.removesuffix(".KS") + ".KQ")
        for lookup_ticker in lookup_tickers:
            try:
                info = yf.Ticker(lookup_ticker).get_info() or {}
            except Exception:
                info = {}
            name = str(info.get("shortName") or info.get("longName") or "").strip()
            if name:
                names[ticker] = name
                break
    return names


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


def fmt_remaining_money(value, currency: str) -> str:
    value = int(round(to_float(value, 0.0)))
    if currency == "USD":
        return f"${value:,}"
    return f"₩{value:,}"


def fmt_quantity(value) -> str:
    value = to_float(value, 0.0)
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return text or "0"


def fmt_pct(value) -> str:
    return f"{to_float(value, 0.0):.1f}%"


def fmt_usd_whole(value) -> str:
    return f"${int(round(to_float(value, 0.0))):,}"


def fmt_goal_quantity(value) -> str:
    value = to_float(value, 0.0)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.1f}".rstrip("0").rstrip(".")


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


def build_target_summary(goal_achievement: dict, priced_holdings: pd.DataFrame) -> tuple[str, str]:
    if goal_achievement.get("ok"):
        return (
            fmt_pct(goal_achievement.get("achievement_pct")),
            (
                f"{fmt_goal_quantity(goal_achievement.get('equivalent_target_quantity'))}"
                f"({fmt_goal_quantity(goal_achievement.get('actual_target_symbol_quantity'))})"
                f" / {fmt_goal_quantity(goal_achievement.get('target_quantity'))}주"
            ),
        )
    if goal_achievement.get("error"):
        return "-", str(goal_achievement.get("error"))
    if priced_holdings is not None and not priced_holdings.empty:
        total = pd.to_numeric(priced_holdings.get("current_value_krw", pd.Series(dtype="float64")), errors="coerce").dropna().sum()
        return "-", f"현재 평가금액 {fmt_won(total)}"
    return "-", "저장된 목표 수량이 없습니다"


def add_weight_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    result = df.copy()
    values = pd.to_numeric(result.get("current_value_krw", pd.Series(dtype="float64")), errors="coerce")
    total = values.dropna().sum()
    result["weight_pct"] = values.apply(lambda value: (value / total * 100.0) if pd.notna(value) and total > 0 else 0.0)
    return result


if _ledger_state_key() not in st.session_state:
    st.session_state["dividend_ledger"] = load_ledger()
    st.session_state[_ledger_state_key()] = True

ledger = st.session_state.get("dividend_ledger", DEFAULT_LEDGER.copy())
ledger["transactions"] = normalize_transactions(ledger.get("transactions", []))
ledger["targets"] = normalize_targets(ledger.get("targets", []))
ledger.setdefault("settings", {})

control_left, control_right = st.columns([1.3, 2.7])
with control_left:
    basis_label = st.radio(
        "배당 표시 기준",
        ["세후", "세전"],
        index=0 if ledger["settings"].get("display_basis", "net") != "gross" else 1,
        horizontal=True,
        help="월별 예상 배당 그래프와 요약 카드 표시 기준입니다. 기본값은 세후입니다.",
    )
ledger["settings"]["display_basis"] = "gross" if basis_label == "세전" else "net"
with control_right:
    with st.form("dividend_ledger_add_target", clear_on_submit=True):
        t1, t2, t3, t4 = st.columns([1, 1.2, 1, 0.8])
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
holding_fetch_tickers = holdings["fetch_ticker"].dropna().tolist() if not holdings.empty else []
target_fetch_tickers = [target.get("fetch_ticker") for target in ledger.get("targets", []) if target.get("fetch_ticker")]
fetch_tickers = tuple(sorted(set(holding_fetch_tickers + target_fetch_tickers)))
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
priced_holdings = add_weight_column(build_price_map(holdings, fetched_prices, usdkrw))
try:
    symbol_names = fetch_symbol_names(fetch_tickers)
except Exception:
    symbol_names = {}
if not priced_holdings.empty:
    priced_holdings["name"] = priced_holdings.apply(
        lambda row: symbol_names.get(row.get("fetch_ticker")) or row.get("ticker"),
        axis=1,
    )
try:
    dividend_histories = fetch_dividend_histories(fetch_tickers)
except Exception:
    dividend_histories = {}
monthly = estimate_monthly_dividends(priced_holdings, dividend_histories, usdkrw)
value_col = "gross_krw" if ledger["settings"].get("display_basis") == "gross" else "net_krw"
basis_text = "세전" if value_col == "gross_krw" else "세후"
primary_goal = ledger["targets"][0] if ledger.get("targets") else None
goal_achievement = compute_goal_achievement(primary_goal, priced_holdings, usdkrw, fetched_prices)

if price_error:
    st.warning(f"현재가 조회 일부 실패: 마지막 거래 단가를 fallback으로 사용합니다. ({price_error})")
if fx_error or usdkrw is None:
    st.info("환율 조회 실패 시 임의 고정 환율을 쓰지 않고, 저장된 거래 환율이 있는 USD 종목만 원화 환산합니다.")
else:
    st.caption("환율은 원화와 달러 평가금액 환산에 사용됩니다.")

st.markdown("### 📌 요약")
m1, m2, m3, m4 = st.columns(4)
convertible_values = pd.to_numeric(priced_holdings.get("current_value_krw", pd.Series(dtype="float64")), errors="coerce").dropna() if not priced_holdings.empty else pd.Series(dtype="float64")
monthly_total = to_float(monthly[value_col].sum(), 0.0) if not monthly.empty else 0.0
target_value, target_sub = build_target_summary(goal_achievement, priced_holdings)
portfolio_usd_values = pd.to_numeric(priced_holdings.get("current_value_usd", pd.Series(dtype="float64")), errors="coerce").dropna() if not priced_holdings.empty else pd.Series(dtype="float64")
valuation_sub = f"달러 기준 {fmt_usd_whole(portfolio_usd_values.sum())}" if usdkrw is not None else "현재 USD 환율 조회 불가"
with m1:
    metric_card("평가금액(환산 가능분)", fmt_won(convertible_values.sum()), valuation_sub, True)
with m2:
    metric_card(f"연간 예상 배당({basis_text})", fmt_won(monthly_total), "최근 배당 이력 기반 추정")
with m3:
    metric_card(f"월평균 예상 배당({basis_text})", fmt_won(monthly_total / 12 if monthly_total else 0), "월별 예상 배당금 기준")
with m4:
    metric_card("목표 달성률", target_value, target_sub)

st.markdown("### 📈 월별 예상 배당금")
chart_df = monthly.copy()
chart_df["month"] = chart_df["month"].astype(str)
chart_df["예상 배당"] = chart_df[value_col]
fig = px.bar(chart_df, x="month", y="예상 배당", text="예상 배당")
fig.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside",
    marker_color="#3182F6",
    hovertemplate="%{x}월<br>예상 배당: %{y:,.0f}원<extra></extra>",
)
fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="원", xaxis_title="월")
fig.update_xaxes(type="category", categoryorder="array", categoryarray=[str(i) for i in range(1, 13)], tickmode="array", tickvals=[str(i) for i in range(1, 13)])
st.plotly_chart(fig, use_container_width=True)
st.caption("배당 이력이 제공되는 종목의 과거 월별 배당을 보유 수량에 적용한 추정치입니다. 실제 지급액과 지급월은 달라질 수 있습니다.")

st.markdown("### 🎯 목표 달성도")
if not ledger.get("targets"):
    st.caption("저장된 목표 수량이 없습니다. 예: SCHD 목표 3300주")
elif not goal_achievement.get("ok"):
    st.warning(goal_achievement.get("error", "목표 달성도 계산 불가"))
else:
    pct = to_float(goal_achievement.get("achievement_pct"))
    target_line = f"{goal_achievement['target_symbol']} {fmt_goal_quantity(goal_achievement['target_quantity'])}주"
    st.progress(
        min(pct / 100.0, 1.0),
        text=(
            f"{target_line}"
            f" · 목표금액 {fmt_usd_whole(goal_achievement.get('target_amount_usd'))}"
            f" · 현재 달성금액 {fmt_usd_whole(goal_achievement.get('portfolio_amount_usd'))}"
            f" · 남은 수량 {fmt_goal_quantity(goal_achievement.get('remaining_actual_quantity'))}주"
            f"(환산시 {fmt_goal_quantity(goal_achievement.get('remaining_equivalent_quantity'))}주)"
            f" · 남은 금액 {fmt_usd_whole(goal_achievement.get('remaining_amount_usd'))}"
            f" · 달성률 {pct:.1f}%"
        ),
    )

st.markdown("### ✍️ 거래 입력")
with st.form("dividend_ledger_add_transaction", clear_on_submit=True):
    c1, c2, c3, c4, c5 = st.columns([1.3, 1.3, 1, 1, 0.9])
    asset_class = c1.selectbox("자산 구분", ASSET_CLASS_OPTIONS, format_func=lambda x: ASSET_CLASS_LABELS[x])
    ticker = c2.text_input("티커", value="SCHD", help="US 예: SCHD, TQQQ, MSFT / 국내 예: 069500, 458730 / 코인 예: BTC")
    quantity = c3.number_input("수량", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    price = c4.number_input("거래 단가", min_value=0.0, value=0.0, step=0.01, format="%.2f", help="US는 달러 단가, 국내/코인은 원화 단가입니다.")

    submitted = c5.form_submit_button("➕ 거래 추가", type="primary", use_container_width=True)
    if submitted:
        info = normalize_ticker(ticker, asset_class)
        item = {
            "id": uuid4().hex,
            "date": datetime.now(KST).date().isoformat(),
            "asset_class": info.asset_class,
            "ticker": info.display_ticker,
            "side": "BUY",
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
    display_holdings["수량"] = display_holdings["quantity"].apply(fmt_quantity)
    display_holdings["평균단가"] = [fmt_money(v, c) for v, c in zip(display_holdings["avg_cost"], display_holdings["currency"])]
    display_holdings["현재가"] = [fmt_money(v, c) for v, c in zip(display_holdings["current_price"], display_holdings["currency"])]
    display_holdings["비중"] = display_holdings["weight_pct"].apply(fmt_pct)
    display_holdings["평가금액(KRW)"] = display_holdings["current_value_krw"].apply(lambda x: fmt_won(x) if pd.notna(x) else "환율 필요")
    display_holdings["평가금액(USD)"] = display_holdings["current_value_usd"].apply(lambda x: fmt_usd_whole(x) if pd.notna(x) else "현재 USD 환율 조회 불가")
    st.dataframe(
        display_holdings[["자산", "ticker", "name", "수량", "평균단가", "현재가", "비중", "평가금액(KRW)", "평가금액(USD)"]],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### 💼 보유 종목 관리")
if ledger["transactions"]:
    tx_df = pd.DataFrame(ledger["transactions"])
    tx_df["delete"] = False
    weight_by_ticker = {}
    if priced_holdings is not None and not priced_holdings.empty:
        weight_by_ticker = {
            (row.get("asset_class"), row.get("ticker")): fmt_pct(row.get("weight_pct"))
            for row in priced_holdings.to_dict("records")
        }
    tx_df["weight"] = tx_df.apply(lambda row: weight_by_ticker.get((row.get("asset_class"), row.get("ticker")), "0.0%"), axis=1)
    editor_columns = ["delete", "date", "asset_class", "ticker", "quantity", "price", "weight", "id", "side"]
    edited = st.data_editor(
        tx_df[editor_columns],
        hide_index=True,
        use_container_width=True,
        column_config={
            "delete": st.column_config.CheckboxColumn("삭제"),
            "date": st.column_config.DateColumn("등록일", disabled=True),
            "asset_class": st.column_config.SelectboxColumn("자산 구분", options=ASSET_CLASS_OPTIONS),
            "ticker": st.column_config.TextColumn("티커"),
            "quantity": st.column_config.NumberColumn("수량", format="%.2f"),
            "price": st.column_config.NumberColumn("평단가", format="%.2f"),
            "weight": st.column_config.TextColumn("비중", disabled=True),
            "id": None,
            "side": None,
        },
        column_order=["delete", "date", "asset_class", "ticker", "quantity", "price", "weight"],
    )
    c_save, c_delete = st.columns(2)
    original_by_id = {str(row.get("id")): row for row in ledger["transactions"]}
    if c_save.button("💾 보유 종목 수정 저장", use_container_width=True):
        rows = []
        for row in edited.drop(columns=["delete", "weight"], errors="ignore").to_dict("records"):
            original = original_by_id.get(str(row.get("id")), {})
            rows.append({**original, **row})
        ledger["transactions"] = normalize_transactions(rows)
        save_ledger(ledger)
        st.toast("보유 종목을 저장했습니다.", icon="✅")
        st.rerun()
    if c_delete.button("🗑️ 선택 종목 삭제", use_container_width=True):
        keep_ids = set(edited.loc[edited["delete"] != True, "id"].astype(str).tolist())  # noqa: E712
        ledger["transactions"] = normalize_transactions([row for row in ledger["transactions"] if str(row.get("id")) in keep_ids])
        save_ledger(ledger)
        st.toast("선택한 보유 종목을 삭제했습니다.", icon="✅")
        st.rerun()
else:
    st.caption("보유 종목을 여기에서 수정/삭제할 수 있습니다.")

last_sync = ledger.get("_last_sync")
if last_sync:
    st.caption(f"마지막 저장: {last_sync}")
