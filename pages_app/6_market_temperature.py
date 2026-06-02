from io import StringIO
from html import escape
from textwrap import dedent

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests

from ui.styles import TOSS_CSS
from logic.market import (
    compute_rsi,
    compute_drawdown_series,
)

# ──────────────────────────────────────────────
# 1. 디자인
# ──────────────────────────────────────────────
st.markdown(TOSS_CSS, unsafe_allow_html=True)

# 시장온도 기준 종목 및 색상
WATCHLIST = ["QQQ", "SCHD", "SPY"]
TICKER_COLORS = {
    "QQQ": "#3182F6",   # 파랑
    "SCHD": "#00875A",  # 초록
    "SPY": "#FF8B00",   # 주황
}

# 표시 기간 옵션 (라벨 → 거래일 수, None 이면 전체)
RANGE_OPTIONS = {
    "6개월": 126,
    "1년": 252,
    "3년": 756,
    "5년": 1260,
    "전체": None,
}

CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CNN_FEAR_GREED_PAGE_URL = "https://edition.cnn.com/markets/fear-and-greed"

MARKET_BRIEFING_TICKERS = {
    "S&P 500": {"ticker": "^GSPC", "kind": "index"},
    "DOW JONES": {"ticker": "^DJI", "kind": "index"},
    "NASDAQ": {"ticker": "^IXIC", "kind": "index"},
    "USD/KRW": {"ticker": "KRW=X", "kind": "krw"},
    "WTI": {"ticker": "CL=F", "kind": "usd_2"},
    "GOLD": {"ticker": "GC=F", "kind": "usd_0"},
    "VIX": {"ticker": "^VIX", "kind": "vix"},
}

FEAR_GREED_RATING_LABELS = {
    "extreme fear": "극단적 공포",
    "fear": "공포",
    "neutral": "중립",
    "greed": "탐욕",
    "extreme greed": "극단적 탐욕",
}

FEAR_GREED_RATING_COLORS = {
    "extreme fear": "#ef4444",
    "fear": "#f97316",
    "neutral": "#eab308",
    "greed": "#22c55e",
    "extreme greed": "#10b981",
}


def render_html_block(html: str) -> None:
    cleaned = dedent(html).strip()
    cleaned = "\n".join(line.lstrip() for line in cleaned.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2. 데이터 헬퍼 (4_conversion_analysis.py 패턴 복제)
#    - 기존 페이지를 import 하면 스크립트가 실행되므로, 안전하게 복제하여 사용
# ──────────────────────────────────────────────
def _normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("조회 결과가 비어 있습니다.")

    normalized = df.copy()

    if isinstance(normalized.columns, pd.MultiIndex):
        if len(normalized.columns.levels) >= 2:
            normalized.columns = normalized.columns.get_level_values(0)
        else:
            normalized.columns = [
                col[0] if isinstance(col, tuple) else col for col in normalized.columns
            ]

    if "Close" not in normalized.columns:
        raise ValueError(f"Close 컬럼이 없습니다. 사용 가능 컬럼: {list(normalized.columns)}")

    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized[~normalized.index.isna()]
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.sort_index()

    if normalized.empty:
        raise ValueError("정규화 후 데이터가 비어 있습니다.")

    return normalized


def _fetch_stooq_history(ticker: str) -> pd.DataFrame:
    stooq_symbol = f"{ticker.lower()}.us"
    stooq_url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    response = requests.get(stooq_url, timeout=20)
    response.raise_for_status()
    raw = response.text.strip()
    if not raw:
        raise ValueError("Stooq 응답이 비어 있습니다.")
    df = pd.read_csv(StringIO(raw))
    if df.empty:
        raise ValueError("Stooq CSV가 비어 있습니다.")
    if "Date" not in df.columns:
        raise ValueError(f"Stooq CSV에 Date 컬럼이 없습니다. 컬럼: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    return _normalize_history_frame(df)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_close_series(ticker: str) -> pd.Series:
    """티커의 전체 종가(Close) Series 를 반환한다 (yfinance → download → Stooq 폴백)."""
    if not ticker:
        raise ValueError("티커가 비어 있습니다.")

    errors = []

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", auto_adjust=False, actions=False, raise_errors=True)
        normalized = _normalize_history_frame(df)
        return normalized["Close"].astype("float64")
    except Exception as e:
        errors.append(f"yfinance.Ticker.history 실패: {e}")

    try:
        df = yf.download(
            ticker,
            period="max",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            timeout=20,
        )
        normalized = _normalize_history_frame(df)
        return normalized["Close"].astype("float64")
    except Exception as e:
        errors.append(f"yfinance.download 실패: {e}")

    try:
        normalized = _fetch_stooq_history(ticker)
        return normalized["Close"].astype("float64")
    except Exception as e:
        errors.append(f"Stooq CSV 실패: {e}")

    raise ValueError(f"{ticker} 종가 조회 실패 | " + " | ".join(errors))


def load_watchlist(tickers):
    """워치리스트 종가를 모아 (성공 dict, 실패 dict) 로 반환한다.

    한 종목이 실패해도 나머지는 정상 표시되도록 종목별로 격리한다.
    """
    closes = {}
    failures = {}
    for ticker in tickers:
        try:
            series = fetch_close_series(ticker)
            if series is None or series.empty:
                failures[ticker] = "데이터가 비어 있습니다."
                continue
            closes[ticker] = series
        except Exception as e:  # noqa: BLE001 - 페이지 전체 중단 방지
            failures[ticker] = str(e)
    return closes, failures


def _slice_recent(series: pd.Series, lookback) -> pd.Series:
    if lookback is None:
        return series
    if series is None or series.empty:
        return series
    return series.tail(int(lookback))


def _last_valid(series: pd.Series):
    if series is None:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def _safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_fng_timestamp(value):
    if value is None:
        return None
    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="coerce")
    else:
        numeric = _safe_float(value)
        if numeric is None:
            return None
        unit = "ms" if numeric > 10_000_000_000 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, errors="coerce")
    if pd.isna(parsed):
        return None
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(None)
    return parsed


def _find_nested_value(payload, keys):
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_nested_value(value, keys)
            if found is not None:
                return found
    return None


def _extract_fng_current(payload: dict) -> dict:
    current = payload.get("fear_and_greed") if isinstance(payload, dict) else None
    if not isinstance(current, dict):
        current = payload if isinstance(payload, dict) else {}

    score = _safe_float(_find_nested_value(current, ("score", "value")))
    rating = _find_nested_value(current, ("rating", "classification", "status"))
    timestamp = _find_nested_value(current, ("timestamp", "date", "asOf"))

    if score is None and isinstance(payload, dict):
        score = _safe_float(_find_nested_value(payload, ("score", "value")))
    if rating is None and isinstance(payload, dict):
        rating = _find_nested_value(payload, ("rating", "classification", "status"))
    if timestamp is None and isinstance(payload, dict):
        timestamp = _find_nested_value(payload, ("timestamp", "date", "asOf"))

    normalized_rating = str(rating).replace("_", " ").strip().lower() if rating else None
    return {
        "score": score,
        "rating": normalized_rating,
        "timestamp": _parse_fng_timestamp(timestamp),
    }


def _coerce_fng_history_item(item):
    if not isinstance(item, dict):
        return None
    score = _safe_float(_find_nested_value(item, ("score", "value", "y")))
    date_value = _find_nested_value(item, ("date", "timestamp", "x", "asOf"))
    rating = _find_nested_value(item, ("rating", "classification", "status"))
    parsed_date = _parse_fng_timestamp(date_value)
    if score is None or parsed_date is None:
        return None
    return {
        "date": parsed_date.normalize(),
        "score": score,
        "rating": str(rating).replace("_", " ").strip().lower() if rating else None,
    }


def _collect_fng_history_candidates(node):
    candidates = []
    if isinstance(node, list):
        rows = [_coerce_fng_history_item(item) for item in node]
        rows = [row for row in rows if row is not None]
        if len(rows) >= 2:
            candidates.append(rows)
        for item in node:
            candidates.extend(_collect_fng_history_candidates(item))
    elif isinstance(node, dict):
        for key, value in node.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in ("histor", "graph", "data")):
                candidates.extend(_collect_fng_history_candidates(value))
            elif isinstance(value, (dict, list)):
                candidates.extend(_collect_fng_history_candidates(value))
    return candidates


@st.cache_data(ttl=60 * 45, show_spinner=False)
def fetch_fear_and_greed_data() -> dict:
    """CNN Fear & Greed 현재값과 히스토리를 방어적으로 조회한다."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.6099.210 Mobile Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(CNN_FEAR_GREED_URL, headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()

    current = _extract_fng_current(payload)
    history_candidates = _collect_fng_history_candidates(payload)
    history_df = pd.DataFrame(columns=["date", "score", "rating"])
    if history_candidates:
        rows = max(history_candidates, key=len)
        history_df = pd.DataFrame(rows).dropna(subset=["date", "score"])
        if not history_df.empty:
            history_df = (
                history_df.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .tail(365)
                .reset_index(drop=True)
            )
    return {"current": current, "history": history_df, "error": None}


def get_fear_and_greed_data() -> dict:
    try:
        return fetch_fear_and_greed_data()
    except Exception as e:  # noqa: BLE001 - 외부 API 실패가 페이지 전체에 전파되지 않도록 격리
        return {
            "current": {"score": None, "rating": None, "timestamp": None},
            "history": pd.DataFrame(columns=["date", "score", "rating"]),
            "error": str(e),
        }


def _snapshot_from_close(close: pd.Series) -> dict:
    close = close.dropna().astype("float64")
    if close.empty:
        raise ValueError("종가 데이터가 비어 있습니다.")
    latest = float(close.iloc[-1])
    previous = float(close.iloc[-2]) if len(close) >= 2 else None
    change = latest - previous if previous not in (None, 0) else None
    change_pct = (change / previous) * 100 if change is not None and previous else None
    return {"value": latest, "previous": previous, "change": change, "change_pct": change_pct}


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_market_snapshot(ticker: str) -> dict:
    """최근 종가 기준 현재값/전일대비를 계산한다. 실패는 호출부에서 카드 단위로 처리한다."""
    df = yf.download(
        ticker,
        period="10d",
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=8,
    )
    normalized = _normalize_history_frame(df)
    return _snapshot_from_close(normalized["Close"])


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_market_snapshot_batch(ticker_map: dict) -> dict:
    tickers = [meta["ticker"] for meta in ticker_map.values()]
    df = yf.download(
        tickers,
        period="10d",
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
        group_by="ticker",
        timeout=10,
    )
    if df is None or df.empty:
        raise ValueError("시장 브리핑 데이터가 비어 있습니다.")

    snapshots = {}
    for label, meta in ticker_map.items():
        ticker = meta["ticker"]
        try:
            if isinstance(df.columns, pd.MultiIndex) and ticker in df.columns.get_level_values(0):
                ticker_df = df[ticker]
            elif isinstance(df.columns, pd.MultiIndex) and ticker in df.columns.get_level_values(-1):
                ticker_df = df.xs(ticker, axis=1, level=-1)
            else:
                ticker_df = df
            normalized = _normalize_history_frame(ticker_df)
            snapshots[label] = {**_snapshot_from_close(normalized["Close"]), "error": None}
        except Exception as e:  # noqa: BLE001 - 배치 결과 중 일부 누락을 카드 단위로 격리
            snapshots[label] = {"value": None, "previous": None, "change": None, "change_pct": None, "error": str(e)}
    return snapshots


def load_market_briefing_snapshots() -> dict:
    try:
        snapshots = fetch_market_snapshot_batch(MARKET_BRIEFING_TICKERS)
    except Exception:  # noqa: BLE001 - 배치 조회 전체 실패 시 종목별 폴백
        snapshots = {}

    for label, meta in MARKET_BRIEFING_TICKERS.items():
        if label in snapshots and not snapshots[label].get("error"):
            continue
        try:
            snapshots[label] = {**fetch_market_snapshot(meta["ticker"]), "error": None}
        except Exception as e:  # noqa: BLE001 - 한 카드 실패가 전체 렌더링을 막지 않음
            snapshots[label] = {"value": None, "previous": None, "change": None, "change_pct": None, "error": str(e)}
    return snapshots


def _format_snapshot_value(label: str, value):
    value = _safe_float(value)
    if value is None:
        return "데이터 없음"
    kind = MARKET_BRIEFING_TICKERS[label]["kind"]
    if kind == "krw":
        return f"{value:,.0f}"
    if kind == "usd_2":
        return f"${value:,.2f}"
    if kind == "usd_0":
        return f"${value:,.0f}"
    if kind == "vix":
        return f"{value:.2f}"
    return f"{value:,.0f}"


def _format_change(change, change_pct, suffix="p"):
    change = _safe_float(change)
    change_pct = _safe_float(change_pct)
    if change is None or change_pct is None:
        return ""
    arrow = "▲" if change >= 0 else "▼"
    return f"{arrow} {abs(change):,.2f}{suffix} &nbsp; {change_pct:+.2f}%"


def _change_class(change):
    change = _safe_float(change)
    if change is None:
        return "neutral"
    return "up" if change >= 0 else "down"


def _fng_rating_label(rating):
    if not rating:
        return "데이터 없음"
    return FEAR_GREED_RATING_LABELS.get(str(rating).lower(), str(rating))


def _fng_rating_color(rating):
    return FEAR_GREED_RATING_COLORS.get(str(rating).lower(), "#64748b")


def _render_fng_svg(history_df: pd.DataFrame) -> str:
    if history_df is None or history_df.empty:
        return "<div class='gorani-market-temp-chart-empty'>히스토리 데이터 없음</div>"

    chart = history_df.dropna(subset=["date", "score"]).tail(220).copy()
    if chart.empty:
        return "<div class='gorani-market-temp-chart-empty'>히스토리 데이터 없음</div>"

    width, height = 560, 210
    pad_l, pad_r, pad_t, pad_b = 34, 12, 12, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    scores = chart["score"].clip(0, 100).tolist()
    dates = pd.to_datetime(chart["date"]).tolist()
    n = len(scores)

    def x_pos(i):
        return pad_l + (plot_w * i / max(n - 1, 1))

    def y_pos(score):
        return pad_t + plot_h - (plot_h * score / 100)

    points = [(x_pos(i), y_pos(score)) for i, score in enumerate(scores)]
    line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area_points = f"{pad_l},{pad_t + plot_h} " + line_points + f" {pad_l + plot_w},{pad_t + plot_h}"

    tick_indexes = sorted(set([0, n // 4, n // 2, (n * 3) // 4, n - 1]))
    tick_labels = []
    for i in tick_indexes:
        label = dates[i].strftime("%y.%m")

        if i == 0:
            anchor = "start"
        elif i == n - 1:
            anchor = "end"
        else:
            anchor = "middle"

        tick_labels.append(
            f"<text x='{x_pos(i):.1f}' y='{height - 7}' text-anchor='{anchor}' "
            "class='gorani-market-temp-svg-label'>" + escape(label) + "</text>"
        )

    grid_lines = []
    for level in (0, 25, 50, 75, 100):
        y = y_pos(level)
        dash = "4 4" if level in (25, 75) else ""
        grid_lines.append(
            f"<line x1='{pad_l}' x2='{pad_l + plot_w}' y1='{y:.1f}' y2='{y:.1f}' "
            f"stroke='#e5e7eb' stroke-width='1' stroke-dasharray='{dash}' />"
        )
        grid_lines.append(
            f"<text x='{pad_l - 8}' y='{y + 4:.1f}' text-anchor='end' "
            "class='gorani-market-temp-svg-label'>" + str(level) + "</text>"
        )

    hover_points = []
    for i, (_, row) in enumerate(chart.iterrows()):
        x, y = points[i]
        score = scores[i]
        date = dates[i]
        tooltip = f"{date.month}/{date.day}\n{score:.1f} · {_fng_rating_label(row.get('rating'))}"
        hover_points.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='10' fill='transparent' "
            "stroke='transparent' pointer-events='all'>"
            f"<title>{escape(tooltip)}</title></circle>"
        )

    return f"""
    <svg class="gorani-market-temp-fng-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Fear and Greed history chart">
      <defs>
        <linearGradient id="goraniMarketTempFngFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#22c55e" stop-opacity="0.24" />
          <stop offset="100%" stop-color="#22c55e" stop-opacity="0.02" />
        </linearGradient>
      </defs>
      {''.join(grid_lines)}
      <polygon points="{area_points}" fill="url(#goraniMarketTempFngFill)" />
      <polyline points="{line_points}" fill="none" stroke="#22c55e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
      {''.join(hover_points)}
      {''.join(tick_labels)}
    </svg>
    """


def render_market_temperature_css():
    st.markdown(
        """
        <style>
          .gorani-market-temp-title-wrap { margin: 18px 0 24px; }
          .gorani-market-temp-title-wrap h1 { margin: 0 0 10px; font-size: 2.35rem; line-height: 1.15; letter-spacing: -0.04em; color: #191f28; }
          .gorani-market-temp-subtitle { color: #6b7280; font-size: 0.98rem; margin: 0; }
          .gorani-market-temp-top-grid { display: grid; grid-template-columns: minmax(360px, 1.8fr) minmax(220px, 1fr) minmax(180px, 0.82fr); gap: 18px; align-items: stretch; margin: 0 0 30px; }
          .gorani-market-temp-index-stack, .gorani-market-temp-macro-stack { display: grid; gap: 14px; }
          .gorani-market-temp-index-stack { grid-template-rows: repeat(3, minmax(118px, 1fr)); }
          .gorani-market-temp-macro-stack { grid-template-rows: repeat(4, minmax(86px, 1fr)); }
          .gorani-market-temp-card { background: #fff; border: 1px solid #e5e8ef; border-radius: 18px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); box-sizing: border-box; overflow: hidden; }
          .gorani-market-temp-fng-card { min-height: 500px; padding: 24px 24px 18px; border-color: #eadff8; background: linear-gradient(180deg, #ffffff 0%, #fff 74%, #faf7ff 100%); }
          .gorani-market-temp-fng-head, .gorani-market-temp-card-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
          .gorani-market-temp-fng-title { font-size: 1.18rem; font-weight: 800; color: #374151; }
          .gorani-market-temp-fng-source { font-size: 0.78rem; color: #9ca3af !important; text-decoration: none; }
          .gorani-market-temp-fng-score-row { display: flex; align-items: baseline; gap: 12px; margin: 20px 0 16px; }
          .gorani-market-temp-fng-score { font-size: 2.65rem; font-weight: 900; letter-spacing: -0.05em; }
          .gorani-market-temp-fng-rating { font-size: 1.06rem; font-weight: 800; }
          .gorani-market-temp-fng-updated { color: #9ca3af; font-size: 0.74rem; margin-top: -8px; margin-bottom: 10px; }
          .gorani-market-temp-gradient-bar { height: 15px; border-radius: 999px; background: linear-gradient(90deg, #ef4444 0%, #f97316 25%, #eab308 50%, #84cc16 70%, #10b981 100%); position: relative; overflow: hidden; }
          .gorani-market-temp-gradient-bar::after { content: ""; position: absolute; inset: 0; background: repeating-linear-gradient(90deg, transparent 0, transparent calc(20% - 2px), rgba(255,255,255,0.9) calc(20% - 2px), rgba(255,255,255,0.9) 20%); }
          .gorani-market-temp-gradient-marker { position: absolute; top: -3px; width: 4px; height: 21px; border-radius: 999px; background: #111827; box-shadow: 0 0 0 2px #fff; z-index: 2; transform: translateX(-2px); }
          .gorani-market-temp-gradient-labels { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin: 12px 0 12px; color: #9ca3af; font-size: 0.72rem; text-align: center; }
          .gorani-market-temp-fng-svg { width: 100%; height: auto; display: block; margin-top: 6px; }
          .gorani-market-temp-svg-label { fill: #8b95a1; font-size: 12px; font-family: sans-serif; }
          .gorani-market-temp-chart-empty { height: 210px; display: flex; align-items: center; justify-content: center; color: #9ca3af; border: 1px dashed #e5e7eb; border-radius: 14px; margin-top: 14px; background: #fbfdff; }
          .gorani-market-temp-small-card { padding: 18px 20px; position: relative; display: flex; flex-direction: column; justify-content: center; }
          .gorani-market-temp-index-card::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: #2f6feb; }
          .gorani-market-temp-label { color: #94a3b8; font-size: 0.82rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }
          .gorani-market-temp-value { color: #111827; font-size: 2rem; line-height: 1.05; font-weight: 900; letter-spacing: -0.05em; margin-top: 8px; }
          .gorani-market-temp-change { margin-top: 10px; font-size: 0.88rem; font-weight: 800; }
          .gorani-market-temp-change.up { color: #dc2626; }
          .gorani-market-temp-change.down { color: #2563eb; }
          .gorani-market-temp-change.neutral { color: #94a3b8; }
          .gorani-market-temp-badge { display: inline-block; margin-top: 8px; width: fit-content; padding: 3px 8px; border-radius: 8px; background: #fff1f2; color: #e11d48; font-size: 0.74rem; font-weight: 800; }
          .gorani-market-temp-macro-card { text-align: center; align-items: center; }
          .gorani-market-temp-macro-card.krw { background: #f3f7fc; border-color: #dde7f4; }
          .gorani-market-temp-macro-card.risk { background: #fff1f2; border-color: #fecdd3; }
          .gorani-market-temp-macro-card.risk .gorani-market-temp-value { color: #dc2626; }
          .gorani-market-temp-error { color: #ef4444; font-size: 0.85rem; font-weight: 700; margin-top: 8px; }
          .gorani-market-temp-section-gap { height: 10px; }
          @media (max-width: 900px) {
            .gorani-market-temp-top-grid { grid-template-columns: 1fr; }
            .gorani-market-temp-fng-card { min-height: auto; }
            .gorani-market-temp-index-stack, .gorani-market-temp-macro-stack { grid-template-rows: none; }
          }
          @media (min-width: 901px) and (max-width: 1150px) {
            .gorani-market-temp-top-grid { grid-template-columns: minmax(340px, 1.4fr) minmax(190px, 0.9fr); }
            .gorani-market-temp-macro-stack { grid-column: 1 / -1; grid-template-columns: repeat(4, minmax(0, 1fr)); }
          }
          @media (max-width: 560px) {
            .gorani-market-temp-title-wrap h1 { font-size: 2rem; }
            .gorani-market-temp-fng-card, .gorani-market-temp-small-card { padding: 18px; }
            .gorani-market-temp-value { font-size: 1.7rem; }
            .gorani-market-temp-gradient-labels { font-size: 0.62rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_title_area():
    st.markdown(
        """
        <div class="gorani-market-temp-title-wrap">
          <h1>🌡️ 시장온도</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_briefing(fng_data: dict, snapshots: dict):
    current = fng_data.get("current", {}) if isinstance(fng_data, dict) else {}
    history = fng_data.get("history", pd.DataFrame()) if isinstance(fng_data, dict) else pd.DataFrame()
    score = _safe_float(current.get("score"))
    rating = current.get("rating")
    score_text = "--" if score is None else f"{round(score):.0f}"
    marker_left = 50 if score is None else max(0, min(100, score))
    rating_text = _fng_rating_label(rating)
    rating_color = _fng_rating_color(rating)
    updated_at = current.get("timestamp")
    updated_text = ""
    if updated_at is not None:
        updated_text = f"<div class='gorani-market-temp-fng-updated'>기준: {escape(pd.Timestamp(updated_at).strftime('%Y-%m-%d %H:%M'))}</div>"
    if fng_data.get("error"):
        updated_text = "<div class='gorani-market-temp-error'>조회 실패 · 잠시 후 다시 시도해주세요.</div>"

    def index_card(label):
        snap = snapshots.get(label, {})
        value = _format_snapshot_value(label, snap.get("value"))
        change_class = _change_class(snap.get("change"))
        change = _format_change(snap.get("change"), snap.get("change_pct"))
        error = "<div class='gorani-market-temp-error'>조회 실패</div>" if snap.get("error") else ""
        badge = ""
        if _safe_float(snap.get("change_pct")) is not None and _safe_float(snap.get("change_pct")) <= -2:
            badge = "<span class='gorani-market-temp-badge'>급락</span>"
        return dedent(f"""
          <div class="gorani-market-temp-card gorani-market-temp-small-card gorani-market-temp-index-card">
            <div class="gorani-market-temp-label">{escape(label)}</div>
            <div class="gorani-market-temp-value">{escape(value)}</div>
            <div class="gorani-market-temp-change {change_class}">{change}</div>
            {badge}{error}
          </div>
        """).strip()

    def macro_card(label):
        snap = snapshots.get(label, {})
        value = _format_snapshot_value(label, snap.get("value"))
        macro_class = "krw" if label == "USD/KRW" else "risk"
        error = "<div class='gorani-market-temp-error'>조회 실패</div>" if snap.get("error") else ""
        return dedent(f"""
          <div class="gorani-market-temp-card gorani-market-temp-small-card gorani-market-temp-macro-card {macro_class}">
            <div class="gorani-market-temp-label">{escape(label)}</div>
            <div class="gorani-market-temp-value">{escape(value)}</div>
            {error}
          </div>
        """).strip()

    fng_html = dedent(f"""
      <div class="gorani-market-temp-card gorani-market-temp-fng-card">
        <div class="gorani-market-temp-fng-head">
          <div class="gorani-market-temp-fng-title">공포 &amp; 탐욕 지수</div>
          <a class="gorani-market-temp-fng-source" href="{CNN_FEAR_GREED_PAGE_URL}" target="_blank" rel="noopener noreferrer">CNN</a>
        </div>
        <div class="gorani-market-temp-fng-score-row">
          <div class="gorani-market-temp-fng-score" style="color:{rating_color};">{score_text}</div>
          <div class="gorani-market-temp-fng-rating" style="color:{rating_color};">{escape(rating_text)}</div>
        </div>
        {updated_text}
        <div class="gorani-market-temp-gradient-bar"><div class="gorani-market-temp-gradient-marker" style="left:{marker_left:.1f}%;"></div></div>
        <div class="gorani-market-temp-gradient-labels"><span>극단적 공포</span><span>공포</span><span>중립</span><span>탐욕</span><span>극단적 탐욕</span></div>
        {_render_fng_svg(history)}
      </div>
    """).strip()

    html = dedent(f"""
    <div class="gorani-market-temp-top-grid">
      {fng_html}
      <div class="gorani-market-temp-index-stack">
        {index_card('S&P 500')}
        {index_card('DOW JONES')}
        {index_card('NASDAQ')}
      </div>
      <div class="gorani-market-temp-macro-stack">
        {macro_card('USD/KRW')}
        {macro_card('WTI')}
        {macro_card('GOLD')}
        {macro_card('VIX')}
      </div>
    </div>
    """).strip()
    render_html_block(html)


def render_tradingview_heatmap():
    st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)
    st.markdown("### 🇺🇸 미국주식 섹터 트리맵")
    st.caption("S&P 500 구성종목의 섹터별 흐름을 TradingView 히트맵으로 확인합니다.")
    st.caption("위젯이 표시되지 않으면 브라우저의 외부 스크립트 차단 설정을 확인해주세요.")
    tradingview_heatmap_html = dedent("""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          html,
          body,
          .tradingview-widget-container,
          .tradingview-widget-container__widget,
          .tradingview-widget-container iframe {
            width: 100%;
            height: 700px;
            margin: 0;
            padding: 0;
          }
          html, body {
            overflow: hidden;
          }
          .tradingview-widget-container {
            box-sizing: border-box;
          }
        </style>
      </head>
      <body>
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
          {
            "exchanges": [],
            "dataSource": "SPX500",
            "grouping": "sector",
            "blockSize": "market_cap_basic",
            "blockColor": "change",
            "locale": "kr",
            "symbolUrl": "",
            "colorTheme": "light",
            "hasTopBar": true,
            "isDataSetEnabled": true,
            "isZoomEnabled": true,
            "hasSymbolTooltip": true,
            "isMonoSize": false,
            "width": "100%",
            "height": 700
          }
          </script>
        </div>
      </body>
    </html>
    """).strip()
    try:
        components.html(tradingview_heatmap_html, height=700, scrolling=False)
    except Exception:  # noqa: BLE001 - 위젯 렌더링 실패 시에도 기존 콘텐츠 유지
        st.info("TradingView 히트맵 위젯을 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    st.markdown(
        "<a href='https://www.tradingview.com/widget/stock-heatmap/' "
        "target='_blank' rel='noopener noreferrer' "
        "style='color:#8b95a1;font-size:12px;text-decoration:none;'>"
        "TradingView Stock Heatmap 공식 위젯 열기</a>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 3. 차트
# ──────────────────────────────────────────────
def build_rsi_chart(rsi_map: dict) -> go.Figure:
    fig = go.Figure()
    for ticker, rsi_series in rsi_map.items():
        if rsi_series is None or rsi_series.dropna().empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rsi_series.index,
                y=rsi_series.values,
                mode="lines",
                name=ticker,
                line=dict(color=TICKER_COLORS.get(ticker, "#3182F6"), width=2.0),
            )
        )

    # 과매수(70) / 과매도(30) 기준선
    fig.add_hline(y=70, line=dict(color="#D93D44", width=1.2, dash="dash"),
                  annotation_text="과매수 70", annotation_position="top left")
    fig.add_hline(y=30, line=dict(color="#1B64DA", width=1.2, dash="dash"),
                  annotation_text="과매도 30", annotation_position="bottom left")

    fig.update_layout(
        title=dict(text="RSI 14 추이", font=dict(size=18, color="#191F28")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(range=[0, 100], hoverformat=".1f"),
    )
    return fig


def build_drawdown_chart(dd_map: dict) -> go.Figure:
    fig = go.Figure()
    for ticker, dd_series in dd_map.items():
        if dd_series is None or dd_series.dropna().empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=dd_series.index,
                y=dd_series.values,
                mode="lines",
                name=ticker,
                line=dict(color=TICKER_COLORS.get(ticker, "#3182F6"), width=2.0),
            )
        )

    # -10% / -20% / -30% / -40% 기준선 (값은 비율)
    for level in (-0.10, -0.20, -0.30, -0.40):
        fig.add_hline(
            y=level,
            line=dict(color="#8B95A1", width=1.0, dash="dot"),
            annotation_text=f"{level:.0%}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text="고점 대비 하락률 추이", font=dict(size=18, color="#191F28")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(tickformat=".0%", hoverformat=".2%"),
    )
    return fig


def build_vix_chart(vix_series: pd.Series, height: int = 300) -> go.Figure:
    """VIX 종가 시계열을 표시하는 참고용 라인 차트를 만든다."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vix_series.index,
            y=vix_series.values,
            mode="lines",
            name="VIX",
            line=dict(color="#6B4FBB", width=2.0),
        )
    )

    # 참고선 20(변동성 주의) / 30(높은 변동성) — 옅게 표시
    fig.add_hline(y=20, line=dict(color="#FFB020", width=1.0, dash="dot"),
                  annotation_text="변동성 주의 20", annotation_position="top left")
    fig.add_hline(y=30, line=dict(color="#D93D44", width=1.0, dash="dot"),
                  annotation_text="높은 변동성 30", annotation_position="top left")

    fig.update_layout(
        title=dict(text="VIX 추이", font=dict(size=16, color="#191F28")),
        height=height,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(title="VIX", hoverformat=".1f"),
    )
    return fig


# ──────────────────────────────────────────────
# 4. 화면
# ──────────────────────────────────────────────
render_market_temperature_css()
# 제목/부제는 새 카드형 상단 디자인의 render_title_area()에서만 1회 렌더링한다.
render_title_area()
# 기존 CNN 안내 블록은 제거하고, 출처 링크는 공포 & 탐욕 카드 오른쪽 위 CNN 링크만 유지한다.
render_market_briefing(get_fear_and_greed_data(), load_market_briefing_snapshots())

st.markdown("<div class='gorani-market-temp-section-gap'></div>", unsafe_allow_html=True)
st.markdown("### RSI/MDD 분석")
st.caption("QQQ · SCHD · SPY 의 RSI 14와 고점 대비 하락률을 확인합니다.")

ctrl_col, _ = st.columns([1, 3])
with ctrl_col:
    range_label = st.selectbox("표시 기간", list(RANGE_OPTIONS.keys()), index=1)
lookback = RANGE_OPTIONS[range_label]

closes, failures = load_watchlist(WATCHLIST)

# 데이터 로드 결과 안내 (실패해도 페이지는 계속 렌더)
if failures:
    failed_names = ", ".join(failures.keys())
    if not closes:
        st.warning(
            f"⚠️ 시세 데이터를 불러오지 못했습니다 ({failed_names}). "
            "잠시 후 다시 시도하거나, 아래 캐시 초기화 버튼을 눌러주세요."
        )
    else:
        st.info(f"ℹ️ 일부 종목 데이터를 불러오지 못했습니다: {failed_names} (나머지는 정상 표시됩니다.)")

# RSI / 하락률은 전체 종가로 미리 계산 (차트에서 함께 사용)
rsi_full = {t: compute_rsi(s, period=14) for t, s in closes.items()}
dd_full = {t: compute_drawdown_series(s) for t, s in closes.items()}


# ──────────────────────────────────────────────
# 4-1. RSI 14 / 고점 대비 하락률
# ──────────────────────────────────────────────
if closes:
    rsi_view = {t: _slice_recent(s, lookback) for t, s in rsi_full.items()}
    dd_view = {t: _slice_recent(s, lookback) for t, s in dd_full.items()}

    # ── 현재 RSI 14 카드 ──
    st.markdown("#### 현재 RSI 14")
    rsi_cols = st.columns(len(WATCHLIST))
    for col, ticker in zip(rsi_cols, WATCHLIST):
        value = _last_valid(rsi_full.get(ticker))
        if value is None:
            col.metric(ticker, "N/A")
        else:
            if value >= 70:
                state = "과매수"
            elif value <= 30:
                state = "과매도"
            else:
                state = "중립"
            col.metric(ticker, f"{value:.1f}", help=f"현재 상태: {state}")

    st.plotly_chart(build_rsi_chart(rsi_view), use_container_width=True)

    # ── 현재 고점 대비 하락률 카드 ──
    st.markdown("#### 현재 고점 대비 하락률")
    dd_cols = st.columns(len(WATCHLIST))
    for col, ticker in zip(dd_cols, WATCHLIST):
        value = _last_valid(dd_full.get(ticker))
        if value is None:
            col.metric(ticker, "N/A")
        else:
            col.metric(ticker, f"{value:.1%}")

    st.plotly_chart(build_drawdown_chart(dd_view), use_container_width=True)

    st.caption(
        "ℹ️ RSI 14 는 Wilder 방식으로 직접 계산하며, 70 이상은 과매수·30 이하는 과매도 신호로 해석합니다. "
        "고점 대비 하락률은 표시 구간 이전을 포함한 전체 고점 기준입니다."
    )


# 캐시 초기화 (시세/RSI/가격 데이터가 일시적으로 비어 있을 때 수동 갱신용)
if st.button("🔄 시세 캐시 초기화", use_container_width=True):
    fetch_close_series.clear()
    fetch_market_snapshot.clear()
    fetch_market_snapshot_batch.clear()
    fetch_fear_and_greed_data.clear()
    st.rerun()


# ──────────────────────────────────────────────
# 5. VIX 참고 그래프 (페이지 하단 참고 섹션)
#    - 현재값 카드가 아닌 시계열 그래프로 표시한다.
#    - 표시 기간 selectbox 와 동일한 lookback 으로 slicing 한다.
#    - VIX 조회 실패는 이 섹션만 안내하고, 다른 섹션은 그대로 유지한다.
# ──────────────────────────────────────────────
st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)
st.markdown("### 📉 VIX 참고 그래프")
st.caption("VIX는 시장 변동성 참고 지표입니다. 수치가 높을수록 시장 불안 심리가 커진 것으로 해석됩니다.")

try:
    vix_series = fetch_close_series("^VIX")
    vix_view = _slice_recent(vix_series, lookback)
    if vix_view is None or vix_view.dropna().empty:
        st.info("ℹ️ VIX 데이터를 표시할 수 없습니다. 잠시 후 다시 시도해주세요.")
    else:
        st.plotly_chart(build_vix_chart(vix_view), use_container_width=True)
except Exception:  # noqa: BLE001 - VIX 실패는 이 섹션만 영향, 페이지는 유지
    st.warning("⚠️ VIX(^VIX) 데이터를 불러오지 못했습니다. 나머지 섹션은 정상 표시됩니다.")


# ──────────────────────────────────────────────
# 6. 시장온도 참고 시트 (구글 스프레드시트 임베드) — 하단 참고 섹션
#    - 구글 시트를 iframe 으로 "보기"만 한다 (Google API/secrets/pandas 미사용).
#    - 시트가 로딩되지 않아도 위 RSI/하락률/VIX/트리맵 화면은 영향을 받지 않는다.
# ──────────────────────────────────────────────
st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)
st.markdown("### 📊 시장온도 참고 시트")

# 구글 시트 '웹에 게시 → 삽입(Embed)' URL.
# HTML 원본의 &amp; 는 Python URL 문자열에서 일반 & 로 사용한다.
sheet_url = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRQsjM2Yp05NyPTnXEeUuHrO8oiOJhuRmtDqIFQHOrsAGNnxVHDvs8eg0_qS-6CR5mnAG29v02j-fJ7/"
    "pubhtml?gid=331043462&single=true&widget=true&headers=false"
)

st.caption(
    "구글 시트가 보이지 않으면 아래 '새 탭에서 열기' 링크로 열거나, "
    "시트의 '웹에 게시(Publish to web)' 설정을 확인해주세요."
)

# iframe 임베드 (높이 800, 스크롤 허용). 외부 시트 로딩 실패는 iframe 내부 문제로
# 한정되며 Streamlit 앱 전체를 중단시키지 않는다.
components.iframe(sheet_url, height=800, scrolling=True)

# 새 탭에서 열기 링크 (iframe 이 막혀도 사용자가 직접 확인할 수 있도록 제공)
st.markdown(
    f"<div style='font-size:13px; margin-top:8px;'>🔗 "
    f"<a href='{sheet_url}' target='_blank' rel='noopener noreferrer'>"
    "새 탭에서 시장온도 참고 시트 열기</a></div>",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# 7. TradingView 미국주식 섹터 트리맵 — 페이지 최하단
# ──────────────────────────────────────────────
render_tradingview_heatmap()
