from html import escape
from io import StringIO
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


def render_html_block(html: str) -> None:
    """HTML 조각을 Markdown 코드블록으로 오인되지 않게 정리해 렌더링한다."""
    cleaned = dedent(html).strip()
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


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


MARKET_BRIEF_CARDS = {
    "sp500": {"label": "S&P 500", "ticker": "^GSPC", "kind": "large", "tone": "blue", "unit": ""},
    "nasdaq": {"label": "NASDAQ", "ticker": "^IXIC", "kind": "large", "tone": "blue", "unit": ""},
    "dow": {"label": "DOW JONES", "ticker": "^DJI", "kind": "large", "tone": "blue", "unit": ""},
    "vix": {"label": "VIX (공포지수)", "ticker": "^VIX", "kind": "large", "tone": "red", "unit": ""},
    "usdkrw": {"label": "USD/KRW", "ticker": "KRW=X", "kind": "small", "tone": "blue", "unit": ""},
    "wti": {"label": "WTI", "ticker": "CL=F", "kind": "small", "tone": "red", "unit": "$"},
    "gold": {"label": "GOLD", "ticker": "GC=F", "kind": "small", "tone": "red", "unit": "$"},
}

MARKET_BRIEF_ORDER = ["sp500", "nasdaq", "dow", "vix", "usdkrw", "wti", "gold"]


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


def _extract_close_from_yfinance(df: pd.DataFrame, ticker: str) -> pd.Series:
    """yfinance 단일/복수 티커 컬럼 구조에서 Close 계열을 안전하게 추출한다."""
    if df is None or df.empty:
        raise ValueError("조회 결과가 비어 있습니다.")

    data = df.copy()
    close = None

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data.xs("Close", axis=1, level=0, drop_level=False)
            if ticker in close.columns.get_level_values(-1):
                close = close.xs(ticker, axis=1, level=-1).squeeze()
            else:
                close = close.droplevel(0, axis=1).iloc[:, 0]
        elif "Close" in data.columns.get_level_values(-1):
            close = data.xs("Close", axis=1, level=-1, drop_level=False)
            close = close.droplevel(-1, axis=1).iloc[:, 0]
    elif "Close" in data.columns:
        close = data["Close"]

    if close is None:
        raise ValueError(f"Close 컬럼이 없습니다. 사용 가능 컬럼: {list(data.columns)}")

    close = pd.to_numeric(pd.Series(close).dropna(), errors="coerce").dropna()
    if close.empty:
        raise ValueError("Close 데이터가 비어 있습니다.")
    return close.astype("float64")


def _format_market_value(value: float | None, unit: str = "") -> str:
    if value is None:
        return "데이터 없음"
    if unit == "$":
        return f"${value:,.2f}" if value < 1000 else f"${value:,.0f}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _format_market_change(change: float | None, change_pct: float | None, label: str) -> str:
    if change is None and change_pct is None:
        return ""
    direction = "▲" if (change or 0) > 0 else "▼" if (change or 0) < 0 else "–"
    change_unit = "p" if label in {"S&P 500", "NASDAQ", "DOW JONES"} else ""
    change_text = ""
    if change is not None:
        change_text = f"{direction} {abs(change):,.2f}{change_unit}"
    pct_text = f"{change_pct:+.2f}%" if change_pct is not None else ""
    return " ".join(part for part in [change_text, pct_text] if part)


def _market_status_badge(card_key: str, change_pct: float | None, value: float | None) -> str:
    if card_key == "vix" and value is not None:
        if value >= 30:
            return "공포 확대"
        if value >= 20:
            return "변동성 주의"
    if change_pct is not None:
        if change_pct <= -2:
            return "조정 진입"
        if change_pct >= 2:
            return "강한 반등"
    return ""


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_market_brief_quote(ticker: str) -> dict:
    """상단 시장 브리핑 카드용 최신가/전일 대비 데이터를 조회한다."""
    if not ticker:
        raise ValueError("티커가 비어 있습니다.")

    errors = []
    for period in ("5d", "1mo"):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                timeout=15,
            )
            close = _extract_close_from_yfinance(df, ticker)
            latest = float(close.iloc[-1])
            previous = float(close.iloc[-2]) if len(close) >= 2 else None
            change = latest - previous if previous not in (None, 0) else None
            change_pct = (change / previous * 100) if change is not None and previous else None
            return {
                "ticker": ticker,
                "value": latest,
                "previous": previous,
                "change": change,
                "change_pct": change_pct,
                "error": "",
            }
        except Exception as e:  # noqa: BLE001 - 카드 단위 실패 격리
            errors.append(f"{period}: {e}")

    return {"ticker": ticker, "value": None, "previous": None, "change": None, "change_pct": None, "error": " | ".join(errors)}


def load_market_brief_cards() -> dict:
    """7개 브리핑 카드 데이터를 카드별로 격리해 조회한다."""
    quotes = {}
    for key, meta in MARKET_BRIEF_CARDS.items():
        try:
            quotes[key] = fetch_market_brief_quote(meta["ticker"])
        except Exception as e:  # noqa: BLE001 - 페이지 전체 중단 방지
            quotes[key] = {"ticker": meta["ticker"], "value": None, "previous": None, "change": None, "change_pct": None, "error": str(e)}
    return quotes


def _render_market_brief_card(key: str, meta: dict, quote: dict) -> str:
    label = escape(meta["label"])
    value = quote.get("value")
    change = quote.get("change")
    change_pct = quote.get("change_pct")
    has_error = value is None
    value_text = "조회 실패" if quote.get("error") and has_error else _format_market_value(value, meta.get("unit", ""))
    change_text = _format_market_change(change, change_pct, meta["label"])
    badge = _market_status_badge(key, change_pct, value)
    direction_class = "is-up" if (change or 0) > 0 else "is-down" if (change or 0) < 0 else "is-flat"
    kind_class = "gorani-market-temp-card--small" if meta.get("kind") == "small" else "gorani-market-temp-card--large"
    tone_class = f"gorani-market-temp-card--{meta.get('tone', 'blue')}"
    error_text = "데이터 없음" if has_error else ""

    return dedent(
        f"""
        <div class="gorani-market-temp-card {kind_class} {tone_class}">
          <div class="gorani-market-temp-label">{label}</div>
          <div class="gorani-market-temp-value">{escape(value_text)}</div>
          <div class="gorani-market-temp-change {direction_class}">{escape(change_text)}</div>
          {f'<div class="gorani-market-temp-badge">{escape(badge)}</div>' if badge else ''}
          {f'<div class="gorani-market-temp-error">{escape(error_text)}</div>' if error_text else ''}
        </div>
        """
    ).strip()


def render_market_brief(quotes: dict) -> None:
    cards = {
        key: _render_market_brief_card(key, MARKET_BRIEF_CARDS[key], quotes.get(key, {}))
        for key in MARKET_BRIEF_ORDER
    }
    html = dedent(
        f"""
        <style>
          .gorani-market-temp-hero {{
            padding: 3.2rem 0 1.3rem;
            background: #FFFFFF;
          }}
          .gorani-market-temp-title {{
            margin: 0 0 0.65rem;
            color: #191F28;
            font-size: clamp(2.1rem, 5vw, 3rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.15;
          }}
          .gorani-market-temp-subtitle {{
            margin: 0;
            color: #8B95A1;
            font-size: 0.98rem;
            line-height: 1.6;
          }}
          .gorani-market-temp-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(150px, 0.28fr);
            gap: 1.35rem;
            align-items: stretch;
            margin: 1.6rem 0 2.3rem;
          }}
          .gorani-market-temp-large-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
          }}
          .gorani-market-temp-small-stack {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
          }}
          .gorani-market-temp-card {{
            position: relative;
            overflow: hidden;
            min-height: 132px;
            padding: 1.05rem 1.25rem 1rem;
            border: 1px solid #E5E8EB;
            border-radius: 16px;
            background: #FFFFFF;
            box-shadow: 0 7px 18px rgba(25, 31, 40, 0.055);
          }}
          .gorani-market-temp-card--large::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: #3182F6;
          }}
          .gorani-market-temp-card--red::before {{ background: #E42939; }}
          .gorani-market-temp-card--small {{
            min-height: 104px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: center;
            padding: 1rem;
          }}
          .gorani-market-temp-small-stack .gorani-market-temp-card--blue {{ background: #F3F6FB; }}
          .gorani-market-temp-small-stack .gorani-market-temp-card--red {{ background: #FFF2F2; border-color: #FFE0E0; }}
          .gorani-market-temp-label {{
            color: #A3ADBD;
            font-size: 0.9rem;
            font-weight: 800;
            letter-spacing: 0.02em;
          }}
          .gorani-market-temp-value {{
            margin-top: 0.42rem;
            color: #191F28;
            font-size: clamp(1.7rem, 3.5vw, 2.35rem);
            font-weight: 850;
            letter-spacing: -0.035em;
            line-height: 1.1;
          }}
          .gorani-market-temp-card--red .gorani-market-temp-value,
          .gorani-market-temp-small-stack .gorani-market-temp-card--red .gorani-market-temp-value {{ color: #E42939; }}
          .gorani-market-temp-change {{
            min-height: 1.45rem;
            margin-top: 0.7rem;
            font-size: 0.98rem;
            font-weight: 800;
          }}
          .gorani-market-temp-change.is-down {{ color: #3182F6; }}
          .gorani-market-temp-change.is-up {{ color: #E42939; }}
          .gorani-market-temp-change.is-flat {{ color: #6B7684; }}
          .gorani-market-temp-badge {{
            display: inline-flex;
            width: fit-content;
            margin-top: 0.55rem;
            padding: 0.24rem 0.5rem;
            border-radius: 8px;
            background: #FFF1F1;
            color: #D93D44;
            font-size: 0.82rem;
            font-weight: 800;
          }}
          .gorani-market-temp-error {{
            margin-top: 0.45rem;
            color: #8B95A1;
            font-size: 0.82rem;
          }}
          @media (max-width: 900px) {{
            .gorani-market-temp-grid {{ grid-template-columns: 1fr; }}
            .gorani-market-temp-small-stack {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
          }}
          @media (max-width: 640px) {{
            .gorani-market-temp-hero {{ padding-top: 2rem; }}
            .gorani-market-temp-large-grid,
            .gorani-market-temp-small-stack {{ grid-template-columns: 1fr; }}
            .gorani-market-temp-card {{ min-height: 116px; }}
          }}
        </style>
        <section class="gorani-market-temp-hero">
          <h1 class="gorani-market-temp-title">🌡️ 시장온도</h1>
          <p class="gorani-market-temp-subtitle">QQQ · SCHD · SPY 의 RSI 14 와 고점 대비 하락률로 시장의 과열/침체 상태를 살펴봅니다.</p>
        </section>
        <section class="gorani-market-temp-grid" aria-label="핵심 시장지표">
          <div class="gorani-market-temp-large-grid">
            {cards['sp500']}
            {cards['nasdaq']}
            {cards['dow']}
            {cards['vix']}
          </div>
          <div class="gorani-market-temp-small-stack">
            {cards['usdkrw']}
            {cards['wti']}
            {cards['gold']}
          </div>
        </section>
        """
    ).strip()
    render_html_block(html)


def render_tradingview_heatmap() -> None:
    st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)
    st.markdown("### 🇺🇸 미국주식 섹터 트리맵")
    st.caption("S&P 500 구성종목의 섹터별 흐름을 TradingView 히트맵으로 확인합니다.")

    tradingview_heatmap_html = dedent(
        """
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="UTF-8" />
            <style>
              html,
              body {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 720px;
                background: #ffffff;
                overflow: hidden;
              }
              .tradingview-widget-container,
              .tradingview-widget-container__widget {
                width: 100%;
                height: 700px;
                margin: 0;
                padding: 0;
              }
              .tradingview-widget-container iframe {
                width: 100% !important;
                height: 700px !important;
                min-height: 700px !important;
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
                "height": "100%"
              }
              </script>
            </div>
          </body>
        </html>
        """
    ).strip()

    try:
        components.html(tradingview_heatmap_html, height=730, scrolling=False)
    except Exception:  # noqa: BLE001 - 외부 위젯 실패가 페이지 전체로 전파되지 않도록 방어
        st.info("TradingView 히트맵 위젯을 불러오지 못했습니다. 브라우저 외부 스크립트 차단 설정을 확인해주세요.")

    render_html_block(
        """
        <div style="font-size:12px; color:#8b95a1; margin-top:6px;">
          <a href="https://www.tradingview.com/widget/stock-heatmap/"
             target="_blank"
             rel="noopener noreferrer"
             style="color:#8b95a1; text-decoration:none;">
            TradingView Stock Heatmap 공식 위젯 열기
          </a>
        </div>
        """
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
market_brief_quotes = load_market_brief_cards()
render_market_brief(market_brief_quotes)

st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)

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
# 4-1. 공포·탐욕(공탐) 지수 안내 — CNN 공식 페이지 바로가기
#      자체 게이지/진단 대신 CNN 공식값을 직접 확인하도록 안내한다.
# ──────────────────────────────────────────────
st.markdown(
    "<div style='font-size:13px;'>🔗 "
    "<a href='https://www.cnn.com/markets/fear-and-greed' "
    "target='_blank' rel='noopener noreferrer'>"
    "CNN Fear & Greed 공식 페이지에서 공탐지수 확인</a></div>",
    unsafe_allow_html=True,
)
st.caption(
    "공탐지수는 CNN 공식 페이지에서 직접 확인합니다. "
    "이 앱은 RSI와 고점대비 하락률 중심으로 시장 상태를 참고합니다."
)


# ──────────────────────────────────────────────
# 4-2. RSI 14 / 고점 대비 하락률
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
    fetch_market_brief_quote.clear()
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
# 6. 시장온도 참고 시트 (구글 스프레드시트 임베드) — 탭 최하단
#    - 구글 시트를 iframe 으로 "보기"만 한다 (Google API/secrets/pandas 미사용).
#    - 시트가 로딩되지 않아도 위 RSI/하락률/VIX 화면은 영향을 받지 않는다.
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
