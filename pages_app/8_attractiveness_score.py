import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from ui.styles import TOSS_CSS


TICKERS = ["QQQ", "SCHD", "VOO", "QLD"]
COLORS = {
    "QQQ": "#ff4fb3",
    "SCHD": "#f2c94c",
    "VOO": "#f2994a",
    "QLD": "#eb5757",
}
PARAMS = {
    "QQQ": {"base_dd": 9, "step_dd": 4, "target_rsi": 40, "rsi_sigma": 15},
    "VOO": {"base_dd": 7, "step_dd": 3, "target_rsi": 42, "rsi_sigma": 14},
    "SCHD": {"base_dd": 7, "step_dd": 3, "target_rsi": 43, "rsi_sigma": 13},
    "QLD": {"base_dd": 15, "step_dd": 6, "target_rsi": 36, "rsi_sigma": 16},
}
LINE_OPTIONS = [f"{ticker} 매력점수" for ticker in TICKERS] + [f"{ticker} 실제 종가" for ticker in TICKERS]
CACHE_TTL_SECONDS = 60 * 60 * 6


# ──────────────────────────────────────────────
# 1. 데이터/계산 함수
# ──────────────────────────────────────────────
def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("조회 결과가 비어 있습니다.")

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    price_col = "Close"
    if price_col not in normalized.columns and "Adj Close" in normalized.columns:
        price_col = "Adj Close"
    if price_col not in normalized.columns:
        raise ValueError(f"Close/Adj Close 컬럼이 없습니다. 사용 가능 컬럼: {list(normalized.columns)}")

    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized[~normalized.index.isna()]
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.sort_index()
    normalized["close"] = pd.to_numeric(normalized[price_col], errors="coerce")
    normalized = normalized.dropna(subset=["close"])
    normalized = normalized[normalized["close"] > 0]

    if normalized.empty:
        raise ValueError("유효한 종가 데이터가 없습니다.")
    return normalized[["close"]].astype("float64")


def _download_single_ticker(ticker: str, start=None, end=None) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        period="max" if start is None else None,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        timeout=20,
    )
    return _normalize_price_frame(df)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_price_data(tickers, start=None, end=None):
    """SCHD의 실제 earliest valid date 이후 공통 기간 가격 데이터를 가져온다.

    Returns:
        tuple[dict[str, DataFrame], dict[str, str], str | None]:
        정상 티커별 가격 데이터, 실패 메시지, SCHD 기준 시작일 문자열.
    """
    requested = [str(t).upper().strip() for t in tickers if str(t).strip()]
    errors = {}
    raw_data = {}

    try:
        schd_full = _download_single_ticker("SCHD")
        schd_start = schd_full.index.min().normalize()
        raw_data["SCHD"] = schd_full
    except Exception as exc:
        errors["SCHD"] = str(exc)
        schd_start = pd.Timestamp(start).normalize() if start else None

    for ticker in requested:
        if ticker == "SCHD" and ticker in raw_data:
            continue
        try:
            raw_data[ticker] = _download_single_ticker(ticker)
        except Exception as exc:
            errors[ticker] = str(exc)

    if not raw_data:
        return {}, errors, None

    valid_starts = [df.index.min().normalize() for df in raw_data.values() if not df.empty]
    common_start = max([d for d in [schd_start, *valid_starts] if d is not None])
    common_end = pd.Timestamp(end).normalize() if end else min(df.index.max().normalize() for df in raw_data.values())

    price_data = {}
    for ticker, df in raw_data.items():
        window = df.loc[(df.index >= common_start) & (df.index <= common_end)].copy()
        if window.empty:
            errors[ticker] = f"공통 기간({common_start:%Y-%m-%d} 이후)에 유효 데이터가 없습니다."
        else:
            price_data[ticker] = window

    schd_start_text = schd_start.strftime("%Y-%m-%d") if schd_start is not None else None
    return price_data, errors, schd_start_text


def calculate_wilder_rsi(close, period=14) -> pd.Series:
    close = pd.Series(close).astype("float64")
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    rsi = pd.Series(np.nan, index=close.index, dtype="float64")
    if len(close) <= period:
        return rsi

    avg_gain = gains.iloc[1 : period + 1].mean()
    avg_loss = losses.iloc[1 : period + 1].mean()

    for i in range(period, len(close)):
        if i > period:
            avg_gain = ((avg_gain * (period - 1)) + gains.iloc[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses.iloc[i]) / period

        if pd.isna(avg_gain) or pd.isna(avg_loss):
            continue
        if avg_loss == 0:
            rsi.iloc[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi.iloc[i] = 100 - (100 / (1 + rs))

    return rsi.clip(0, 100)


def calculate_attractiveness_for_ticker(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    scored = df.copy()
    scored["close"] = pd.to_numeric(scored["close"], errors="coerce")
    scored = scored.dropna(subset=["close"])
    scored = scored[scored["close"] > 0]

    scored["running_high"] = scored["close"].cummax()
    scored["dd_pct"] = ((1 - scored["close"] / scored["running_high"]) * 100).clip(lower=0)

    x = (scored["dd_pct"] - params["base_dd"]) / params["step_dd"]
    scored["dd_score"] = np.where(x >= 0, 100 - 50 * (0.5**x), 50 * (2**x))
    scored["dd_score"] = pd.Series(scored["dd_score"], index=scored.index).clip(0, 100)

    scored["rsi14"] = calculate_wilder_rsi(scored["close"], period=14)
    scored["rsi_score"] = 100 * np.exp(
        -((scored["rsi14"] - params["target_rsi"]) ** 2) / (2 * params["rsi_sigma"] ** 2)
    )

    scored["mdd_rarity_score"] = scored["dd_pct"].rank(pct=True, method="average") * 100
    scored["max_dd"] = float(scored["dd_pct"].max()) if not scored.empty else 0.0
    scored["depth_ratio"] = np.where(scored["max_dd"] > 0, scored["dd_pct"] / scored["max_dd"] * 100, 0)
    scored["mdd_score"] = scored["mdd_rarity_score"] * 0.75 + scored["depth_ratio"] * 0.25

    dd_component = ((scored["dd_score"] - 50).clip(lower=0) / 50).fillna(0)
    mdd_component = ((scored["mdd_score"] - 50).clip(lower=0) / 50).fillna(0)
    rsi_component = ((scored["rsi_score"] - 50).clip(lower=0) / 50).fillna(0)
    scored["combination_bonus"] = 8 * np.sqrt(dd_component * mdd_component * rsi_component)

    sigmoid = 1 / (1 + np.exp(-((scored["rsi14"] - 70) / 4)))
    scored["overheat_penalty"] = 10 * sigmoid * (1 - scored["dd_score"] / 100)

    scored["attractiveness_score"] = (
        scored["dd_score"] * 0.45
        + scored["mdd_score"] * 0.35
        + scored["rsi_score"] * 0.20
        + scored["combination_bonus"]
        - scored["overheat_penalty"]
    ).clip(0, 100)

    scored = scored.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [
        "close",
        "running_high",
        "dd_pct",
        "dd_score",
        "rsi14",
        "rsi_score",
        "mdd_rarity_score",
        "max_dd",
        "depth_ratio",
        "mdd_score",
        "combination_bonus",
        "overheat_penalty",
        "attractiveness_score",
    ]
    scored[numeric_cols] = scored[numeric_cols].astype("float64")
    return scored


def build_combined_chart(scored_df: pd.DataFrame, selected_lines, normalize_price=False) -> go.Figure:
    fig = go.Figure()
    selected = set(selected_lines or [])

    for ticker in TICKERS:
        ticker_df = scored_df[scored_df["ticker"] == ticker].sort_index()
        if ticker_df.empty:
            continue

        score_name = f"{ticker} 매력점수"
        price_name = f"{ticker} 실제 종가"
        color = COLORS[ticker]

        if score_name in selected:
            fig.add_trace(
                go.Scatter(
                    x=ticker_df.index,
                    y=ticker_df["attractiveness_score"],
                    mode="lines",
                    name=score_name,
                    line=dict(color=color, width=2.4),
                    yaxis="y",
                    customdata=np.column_stack([ticker_df.index.strftime("%Y-%m-%d")]),
                    hovertemplate=f"{ticker}<br>날짜: %{{customdata[0]}}<br>매력점수: %{{y:.2f}}<extra></extra>",
                )
            )

        if price_name in selected:
            price_series = ticker_df["close"].copy()
            if normalize_price:
                first_valid = price_series.dropna().iloc[0] if not price_series.dropna().empty else np.nan
                price_series = price_series / first_valid * 100 if pd.notna(first_valid) and first_valid != 0 else price_series * np.nan
            fig.add_trace(
                go.Scatter(
                    x=ticker_df.index,
                    y=price_series,
                    mode="lines",
                    name=price_name if not normalize_price else f"{ticker} 가격지수",
                    line=dict(color=color, width=1.8, dash="dot"),
                    yaxis="y2",
                    customdata=np.column_stack([ticker_df.index.strftime("%Y-%m-%d")]),
                    hovertemplate=(
                        f"{ticker}<br>날짜: %{{customdata[0]}}<br>"
                        + ("가격지수: %{y:.2f}" if normalize_price else "종가: $%{y:,.2f}")
                        + "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(text="ETF 매수 매력도 점수와 가격", font=dict(size=19, color="#191F28")),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        height=680,
        margin=dict(l=20, r=22, t=62, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="날짜", hoverformat="%Y-%m-%d", rangeslider=dict(visible=False)),
        yaxis=dict(title="매수 매력도 점수", range=[0, 100], tickformat=".0f", gridcolor="#F2F4F6"),
        yaxis2=dict(
            title="가격 지수, 시작일=100" if normalize_price else "실제 종가 USD",
            overlaying="y",
            side="right",
            showgrid=False,
            tickprefix="" if normalize_price else "$",
            tickformat=",.0f" if normalize_price else ",.2f",
        ),
    )
    return fig


def build_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        ticker_df = scored_df[scored_df["ticker"] == ticker].dropna(subset=["attractiveness_score"])
        if ticker_df.empty:
            continue
        latest = ticker_df.sort_index().iloc[-1]
        rows.append(
            {
                "Ticker": ticker,
                "Attractiveness Score": float(latest["attractiveness_score"]),
                "DD %": float(latest["dd_pct"]),
                "RSI14": float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else np.nan,
                "Date": latest.name,
            }
        )
    return pd.DataFrame(rows)


def build_detail_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    detail = scored_df.reset_index().rename(
        columns={
            "index": "Date",
            "ticker": "Ticker",
            "close": "Close",
            "dd_pct": "DD %",
            "rsi14": "RSI14",
            "dd_score": "DD Score",
            "mdd_score": "MDD Score",
            "rsi_score": "RSI Score",
            "combination_bonus": "Bonus",
            "overheat_penalty": "Penalty",
            "attractiveness_score": "Attractiveness Score",
        }
    )
    columns = [
        "Date",
        "Ticker",
        "Close",
        "DD %",
        "RSI14",
        "DD Score",
        "MDD Score",
        "RSI Score",
        "Bonus",
        "Penalty",
        "Attractiveness Score",
    ]
    detail = detail[columns].copy()
    detail["Date"] = pd.to_datetime(detail["Date"]).dt.date
    numeric_cols = [col for col in columns if col not in {"Date", "Ticker"}]
    detail[numeric_cols] = detail[numeric_cols].round(2)
    return detail.sort_values("Date", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_scored_data(tickers, start=None, end=None):
    price_data, errors, schd_start = get_price_data(tickers, start, end)
    scored_frames = []
    for ticker, df in price_data.items():
        params = PARAMS.get(ticker)
        if params is None:
            continue
        scored = calculate_attractiveness_for_ticker(df, params)
        scored["ticker"] = ticker
        scored_frames.append(scored)

    if not scored_frames:
        return pd.DataFrame(), errors, schd_start

    scored_df = pd.concat(scored_frames, axis=0).sort_index()
    scored_df = scored_df.replace([np.inf, -np.inf], np.nan)
    return scored_df, errors, schd_start


# ──────────────────────────────────────────────
# 2. UI 함수
# ──────────────────────────────────────────────
def _inject_page_css() -> None:
    st.markdown(
        """
        <style>
            .attractiveness-card-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
                gap: 14px;
                margin: 14px 0 20px 0;
            }
            .attractiveness-card {
                background: #FFFFFF;
                border: 1px solid #F2F4F6;
                border-radius: 18px;
                padding: 18px 18px 16px 18px;
                box-shadow: 0 6px 18px rgba(25, 31, 40, 0.05);
            }
            .attractiveness-card-title {
                color: #4E5968;
                font-size: 14px;
                font-weight: 700;
                margin-bottom: 8px;
            }
            .attractiveness-card-score {
                color: #191F28;
                font-size: 28px;
                font-weight: 800;
                line-height: 1.15;
                margin-bottom: 10px;
            }
            .attractiveness-card-meta {
                color: #6B7684;
                font-size: 13px;
                line-height: 1.55;
            }
            .attractiveness-note {
                color: #8B95A1;
                font-size: 12px;
                line-height: 1.55;
            }
            @media (max-width: 640px) {
                .attractiveness-card-grid {
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 10px;
                }
                .attractiveness-card { padding: 15px 14px; border-radius: 16px; }
                .attractiveness-card-score { font-size: 24px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_number(value, suffix="", prefix="") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{prefix}{float(value):,.2f}{suffix}"


def _render_summary_cards(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    card_html = ['<div class="attractiveness-card-grid">']
    for _, row in summary_df.iterrows():
        ticker = row["Ticker"]
        color = COLORS.get(ticker, "#3182F6")
        card_html.append(
            f"""
            <div class="attractiveness-card">
                <div class="attractiveness-card-title" style="color:{color};">{ticker}</div>
                <div class="attractiveness-card-score">{_format_number(row['Attractiveness Score'])}</div>
                <div class="attractiveness-card-meta">
                    최신 DD: <b>{_format_number(row['DD %'], '%')}</b><br>
                    최신 RSI: <b>{_format_number(row['RSI14'])}</b>
                </div>
            </div>
            """
        )
    card_html.append("</div>")
    st.markdown("".join(card_html), unsafe_allow_html=True)


def _render_detail_expander(detail_df: pd.DataFrame) -> None:
    with st.expander("날짜별 상세 데이터 보기", expanded=False):
        if detail_df.empty:
            st.info("표시할 상세 데이터가 없습니다.")
            return

        filter_cols = st.columns([1, 1.4])
        with filter_cols[0]:
            selected_tickers = st.multiselect(
                "티커 필터",
                options=TICKERS,
                default=[ticker for ticker in TICKERS if ticker in detail_df["Ticker"].unique()],
                key="attractiveness_detail_ticker_filter",
            )
        min_date = detail_df["Date"].min()
        max_date = detail_df["Date"].max()
        with filter_cols[1]:
            selected_range = st.date_input(
                "날짜 범위 필터",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                format="YYYY/MM/DD",
                key="attractiveness_detail_date_filter",
            )

        filtered = detail_df.copy()
        if selected_tickers:
            filtered = filtered[filtered["Ticker"].isin(selected_tickers)]
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
            filtered = filtered[(filtered["Date"] >= start_date) & (filtered["Date"] <= end_date)]

        display_df = filtered.head(500)
        st.caption(f"기본 표시: 최근 500행 / 필터 후 전체 {len(filtered):,}행")
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=420)

        csv = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 필터된 전체 데이터 CSV 다운로드",
            data=csv,
            file_name=f"attractiveness_detail_{date.today():%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_page() -> None:
    st.markdown(TOSS_CSS, unsafe_allow_html=True)
    _inject_page_css()

    st.markdown("# 📈 매력점수")
    st.caption(
        "매력점수는 고점 대비 하락률(DD), MDD 레어도, RSI 냉각도를 조합한 순수 시각화용 지표입니다. "
        "자동매수 신호가 아니며, 점수가 높을수록 상대적으로 가격 매력이 커졌다는 뜻입니다."
    )
    st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6;'>", unsafe_allow_html=True)

    with st.spinner("QQQ · SCHD · VOO · QLD 가격 데이터와 매력점수를 계산하는 중입니다..."):
        scored_df, errors, schd_start = _load_scored_data(tuple(TICKERS), None, None)

    if schd_start:
        st.caption(
            f"데이터 기준: yfinance에서 확인한 SCHD 실제 시작일({schd_start}) 이후 공통 기간입니다. "
            "SCHD처럼 배당 영향이 큰 ETF는 adjusted 기준 가격 계산을 우선 사용합니다."
        )
    st.markdown(
        "<div class='attractiveness-note'>MDD 레어도는 프로토타입 시각화를 위해 전체 기간 DD 분포의 percentile rank를 사용합니다. "
        "따라서 look-ahead bias가 있으며 백테스트/자동매수 판단용이 아닙니다.</div>",
        unsafe_allow_html=True,
    )

    if errors:
        st.warning("일부 티커 데이터 조회에 실패했습니다. 가능한 티커만 차트와 표에 표시합니다.")
        for ticker, message in errors.items():
            st.caption(f"{ticker}: {message}")

    if scored_df.empty:
        st.error("표시 가능한 ETF 가격 데이터가 없습니다. 네트워크 상태 또는 yfinance 응답을 확인해주세요.")
        if st.button("🔄 매력점수 캐시 초기화", use_container_width=True):
            get_price_data.clear()
            _load_scored_data.clear()
            st.rerun()
        return

    summary_df = build_summary(scored_df)
    _render_summary_cards(summary_df)

    available_lines = []
    for line in LINE_OPTIONS:
        ticker = line.split()[0]
        if ticker in scored_df["ticker"].unique():
            available_lines.append(line)

    controls = st.columns([2, 1, 1])
    with controls[0]:
        selected_lines = st.multiselect(
            "표시할 선 선택",
            options=available_lines,
            default=available_lines,
            key="attractiveness_selected_lines",
        )
    with controls[1]:
        normalize_price = st.toggle("가격선을 100 기준 정규화해서 보기", value=False)
    with controls[2]:
        show_range_slider = st.toggle("기간 슬라이더 표시", value=False)

    fig = build_combined_chart(scored_df, selected_lines, normalize_price=normalize_price)
    fig.update_xaxes(rangeslider=dict(visible=show_range_slider))
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": True})

    st.caption(
        "ℹ️ 실제 종가 4개는 동일한 오른쪽 Y축에 표시되어 가격대 차이 때문에 일부 점선이 덜 두드러질 수 있습니다. "
        "필요하면 정규화 토글을 켜서 시작일=100 기준 가격 흐름을 비교하세요."
    )

    detail_df = build_detail_table(scored_df)
    _render_detail_expander(detail_df)

    if st.button("🔄 매력점수 캐시 초기화", use_container_width=True):
        get_price_data.clear()
        _load_scored_data.clear()
        st.rerun()


if not os.environ.get("GORANI_SKIP_PAGE_RENDER"):
    render_page()
