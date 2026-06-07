import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from logic.tracker import parse_data, process_data
from logic.tracker_performance import build_tracker_performance, default_start_date_for_latest
from ui.styles import TOSS_CSS
from core.sync import auto_save_tracker_data, auto_save_config 

# ---------------------------------------------------------
# ✅ HTML 버전의 자산군 그룹화 및 색상 로직
# ---------------------------------------------------------
COLOR_CASH = '#76FF03'
COLOR_DOLLAR = '#2E7D32'
COLOR_LEVERAGE = '#890600'
COLOR_NASDAQ = '#FC3A2F'
COLOR_SPY = '#FF6600'
COLOR_DIVIDEND = '#FFEB3B'
COLOR_OTHER = ['#2196F3','#9C27B0','#00BCD4','#E91E63','#673AB7','#03A9F4','#3F51B5','#009688']

def get_asset_type(tag):
    lower = tag.lower()
    if any(x in lower for x in ['달러', 'usd', 'dollar']): return 'dollar'
    if any(x in lower for x in ['현금', '예금', '적금', '채권', 'rp', '저축', 'cma', 'mmf', '파킹', '입출금']): return 'cash'
    if any(x in lower for x in ['tqqq', 'qld', 'upro', 'soxl', 'tecl', 'fngu', 'bulz', 'sso', '레버리지', '3x', '2x']): return 'leverage'
    if any(x in lower for x in ['qqq', 'qqqm', '나스닥']): return 'nasdaq'
    if any(x in lower for x in ['spy', 'voo', 'ivv', 'splg', 's&p', 'sp500', 'snp']): return 'spy'
    if any(x in lower for x in ['msft', 'schd', 'vym', 'dgro', 'aapl', 'ko', 'jnj', 'pg', 'vti', 'vtv', 'vug', 'dia', '배당', 'dividend']): return 'dividend'
    return 'other'

def get_super_group(atype):
    if atype in ['spy', 'dividend']: return 'spy_div'
    if atype in ['cash', 'dollar']: return 'cash_dol'
    if atype in ['leverage', 'nasdaq']: return 'lev_nas'
    return 'other_grp'

def assign_colors(tag_list):
    colors = {}
    other_idx = 0
    for tag in tag_list:
        atype = get_asset_type(tag)
        if atype == 'cash': colors[tag] = COLOR_CASH
        elif atype == 'dollar': colors[tag] = COLOR_DOLLAR
        elif atype == 'leverage': colors[tag] = COLOR_LEVERAGE
        elif atype == 'nasdaq': colors[tag] = COLOR_NASDAQ
        elif atype == 'spy': colors[tag] = COLOR_SPY
        elif atype == 'dividend': colors[tag] = COLOR_DIVIDEND
        else:
            colors[tag] = COLOR_OTHER[other_idx % len(COLOR_OTHER)]
            other_idx += 1
    return colors

def sort_tags_by_super_group(tag_entries):
    sg_totals = {'spy_div': 0, 'cash_dol': 0, 'lev_nas': 0, 'other_grp': 0}
    type_totals = {}
    type_map = {}
    
    for tag, val in tag_entries:
        atype = get_asset_type(tag)
        type_map[tag] = atype
        sg = get_super_group(atype)
        sg_totals[sg] += val
        type_totals[atype] = type_totals.get(atype, 0) + val
        
    def sort_key(item):
        tag, val = item
        atype = type_map[tag]
        sg = get_super_group(atype)
        return (sg_totals[sg], type_totals[atype], val)
        
    return sorted(tag_entries, key=sort_key, reverse=True)

# 1. 디자인 및 유틸리티 설정
st.markdown(TOSS_CSS, unsafe_allow_html=True)

def fmt_won(v): return f"{int(round(v)):,}원"

def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v * 100:+.1f}%"

def render_performance_card(title, value, subtitle, color="#191f28", border_color="#E5E8EB"):
    st.markdown(
        f"""
        <div style="background:#fffefa; border:1px solid {border_color}; border-radius:16px; padding:20px; min-height:118px; box-shadow:0 4px 18px rgba(0,0,0,0.03);">
            <div style="font-size:14px; color:#6b7684; font-weight:700; margin-bottom:18px;">{title}</div>
            <div style="font-size:24px; color:{color}; font-weight:800; letter-spacing:-0.5px; word-break:keep-all;">{value}</div>
            <div style="font-size:12px; color:#4e5968; font-weight:700; margin-top:12px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def card_open():
    st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6; opacity:0.2; margin:20px 0;'>", unsafe_allow_html=True)

def card_close(): pass

@st.dialog("⚠️ 전체 삭제")
def confirm_delete_dialog():
    st.markdown("모든 데이터가 영구히 삭제됩니다.")
    if st.button("✅ 네, 삭제합니다", type="primary", use_container_width=True):
        st.session_state.asset_data = {}
        if "user" in st.session_state:
            auto_save_tracker_data()
        st.session_state.selected_month = None
        st.rerun()

@st.dialog("🗑️ 선택한 월 삭제")
def delete_month_dialog(month_key):
    st.markdown(f"**{month_key[:4]}년 {int(month_key[5:])}월** 데이터를 정말 삭제하시겠습니까?")
    if st.button("✅ 네, 삭제합니다", type="primary", use_container_width=True):
        if month_key in st.session_state.asset_data:
            del st.session_state.asset_data[month_key]
            if "user" in st.session_state:
                auto_save_tracker_data()
            if st.session_state.selected_month == month_key:
                st.session_state.selected_month = None
            st.rerun()

st.markdown("# 💰 고라니 자산 트래커")

if st.session_state.get("show_success_msg"):
    st.toast(st.session_state.success_msg_text, icon="✅")
    st.session_state.show_success_msg = False

# 2. 클라우드 설정값 불러오기
tc = st.session_state.get("tracker_cfg", {})
KST = timezone(timedelta(hours=9))
now = datetime.now(KST)

if 'asset_data' not in st.session_state: st.session_state.asset_data = {}
if 'selected_month' not in st.session_state: st.session_state.selected_month = None

# 3. 화면 레이아웃
L, R = st.columns([1, 1])

with L:
    card_open()
    st.markdown("### 📋 데이터 입력")
    cy, cm = st.columns(2)
    with cy: in_year = st.number_input("년", 2000, 2100, tc.get("in_year", now.year))
    with cm: in_month = st.number_input("월", 1, 12, tc.get("in_month", now.month))
    raw = st.text_area("뱅크샐러드 데이터(상품명금액부터 ctrl+v)", height=180)
    
    if st.button("📊 데이터 추가", use_container_width=True):
        if raw.strip():
            items = parse_data(raw)
            if items:
                key = f"{int(in_year)}-{int(in_month):02d}"
                st.session_state.asset_data[key] = process_data(items)
                
                new_cfg = {"in_year": in_year, "in_month": in_month}
                st.session_state["tracker_cfg"] = new_cfg
                
                # 로그인 상태(user 정보가 있을 때)에만 동기화 시도
                if "user" in st.session_state:
                    auto_save_config("tracker", new_cfg)
                    auto_save_tracker_data() 
                
                st.session_state.selected_month = key
                st.session_state.show_success_msg = True
                st.session_state.success_msg_text = f"{in_year}년 {in_month}월 데이터가 저장되었습니다!"
                st.rerun()
    
    st.markdown("### 📅 입력된 월")
    sorted_keys = sorted(st.session_state.asset_data.keys())
    if sorted_keys:
        # ✅ 버튼을 작게 만들고 5열로 배치하여 오밀조밀하게 구성
        cols = st.columns(5)
        for i, key_val in enumerate(sorted_keys):
            is_selected = (key_val == st.session_state.selected_month)
            btn_type = "primary" if is_selected else "secondary"
            display_label = f"{key_val[2:4]}-{key_val[5:7]}"
            with cols[i % 5]:
                # ✅ use_container_width=True 제거하여 글자 크기에 딱 맞춤
                if st.button(display_label, key=f"hist_{key_val}", type=btn_type):
                    st.session_state.selected_month = key_val
                    st.rerun()
    else:
        st.caption("저장된 데이터가 없습니다.")
    card_close()

with R:
    card_open()
    st.markdown("### 🍩 종목별 비중")
    sel_key = st.session_state.selected_month
    if sel_key and sel_key in st.session_state.asset_data:
        month_data = st.session_state.asset_data[sel_key]
        total_assets = sum(month_data.values())
        
        c1, c2 = st.columns([3, 1])
        with c1: st.caption(f"**{sel_key[:4]}년 {int(sel_key[5:])}월** | 총 자산: {fmt_won(total_assets)}")
        with c2:
            if st.button("🗑️ 삭제", key=f"del_btn_{sel_key}", use_container_width=True):
                delete_month_dialog(sel_key)
        
        entries = sort_tags_by_super_group(list(month_data.items()))
        
        if entries:
            labels = [e[0] for e in entries]
            values = [e[1] for e in entries]
            colors_dict = assign_colors(labels)
            marker_colors = [colors_dict.get(l, '#CCCCCC') for l in labels]
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=0.5, 
                sort=False, 
                marker=dict(colors=marker_colors),
                textinfo='label+percent'
            )])
            fig_donut.update_layout(
                margin=dict(t=10, b=80, l=10, r=10), # b=80으로 하단 여백 확보
                height=350, # 높이를 250에서 350으로 늘려 범례 공간 확보
                showlegend=True,
                legend=dict(
                    orientation="h", 
                    yanchor="top", # 범례의 기준점을 위쪽으로 변경
                    y=-0.1,        # 차트 바로 아래에 위치하도록 조정
                    xanchor="center", 
                    x=0.5
                )
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
            with st.expander("자산 상세 내역 보기", expanded=True):
                for label, val in entries:
                    st.write(f"**{label}**: {fmt_won(val)} ({(val/total_assets*100):.1f}%)")
    else:
        st.info("좌측에서 조회할 월을 선택해주세요.")
    card_close()

# 하단 영역: 월별 자산 추이 누적 그래프
st.markdown("---")
st.markdown("### 📈 월별 자산 추이")

all_keys = sorted(st.session_state.asset_data.keys())
if len(all_keys) > 0:
    if len(all_keys) > 1:
        start_key, end_key = st.select_slider(
            "조회할 기간을 선택하세요",
            options=all_keys,
            value=(all_keys[0], all_keys[-1])
        )
        filtered_keys = [k for k in all_keys if start_key <= k <= end_key]
    else:
        filtered_keys = all_keys

    if filtered_keys:
        tag_totals = {}
        for k in filtered_keys:
            for tag, val in st.session_state.asset_data[k].items():
                tag_totals[tag] = tag_totals.get(tag, 0) + val
                
        sorted_entries = sort_tags_by_super_group(list(tag_totals.items()))
        tag_list = [e[0] for e in sorted_entries]
        
        datasets = []
        for tag in tag_list:
            datasets.append([st.session_state.asset_data[k].get(tag, 0) for k in filtered_keys])

        if tag_list and datasets:
            fig_trend = go.Figure()
            colors_dict = assign_colors(tag_list)
            
            for tag, data_values in reversed(list(zip(tag_list, datasets))):
                color = colors_dict.get(tag, '#CCCCCC')
                fill_color = color
                
                fig_trend.add_trace(go.Scatter(
                    x=[f"{k[2:4]}.{k[5:7]}" for k in filtered_keys], 
                    y=data_values,
                    mode='lines', 
                    name=tag,
                    stackgroup='one', 
                    line=dict(width=2, color=color),
                    fillcolor=fill_color
                ))
            
            fig_trend.update_layout(
                hovermode="x unified",
                margin=dict(t=20, b=20, l=10, r=10), 
                height=400,
                legend=dict(traceorder="reversed")
            )
            st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("데이터를 추가하면 월별 자산 추이 그래프가 나타납니다.")

# 하단 영역: 현재 포트폴리오 기준 가상 성과 분석
st.markdown("---")
st.markdown("## 📊 성과 분석")
st.caption("가장 최근 등록된 포트폴리오를 과거 시작일에 한 번에 매수해 그대로 보유했다고 가정합니다.")

all_perf_keys = sorted(st.session_state.asset_data.keys())
if all_perf_keys:
    latest_perf_key = all_perf_keys[-1]
    default_perf_start = default_start_date_for_latest(latest_perf_key, today=now.date())
    start_state_key = f"tracker_perf_start_{latest_perf_key}"
    if start_state_key not in st.session_state:
        st.session_state[start_state_key] = default_perf_start

    ctrl_cols = st.columns([1, 3])
    with ctrl_cols[0]:
        perf_start_date = st.date_input(
            "시작일",
            key=start_state_key,
            help="휴장일이면 이후 사용 가능한 첫 가격 데이터를 사용합니다.",
        )
    with ctrl_cols[1]:
        st.write("")
        st.caption(f"분석 기준 스냅샷: **{latest_perf_key}** · 기본값: 최근 스냅샷 기준 약 2년 전")

    with st.spinner("성과 분석용 가격 데이터를 불러오는 중입니다..."):
        perf = build_tracker_performance(st.session_state.asset_data, perf_start_date, today=now.date())

    if perf.chart.empty or perf.initial_capital <= 0:
        st.info("가격 데이터가 있는 자산이 부족해 성과 분석 그래프를 표시할 수 없습니다.")
        for warning in perf.warnings:
            st.caption(f"- {warning}")
        if perf.excluded_assets:
            st.caption("가격 데이터 없음/현금성 자산으로 성과 계산에서 제외된 항목: " + ", ".join(perf.excluded_assets))
    else:
        if perf.effective_start_date and perf.effective_start_date != perf.requested_start_date:
            st.caption(f"선택한 시작일의 가격이 없어 **{perf.effective_start_date.isoformat()}**부터 계산했습니다.")

        card_defs = [
            ("순투자원금", "initial_capital", "시작일 기준 역산 매수금액", "#334155", "#CBD5E1"),
            ("내 포트폴리오", "portfolio", fmt_pct(perf.cards.get("portfolio", {}).get("return")), "#0F766E", "#14B8A6"),
            ("KOSPI 투자 시", "kospi", fmt_pct(perf.cards.get("kospi", {}).get("return")), "#2563EB", "#3B82F6"),
            ("S&P 500 투자 시", "sp500", fmt_pct(perf.cards.get("sp500", {}).get("return")), "#EA580C", "#F97316"),
            ("QQQ 투자 시", "qqq", fmt_pct(perf.cards.get("qqq", {}).get("return")), "#DB2777", "#EC4899"),
        ]
        card_cols = st.columns(5)
        for col, (title, key, subtitle, color, border) in zip(card_cols, card_defs):
            with col:
                value = perf.cards.get(key, {}).get("value")
                render_performance_card(title, fmt_won(value or 0), subtitle, color=color, border_color=border)

        fig_perf = go.Figure()
        trace_specs = [
            ("portfolio", "포트폴리오", "#14B8A6", "solid", 3),
            ("initial_capital", "순투자원금", "#CBD5E1", "dash", 2),
            ("kospi", "KOSPI", "#3B82F6", "dot", 2),
            ("sp500", "S&P 500", "#F97316", "dot", 2),
            ("qqq", "QQQ", "#EC4899", "solid", 2),
        ]
        for key, label, color, dash, width in trace_specs:
            if key not in perf.chart.columns or perf.chart[key].dropna().empty:
                continue
            returns = perf.chart[key] / perf.initial_capital - 1.0
            fig_perf.add_trace(go.Scatter(
                x=perf.chart.index,
                y=perf.chart[key],
                mode="lines",
                name=label,
                line=dict(color=color, dash=dash, width=width),
                customdata=returns,
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,.0f}원<br>수익률: %{customdata:+.1%}<extra></extra>",
            ))
        fig_perf.update_layout(
            height=460,
            hovermode="x unified",
            margin=dict(t=25, b=30, l=10, r=10),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            yaxis=dict(title="KRW", tickformat=",.0f"),
            xaxis=dict(dtick="M1", tickformat="%y.%m"),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        if perf.excluded_assets:
            st.caption("가격 데이터 없음/현금성 자산으로 성과 계산에서 제외된 항목: " + ", ".join(perf.excluded_assets))
        if perf.warnings:
            with st.expander("성과 분석 데이터 안내", expanded=False):
                for warning in perf.warnings:
                    st.write(f"- {warning}")
        st.caption(
            "이 그래프는 가장 최근 등록된 포트폴리오를 기준으로, 선택한 시작일에 동일한 보유수량을 매수해 현재까지 보유했다고 가정한 가상 성과입니다. "
            "순투자원금은 실제 입금액이 아니라 현재 보유자산을 시작일 가격으로 역산한 금액입니다. USD 자산과 S&P 500/QQQ는 조회 가능한 USD/KRW 환율로 원화 환산합니다."
        )

        st.markdown("### 월별 수익/손실 추이")
        monthly = perf.monthly.copy()
        if monthly.empty:
            st.info("월별 손익을 계산하기에 충분한 월말 가격 데이터가 없습니다.")
        else:
            rolling_profit = float(monthly["profit"].sum())
            monthly_cols = st.columns([3, 1])
            with monthly_cols[1]:
                render_performance_card(
                    "최근 12개월 손익",
                    fmt_won(rolling_profit),
                    "월말 가상 평가액 변화 합계",
                    color="#EF4444" if rolling_profit >= 0 else "#3B82F6",
                    border_color="#E5E8EB",
                )
            with monthly_cols[0]:
                fig_monthly = go.Figure()
                bar_colors = ["#EF4444" if v >= 0 else "#3B82F6" for v in monthly["profit"]]
                fig_monthly.add_trace(go.Bar(
                    x=monthly["label"],
                    y=monthly["profit"],
                    name="수익" if rolling_profit >= 0 else "손익",
                    marker_color=bar_colors,
                    hovertemplate="%{x}<br>월별 손익: %{y:,.0f}원<extra></extra>",
                ))
                fig_monthly.add_trace(go.Scatter(
                    x=monthly["label"],
                    y=monthly["portfolio"],
                    mode="lines+markers",
                    name="총 자산",
                    yaxis="y2",
                    line=dict(color="#14B8A6", width=3),
                    hovertemplate="%{x}<br>가상 포트폴리오: %{y:,.0f}원<extra></extra>",
                ))
                fig_monthly.update_layout(
                    height=380,
                    hovermode="x unified",
                    margin=dict(t=20, b=30, l=10, r=10),
                    legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
                    yaxis=dict(title="월별 손익", tickformat=",.0f"),
                    yaxis2=dict(title="총 자산", overlaying="y", side="right", tickformat=",.0f"),
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            st.caption("월별 손익은 가상 포트폴리오의 월말 평가액 변화로 계산됩니다. 실제 추가 입금, 매수, 매도 내역은 반영하지 않습니다.")
else:
    st.info("데이터를 추가하면 성과 분석 섹션이 나타납니다.")
