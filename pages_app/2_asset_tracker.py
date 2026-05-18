import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from logic.tracker import parse_data, process_data
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
