import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from logic.tracker import parse_data, process_data, get_sorted_entries, assign_colors, aggregate_for_trend
from ui.styles import TOSS_CSS
from core.sync import auto_save_tracker_data, auto_save_config # 💡 로봇 추가

# 1. 디자인 및 유틸리티 설정
st.markdown(TOSS_CSS, unsafe_allow_html=True)

def fmt_won(v): return f"{int(round(v)):,}원"

# ✅ [수정] 하얀 상자 대신 다크모드에 최적화된 연한 가로선으로 변경
def card_open():
    st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6; opacity:0.2; margin:20px 0;'>", unsafe_allow_html=True)

def card_close():
    # 이제 상자를 닫을 필요가 없으므로 아무것도 하지 않습니다.
    pass

@st.dialog("⚠️ 정말 삭제하시겠습니까?")
def confirm_delete_dialog():
    st.markdown("모든 데이터가 영구히 삭제됩니다.")
    if st.button("✅ 네, 삭제합니다", type="primary", use_container_width=True):
        st.session_state.asset_data = {}
        auto_save_tracker_data()
        st.session_state.selected_month = None
        st.rerun()

st.markdown("# 💰 자산 트래커")

# 2. 클라우드 설정값 불러오기
tc = st.session_state.get("tracker_cfg", {})
KST = timezone(timedelta(hours=9))
now = datetime.now(KST)

# 3. 화면 레이아웃
L, R = st.columns([1, 1])

with L:
    card_open() # 👈 이제 여기서 깔끔한 가로선이 그려집니다.
    st.markdown("### 📋 데이터 입력")
    cy, cm = st.columns(2)
    with cy: in_year = st.number_input("년", 2000, 2100, tc.get("in_year", now.year))
    with cm: in_month = st.number_input("월", 1, 12, tc.get("in_month", now.month))
    raw = st.text_area("뱅크샐러드 데이터", height=180)
    
    if st.button("📊 데이터 추가", use_container_width=True):
        if raw.strip():
            items = parse_data(raw)
            if items:
                key = f"{int(in_year)}-{int(in_month):02d}"
                st.session_state.asset_data[key] = process_data(items)
                
                # 💡 데이터 추가 시 설정값(년, 월)도 같이 저장!
                new_cfg = {"in_year": in_year, "in_month": in_month}
                st.session_state["tracker_cfg"] = new_cfg
                auto_save_config("tracker", new_cfg)
                auto_save_tracker_data() 
                
                st.session_state.selected_month = key
                st.rerun()
    
    card_close() # 👈 아무 일도 일어나지 않고 안전하게 종료됩니다.

# (이후 차트 및 하단 로직은 기존 선생님의 코드가 그대로 이어집니다)