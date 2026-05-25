import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timezone, timedelta
from logic.simulator import (
    SimConfig, YearPlan, assign_statuses, find_retire_index,
    simulate_deposits, apply_returns, get_real_balances,
    simulate_total_withdraw, simulate_tax_account_withdraw,
)
from ui.styles import TOSS_CSS
from core.sync import auto_save_config, _safe_uid
from core.firebase import load_data

# 1. 디자인 및 유틸리티 설정
st.markdown(TOSS_CSS, unsafe_allow_html=True)

def fmt_man(v): return f"{int(round(v)):,}"

def card_open():
    st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6; opacity:0.2; margin:20px 0;'>", unsafe_allow_html=True)

def card_close():
    pass

def metric(label, value, sub=None, kind=""):
    cls = f"toss-metric {kind}"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>', unsafe_allow_html=True)


def build_default_plan_records(start_year: int, sim_years: int):
    years = list(range(start_year, start_year + sim_years))
    new_df = pd.DataFrame({
        "년도": years,
        "월적립액(만원)": [300.0 if i <= 7 else 0.0 for i in range(len(years))],
        "ISA적립": [True if i <= 7 else False for i in range(len(years))],
        "연금저축적립": [True if i <= 7 else False for i in range(len(years))],
        "ISA연금이전": [False] * len(years),
    })
    return json.loads(new_df.to_json(orient="records"))


def is_valid_cached_plan(cached_plan, current_sim: int, current_start: int) -> bool:
    if not isinstance(cached_plan, list):
        return False
    if not cached_plan:
        return False
    if len(cached_plan) != current_sim:
        return False
    first = cached_plan[0]
    if not isinstance(first, dict):
        return False
    if "년도" not in first:
        return False
    if first["년도"] != current_start:
        return False
    return True

# 2. 강력한 새로고침 및 페이지 이동 방어 시스템
def save_to_firebase():
    cfg = {
        "start_year": st.session_state.start_year,
        "sim_years": st.session_state.sim_years,
        "return_rate": st.session_state.return_rate,
        "inflation_rate": st.session_state.inflation_rate,
        "init_isa": st.session_state.init_isa,
        "init_pension": st.session_state.init_pension,
        "init_general": st.session_state.init_general,
        "init_dividend": st.session_state.get("init_dividend", 0.0),
        "withdraw_rate": st.session_state.withdraw_rate,
        "withdraw_increase": st.session_state.withdraw_increase,
        "withdraw_delay": st.session_state.withdraw_delay,
        "plan_data": st.session_state.get("plan_data_cache", [])
    }
    # ★ 핵심: 파이어베이스 저장과 동시에 로컬 캐시(sim_cfg)도 동기화
    st.session_state["sim_cfg"] = cfg
    try:
        auto_save_config("sim", cfg)
    except Exception:
        pass

# 3. 세션 및 초기 데이터 로드
KST = timezone(timedelta(hours=9))
now = datetime.now(KST)

# (1) 앱 최초 구동 시 (sim_cfg가 아예 없을 때) 파이어베이스에서 로드
if "sim_cfg" not in st.session_state:
    if "user" in st.session_state:
        try:
            uid = _safe_uid(st.session_state["user"]["uid"])
            st.session_state["sim_cfg"] = load_data(uid, "sim_config") or {}
        except Exception as e:
            print(f"Firebase 로드 에러: {e}")
            st.session_state["sim_cfg"] = {}
    else:
        st.session_state["sim_cfg"] = {}

saved = st.session_state["sim_cfg"]

# (2) 페이지 이동으로 스트림릿 위젯 상태(key)가 증발한 경우, 저장된 캐시에서 강제 복구
if "start_year" not in st.session_state or "init_isa" not in st.session_state:
    st.session_state.start_year = int(saved.get("start_year", now.year))
    st.session_state.sim_years = int(saved.get("sim_years", 30))
    st.session_state.return_rate = float(saved.get("return_rate", 6.0))
    st.session_state.inflation_rate = float(saved.get("inflation_rate", 3.0))
    st.session_state.init_isa = float(saved.get("init_isa", 2000.0))
    st.session_state.init_pension = float(saved.get("init_pension", 11897.0))
    st.session_state.init_general = float(saved.get("init_general", 0.0))
    st.session_state.init_dividend = float(saved.get("init_dividend", 0.0))
    st.session_state.withdraw_rate = float(saved.get("withdraw_rate", 3.5))
    st.session_state.withdraw_increase = float(saved.get("withdraw_increase", 3.0))
    st.session_state.withdraw_delay = int(saved.get("withdraw_delay", 1))
    st.session_state.plan_data_cache = saved.get("plan_data", [])

# 구버전/손상 캐시가 남아 있어도 즉시 정상화
if not is_valid_cached_plan(
    st.session_state.get("plan_data_cache"),
    int(st.session_state.get("sim_years", 30)),
    int(st.session_state.get("start_year", now.year)),
):
    st.session_state.plan_data_cache = build_default_plan_records(
        int(st.session_state.get("start_year", now.year)),
        int(st.session_state.get("sim_years", 30)),
    )
    save_to_firebase()

# ----------------- UI 렌더링 -----------------

c_title, c_btn = st.columns([3, 1])
with c_title:
    st.markdown("# 📊 자산 시뮬레이터")
    st.caption("ISA·연금저축·일반위탁 적립과 인출까지 한 번에 시뮬레이션")

with c_btn:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    if st.button("💾 Firebase 즉시 저장", use_container_width=True, type="primary"):
        save_to_firebase()
        st.toast("✅ Firebase 클라우드 저장 완료!")

# 4. 기본 설정 카드 (모든 위젯은 위젯의 key를 통해 세션과 연동되며, 변경 시 save_to_firebase 실행)
card_open()
st.markdown("### ⛏️ 기본 설정")
c1, c2 = st.columns(2)
with c1:
    st.number_input("시작년도", 2020, 2100, key="start_year", on_change=save_to_firebase)
    st.number_input("시뮬레이션 기간 (년)", 1, 50, key="sim_years", on_change=save_to_firebase)
    st.number_input("예상 연간 수익률 (%)", 0.0, 100.0, step=0.1, key="return_rate", on_change=save_to_firebase)
    st.number_input("물가상승률 (%)", 0.0, 100.0, step=0.1, key="inflation_rate", on_change=save_to_firebase)
with c2:
    st.number_input("기존 ISA 잔고 (만원)", 0.0, 1e9, step=100.0, key="init_isa", on_change=save_to_firebase)
    st.number_input("기존 연금저축 잔고 (만원)", 0.0, 1e9, step=100.0, key="init_pension", on_change=save_to_firebase)
    st.number_input("추가 투입 예비금 (만원)", 0.0, 1e9, step=100.0, key="init_general", on_change=save_to_firebase)
    st.number_input("💸배당용 위탁잔고(만원)", 0.0, 1e9, step=100.0, key="init_dividend", on_change=save_to_firebase)

c3, c4, c5 = st.columns(3)
with c3: st.number_input("인출률 (%)", 0.0, 100.0, step=0.1, key="withdraw_rate", on_change=save_to_firebase)
with c4: st.number_input("인출금 연간 증액률 (%)", 0.0, 100.0, step=0.1, key="withdraw_increase", on_change=save_to_firebase)
with c5: st.number_input("🕐 인출 미룰 년수 (1~15)", 1, 15, key="withdraw_delay", on_change=save_to_firebase)
card_close()

# 5. 시뮬레이션 계산 로직
@st.cache_data(show_spinner=False)
def run_simulation(start_y, sim_y, init_i, init_p, init_g, ret_r, inf_r, w_rate, w_inc, w_delay, plan_records):
    cfg = SimConfig(
        start_year=int(start_y), sim_years=int(sim_y),
        init_isa=init_i, init_pension=init_p, init_general=init_g,
        return_rate=ret_r / 100, inflation_rate=inf_r / 100,
        withdraw_rate=w_rate / 100, withdraw_increase=w_inc / 100,
        withdraw_delay=int(w_delay),
    )
    plans = []
    for row in plan_records:
        plans.append(YearPlan(
            year=int(row["년도"]), 
            monthly=float(row["월적립액(만원)"] or 0),
            pension_check=bool(row["연금저축적립"]), 
            isa_check=bool(row["ISA적립"]), 
            isa_transfer=bool(row["ISA연금이전"])
        ))
    assign_statuses(plans)
    results = simulate_deposits(cfg, plans)
    apply_returns(cfg, results)
    real_data = get_real_balances(cfg, results)
    retire_idx = find_retire_index(plans)
    total_w = simulate_total_withdraw(cfg, results, retire_idx)
    plan = simulate_tax_account_withdraw(cfg, results, retire_idx)
    return cfg, results, real_data, total_w, plan, plans, retire_idx

# 6. 년도별 투자 계획
card_open()
st.markdown("### 📅 년도별 투자 계획")
current_start = st.session_state.start_year
current_sim = st.session_state.sim_years
cached_plan = st.session_state.plan_data_cache

if not is_valid_cached_plan(cached_plan, current_sim, current_start):
    st.session_state.plan_data_cache = build_default_plan_records(current_start, current_sim)
    save_to_firebase()

df_for_edit = pd.DataFrame(st.session_state.plan_data_cache)

# 💡 column_order 속성을 추가하여 열 순서를 고정했습니다.
edited_df = st.data_editor(df_for_edit, hide_index=True, use_container_width=True, num_rows="fixed",
    column_order=["년도", "월적립액(만원)", "ISA적립", "연금저축적립", "ISA연금이전"],
    column_config={
        "년도": st.column_config.NumberColumn("년도", width="small", disabled=True, format="%d"),
        "월적립액(만원)": st.column_config.NumberColumn("월적립(만)", width="small", min_value=0, step=10, format="%.0f"),
        "ISA적립": st.column_config.CheckboxColumn("ISA", width="small"),
        "연금저축적립": st.column_config.CheckboxColumn("연금", width="small"),
        "ISA연금이전": st.column_config.CheckboxColumn("ISA이전", width="small"),
    }
)

edited_records = json.loads(edited_df.to_json(orient="records"))
# 데이터 에디터 변환 과정에서 년도 컬럼 누락/손상이 생기면 즉시 재계산
if not is_valid_cached_plan(edited_records, current_sim, current_start):
    edited_records = build_default_plan_records(current_start, current_sim)
if st.session_state.plan_data_cache != edited_records:
    st.session_state.plan_data_cache = edited_records
    save_to_firebase()
    st.rerun()
card_close()

# 7. 결과 계산 및 차트 출력
cfg, results, real_data, total_w, plan_out, plans_obj, retire_idx = run_simulation(
    st.session_state.start_year, st.session_state.sim_years, 
    st.session_state.init_isa, st.session_state.init_pension, st.session_state.init_general,
    st.session_state.return_rate, st.session_state.inflation_rate, 
    st.session_state.withdraw_rate, st.session_state.withdraw_increase, st.session_state.withdraw_delay,
    st.session_state.plan_data_cache
)

# 데이터 전처리
plan_rows_dict = {pr.year: pr for pr in plan_out.rows} if plan_out else {}
div_records = []
tax_bal_nom, tax_bal_real = [], []
div_bal_nom, div_bal_real = [], []
tax_m_net_l, tax_m_real_l = [], []
div_m_net_l, div_m_real_l = [], []
tot_m_net_l, tot_m_real_l = [], []

current_div_bal = st.session_state.init_dividend
ret_r, inf_r = st.session_state.return_rate/100, st.session_state.inflation_rate/100
wd_r, start_y = st.session_state.withdraw_rate/100, st.session_state.start_year

for r in results:
    y = r.year
    is_wd = "인출" in str(r.status)
    growth = current_div_bal * ret_r
    after_g = current_div_bal + growth
    
    if is_wd:
        gross_div = current_div_bal * wd_r
        current_div_bal = max(0.0, after_g - gross_div)
    else:
        gross_div = 0.0
        current_div_bal = after_g
    
    net_div = gross_div * 0.85
    discount = (1 + inf_r) ** (y - start_y)
    r_div_bal, r_net_div = current_div_bal/discount, net_div/discount
    dm_net, dm_real = net_div/12, r_net_div/12
    
    div_bal_nom.append(current_div_bal)
    div_bal_real.append(r_div_bal)
    div_m_net_l.append(dm_net)
    div_m_real_l.append(dm_real)

    pr = plan_rows_dict.get(y)
    if pr:
        tm_net = 0.0 if pr.is_delay else pr.monthly_net
        tm_real = 0.0 if pr.is_delay else pr.monthly_net_real
        t_nom = pr.isa_balance + pr.pension_balance
    else:
        tm_net, tm_real = 0.0, 0.0
        t_nom = r.isa_nominal + r.pension_nominal
    
    tax_bal_nom.append(t_nom)
    tax_bal_real.append(t_nom / discount)
    tax_m_net_l.append(tm_net)
    tax_m_real_l.append(tm_real)
    tot_m_net_l.append(dm_net + tm_net)
    tot_m_real_l.append(dm_real + tm_real)

    div_records.append({
        "년도": y, "배당용 위탁잔고(명목)": fmt_man(current_div_bal), "배당용 위탁잔고(실질)": fmt_man(r_div_bal),
        "세후 연간 배당금(명목)": fmt_man(net_div), "세후 연간 배당금(실질)": fmt_man(r_net_div),
        "세후 월별 배당금(명목)": fmt_man(dm_net), "세후 월별 배당금(실질)": fmt_man(dm_real),
        "월배당합(절세+위탁)(명목)": fmt_man(dm_net + tm_net),
        "월배당합(절세+위탁)(실질)": fmt_man(dm_real + tm_real)
    })

total_bal_nom = [t + d for t, d in zip(tax_bal_nom, div_bal_nom)]
total_bal_real = [t + d for t, d in zip(tax_bal_real, div_bal_real)]
HOVER_TEMPLATE = "%{data.name} (%{x}, %{y:,.0f}만)<extra></extra>"

st.markdown("---")
st.markdown("### 📊 시뮬레이션 결과")

# CSS 스타일 적용 (6개 항목이 한 줄에 겹치지 않도록 강제 조정)
st.markdown("""
<style>
[data-testid="column"] .toss-metric { padding: 10px 8px !important; border-radius: 12px; background: #f9fafb; border: 1px solid #f2f4f6; }
[data-testid="column"] .toss-metric .label { font-size: 11px !important; color: #6b7684; margin-bottom: 4px; word-break: keep-all; letter-spacing: -0.3px; }
[data-testid="column"] .toss-metric .value { font-size: 15px !important; font-weight: 700; color: #333d4b; word-break: keep-all; letter-spacing: -0.5px; }
[data-testid="column"] .toss-metric .sub { font-size: 10px !important; color: #8b95a1; }
</style>
""", unsafe_allow_html=True)

# 💡 6개 지표 출력 부분
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: metric("최종 명목 잔고(인출X)", f"{fmt_man(results[-1].total_nominal)} 만원", kind="accent")
with c2: metric("최종 실질 잔고(인출X)", f"{fmt_man(real_data[-1]['total_real'])} 만원", kind="success")
with c3: metric("합산 명목 잔고(절세+배당위탁)", f"{fmt_man(total_bal_nom[-1])} 만원", kind="accent")
with c4: metric("합산 실질 잔고(절세+배당위탁)", f"{fmt_man(total_bal_real[-1])} 만원", kind="success")
with c5: metric("은퇴년도", f"{plans_obj[retire_idx].year if retire_idx >= 0 else '-'}년", sub=f"인출 {plan_out.actual_start_year if plan_out else '-'}년~")
with c6: metric("연금저축 한도", f"{fmt_man(cfg.init_pension + results[-1].total_pension_deposit)} 만원", kind="warn")

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 잔고 추이 차트", "📈 배당금 추이 차트", "💰적립 현황", "🏧절세계좌인출(원금만)", "💸위탁계좌(배당용) 잔고"])

with tab1:
    fig = go.Figure()
    years = [r.year for r in results]
    fig.add_trace(go.Scatter(x=years, y=tax_bal_nom, name="절세전체잔고(명목)", line=dict(color="#3182F6", width=3), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=tax_bal_real, name="절세전체잔고(실질)", line=dict(color="#3182F6", width=3, dash="dot"), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=div_bal_nom, name="배당용위탁잔고(명목)", line=dict(color="#00875A", width=3), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=div_bal_real, name="배당용위탁잔고(실질)", line=dict(color="#00875A", width=3, dash="dot"), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=total_bal_nom, name="합산 명목 잔고(절세+위탁)", line=dict(color="#8B5CF6", width=3), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=total_bal_real, name="합산 실질 잔고(절세+위탁)", line=dict(color="#8B5CF6", width=3, dash="dot"), hovertemplate=HOVER_TEMPLATE))
    if retire_idx >= 0: fig.add_vline(x=plans_obj[retire_idx].year, line_dash="dash", line_color="#FF8B00", annotation_text="은퇴")
    fig.update_layout(height=450, hovermode="x unified", yaxis_title="만원", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=tax_m_net_l, name="절세 월인출금(명목)", line=dict(color="#3182F6", width=2), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=tax_m_real_l, name="절세 월인출금(실질)", line=dict(color="#3182F6", width=2, dash="dot"), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=div_m_net_l, name="위탁 월배당금(명목)", line=dict(color="#00875A", width=2), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=div_m_real_l, name="위탁 월배당금(실질)", line=dict(color="#00875A", width=2, dash="dot"), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=tot_m_net_l, name="합산 월수익(명목)", line=dict(color="#8B5CF6", width=3), hovertemplate=HOVER_TEMPLATE))
    fig.add_trace(go.Scatter(x=years, y=tot_m_real_l, name="합산 월수익(실질)", line=dict(color="#8B5CF6", width=3, dash="dot"), hovertemplate=HOVER_TEMPLATE))
    if retire_idx >= 0: fig.add_vline(x=plans_obj[retire_idx].year, line_dash="dash", line_color="#FF8B00", annotation_text="은퇴")
    fig.update_layout(height=450, hovermode="x unified", yaxis_title="만원", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.dataframe(pd.DataFrame([{"년도": r.year, "상태": r.status, "연금적립": fmt_man(r.pension_deposit), "연금잔고": fmt_man(r.pension_balance), "ISA적립": fmt_man(r.isa_deposit), "ISA잔고": fmt_man(r.isa_balance), "적립액from예비금": fmt_man(r.general_deposit), "예비금 잔고": fmt_man(r.general_balance), "전체잔고": fmt_man(r.total_balance)} for r in results]), use_container_width=True, hide_index=True)

with tab4:
    if plan_out:
        st.dataframe(pd.DataFrame([{"년도": r.year, "구분": r.period_label, "ISA잔고(명목)": fmt_man(r.isa_balance), "연금잔고(명목)": fmt_man(r.pension_balance), "월수령(명목,실질)": "대기중" if r.is_delay else f"{fmt_man(r.monthly_net)} 만원 ({fmt_man(r.monthly_net_real)} 만원)"} for r in plan_out.rows]), use_container_width=True, hide_index=True)

with tab5:
    st.dataframe(pd.DataFrame(div_records), use_container_width=True, hide_index=True)
