import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from logic.simulator import (
    SimConfig, YearPlan, assign_statuses, find_retire_index,
    simulate_deposits, apply_returns, get_real_balances,
    simulate_total_withdraw, simulate_tax_account_withdraw,
)
from ui.styles import TOSS_CSS
from core.sync import auto_save_config

# 1. 디자인 및 유틸리티 설정
st.markdown(TOSS_CSS, unsafe_allow_html=True)

def fmt_man(v): return f"{int(round(v)):,}"

# ✅ [수정] 하얀 상자 대신 깔끔한 가로선으로 변경
def card_open():
    st.markdown("<hr style='border:0; border-top:1px solid #F2F4F6; opacity:0.2; margin:20px 0;'>", unsafe_allow_html=True)

def card_close():
    pass # 이제 상자를 닫을 필요가 없으므로 아무것도 하지 않습니다.

def metric(label, value, sub=None, kind=""):
    cls = f"toss-metric {kind}"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>', unsafe_allow_html=True)

st.markdown("# 📊 자산 시뮬레이터")
st.caption("ISA·연금저축·일반위탁 적립과 인출까지 한 번에 시뮬레이션")

# 2. 클라우드 설정값 및 시간 설정
sc = st.session_state.get("sim_cfg", {})
KST = timezone(timedelta(hours=9))
now = datetime.now(KST)

# 3. 기본 설정 카드 (자동 로드 적용)
card_open()
st.markdown("### ⛏️ 기본 설정")
c1, c2 = st.columns(2)
with c1:
    start_year = st.number_input("시작년도", 2020, 2100, sc.get("start_year", now.year))
    sim_years = st.number_input("시뮬레이션 기간 (년)", 1, 50, sc.get("sim_years", 30))
    return_rate = st.number_input("예상 연간 수익률 (%)", 0.0, 100.0, sc.get("return_rate", 6.0), 0.1)
    inflation_rate = st.number_input("물가상승률 (%)", 0.0, 100.0, sc.get("inflation_rate", 3.0), 0.1)
with c2:
    init_isa = st.number_input("기존 ISA 잔고 (만원)", 0.0, 1e9, sc.get("init_isa", 2000.0), 100.0)
    init_pension = st.number_input("기존 연금저축 잔고 (만원)", 0.0, 1e9, sc.get("init_pension", 11897.0), 100.0)
    init_general = st.number_input("기존 일반위탁 잔고 (만원)", 0.0, 1e9, sc.get("init_general", 0.0), 100.0)

c3, c4, c5 = st.columns(3)
with c3: withdraw_rate = st.number_input("인출률 (%, 2051년~)", 0.0, 100.0, sc.get("withdraw_rate", 3.5), 0.1)
with c4: withdraw_increase = st.number_input("인출금 연간 증액률 (%)", 0.0, 100.0, sc.get("withdraw_increase", 3.0), 0.1)
with c5: withdraw_delay = st.number_input("🕐 인출 미룰 년수 (1~15)", 1, 15, sc.get("withdraw_delay", 1), 1)
card_close()

cfg = SimConfig(
    start_year=int(start_year), sim_years=int(sim_years),
    init_isa=init_isa, init_pension=init_pension, init_general=init_general,
    return_rate=return_rate / 100, inflation_rate=inflation_rate / 100,
    withdraw_rate=withdraw_rate / 100, withdraw_increase=withdraw_increase / 100,
    withdraw_delay=int(withdraw_delay),
)

# 4. 년도별 투자 계획
card_open()
st.markdown("### 📅 년도별 투자 계획")

plan_key = f"plan_df_{cfg.start_year}_{cfg.sim_years}"

if plan_key not in st.session_state:
    saved_plan = sc.get("plan_data")
    if saved_plan:
        st.session_state[plan_key] = pd.DataFrame(saved_plan)
    else:
        years = list(range(cfg.start_year, cfg.start_year + cfg.sim_years))
        st.session_state[plan_key] = pd.DataFrame({
            "년도": years,
            "월적립액(만원)": [300.0 if i <= 7 else 0.0 for i in range(len(years))],
            "연금저축적립": [True if i <= 7 else False for i in range(len(years))],
            "ISA적립": [True if i <= 7 else False for i in range(len(years))],
            "ISA연금이전": [False] * len(years),
        })

edited_df = st.data_editor(
    st.session_state[plan_key],
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "년도": st.column_config.NumberColumn(disabled=True, format="%d"),
        "월적립액(만원)": st.column_config.NumberColumn(min_value=0, step=10, format="%.0f"),
        "연금저축적립": st.column_config.CheckboxColumn(),
        "ISA적립": st.column_config.CheckboxColumn(),
        "ISA연금이전": st.column_config.CheckboxColumn(),
    },
    key=f"editor_widget_{plan_key}" 
)

if not st.session_state[plan_key].equals(edited_df):
    st.session_state[plan_key] = edited_df
    st.rerun()

if st.button("🚀 시뮬레이션 실행", use_container_width=True):
    current_cfg = {
        "start_year": start_year, "sim_years": sim_years, "return_rate": return_rate,
        "inflation_rate": inflation_rate, "init_isa": init_isa, "init_pension": init_pension,
        "init_general": init_general, "withdraw_rate": withdraw_rate, 
        "withdraw_increase": withdraw_increase, "withdraw_delay": withdraw_delay,
        "plan_data": st.session_state[plan_key].to_dict('records')
    }
    st.session_state["sim_cfg"] = current_cfg
    auto_save_config("sim", current_cfg)
    st.session_state["run_sim"] = True

card_close()

# 5. 시뮬레이션 결과 출력
if st.session_state.get("run_sim"):
    plans = []
    for _, row in st.session_state[plan_key].iterrows():
        plans.append(YearPlan(year=int(row["년도"]), monthly=float(row["월적립액(만원)"] or 0),
                              pension_check=bool(row["연금저축적립"]), isa_check=bool(row["ISA적립"]), isa_transfer=bool(row["ISA연금이전"])))
    assign_statuses(plans)
    results = simulate_deposits(cfg, plans)
    apply_returns(cfg, results)
    real_data = get_real_balances(cfg, results)
    retire_idx = find_retire_index(plans)
    total_w = simulate_total_withdraw(cfg, results, retire_idx)
    plan = simulate_tax_account_withdraw(cfg, results, retire_idx)

    last = results[-1]
    retire_year = plans[retire_idx].year if retire_idx >= 0 else None
    
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric("최종 명목 잔고", f"{fmt_man(last.total_nominal)} 만원", kind="accent")
    with c2: metric("최종 실질 잔고", f"{fmt_man(real_data[-1]['total_real'])} 만원", kind="success")
    with c3: metric("은퇴년도", f"{retire_year}년" if retire_year else "—", sub=f"실제 인출 {plan.actual_start_year}년~" if plan else None)
    with c4: metric("연금저축 한도", f"{fmt_man(cfg.init_pension + last.total_pension_deposit)} 만원", sub=f"적립 {fmt_man(last.total_pension_deposit)} 만원", kind="warn")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 추이 차트", "💰 적립 현황", "📊 수익률 반영", "💸 전체자산 인출", "🏧 절세계좌 인출"])
    
    with tab1:
        years = [r.year for r in results]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=[r.total_nominal for r in results], name="명목 전체", mode="lines+markers", line=dict(color="#3182F6", width=3)))
        fig.add_trace(go.Scatter(x=years, y=[d["total_real"] for d in real_data], name="실질 전체", mode="lines+markers", line=dict(color="#00875A", width=3, dash="dot")))
        if retire_year: fig.add_vline(x=retire_year, line_color="#FF8B00", line_dash="dash", annotation_text="은퇴")
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="만원")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df = pd.DataFrame([{"년도": r.year, "상태": r.status, "연금적립": fmt_man(r.pension_deposit), "연금잔고": fmt_man(r.pension_balance), "ISA적립": fmt_man(r.isa_deposit), "ISA잔고": fmt_man(r.isa_balance), "일반적립": fmt_man(r.general_deposit), "일반잔고": fmt_man(r.general_balance), "전체잔고": fmt_man(r.total_balance)} for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab3:
        df = pd.DataFrame([{"년도": r.year, "연금명목": fmt_man(r.pension_nominal), "연금실질": fmt_man(rd["pension_real"]), "ISA명목": fmt_man(r.isa_nominal), "ISA실질": fmt_man(rd["isa_real"]), "일반명목": fmt_man(r.general_nominal), "일반실질": fmt_man(rd["general_real"]), "전체명목": fmt_man(r.total_nominal), "전체실질": fmt_man(rd["total_real"])} for r, rd in zip(results, real_data)])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab4:
        df = pd.DataFrame([{"년도": w.year, "전체명목": fmt_man(w.total_nominal), "연간인출": fmt_man(w.withdraw) if w.is_withdraw else "—", "월환산": fmt_man(w.monthly) if w.is_withdraw else "—", "인출후잔고": fmt_man(w.after_balance) if w.is_withdraw else "—", "실질가치 인출": fmt_man(w.real_withdraw) if w.is_withdraw else "—"} for w in total_w])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab5:
        if plan:
            st.markdown("#### 📋 인출 계획")
            df = pd.DataFrame([{"년도": r.year, "구분": r.period_label, "ISA세전": "—" if r.is_delay else fmt_man(r.isa_gross), "ISA세후": "—" if r.is_delay else fmt_man(r.isa_net), "ISA잔고": fmt_man(r.isa_balance), "연금세전": "—" if r.is_delay else fmt_man(r.pension_gross), "연금세후": "—" if r.is_delay else fmt_man(r.pension_net), "연금잔고": fmt_man(r.pension_balance), "월수령": "대기중" if r.is_delay else f"{fmt_man(r.monthly_net)} 만원 (현재가치 {fmt_man(r.monthly_net_real)} 만원)"} for r in plan.rows])
            st.dataframe(df, use_container_width=True, hide_index=True)