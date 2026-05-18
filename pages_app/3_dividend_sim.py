import streamlit as st
import pandas as pd
import yfinance as yf
# from ui.styles import TOSS_CSS # 실제 환경에 맞게 주석 해제하세요

# CSS 적용
# st.markdown(TOSS_CSS, unsafe_allow_html=True) 

st.markdown("# 💸 양도세치기 배당시뮬")

# ==========================================
# 1. 전략 설정 영역
# ==========================================
st.markdown("### ⚙️ 전략 설정")

col1, col2, col3 = st.columns(3)

with col1:
    ticker_input = st.text_input("티커 (예: SCHD, ARCC)", "ARCC")
    ticker = ticker_input.upper() 
    invest_capital = st.number_input("투자자금 (달러)", min_value=0, value=10000, step=1000)

with col2:
    buy_type = st.selectbox("매수가 기준", ["D-1 종가", "D-1 시가", "D-2 종가", "D-2 시가"])
    sell_window = st.number_input("매도허용기간 (N거래일)예:0,5", min_value=0, max_value=600, value=0)

with col3:
    # 🚨 수정 포인트: 세율을 입력받는 number_input 추가
    tax_rate = st.number_input("배당소득세율 (%)", min_value=1.0, max_value=99.0, value=15.0, step=1.0)
    recent_5y_only = st.checkbox("최근 5년 데이터만 보기", value=False)
    run_btn = st.button("백테스트 실행!", use_container_width=True)

st.divider()

# 🛠️ 데이터 캐싱
@st.cache_data(ttl=3600)
def fetch_data(ticker_name):
    data = yf.Ticker(ticker_name)
    df = data.history(period="max", auto_adjust=False)
    
    if not df.empty:
        df.index = df.index.tz_localize(None).normalize()
        
    if 'Dividends' in df.columns:
        divs = df[df['Dividends'] > 0]['Dividends']
    else:
        divs = pd.Series(dtype=float)
        
    return df, divs

# ==========================================
# 🚨 핵심 수정 포인트: 세션 상태(session_state)를 이용해 실행 상태 기억
# ==========================================
if "run_backtest" not in st.session_state:
    st.session_state.run_backtest = False

# 버튼이 눌렸을 때 상태를 True로 변경
if run_btn:
    st.session_state.run_backtest = True

# 버튼 변수 대신 세션 상태로 백테스트 진입
if st.session_state.run_backtest:
    with st.spinner(f'{ticker} 데이터를 가져오는 중입니다...'):
        df, divs = fetch_data(ticker)
        
        if df.empty:
            st.error(f"{ticker}의 주가 데이터를 찾을 수 없습니다.")
        else:
            if recent_5y_only and not divs.empty:
                cutoff_date = pd.Timestamp.now().normalize() - pd.DateOffset(years=5)
                divs = divs[divs.index >= cutoff_date]
            
            results = []
            
            for ex_date, div_amount in divs.items():
                try:
                    idx = df.index.get_loc(ex_date)
                except KeyError:
                    continue 
                
                if idx < 2 or idx + sell_window >= len(df):
                    continue
                
                # 매수가 결정
                if buy_type == "D-1 종가":
                    buy_price = df.iloc[idx-1]['Close']
                elif buy_type == "D-1 시가":
                    buy_price = df.iloc[idx-1]['Open']
                elif buy_type == "D-2 종가":
                    buy_price = df.iloc[idx-2]['Close']
                else: 
                    buy_price = df.iloc[idx-2]['Open']
                    
                # 🚨 수정 포인트: 고정된 0.85 대신 입력받은 tax_rate를 적용하여 세후 배당금 계산
                after_tax_div = div_amount * (1 - (tax_rate / 100))
                bep = buy_price - after_tax_div
                
                window_data = df.iloc[idx : idx + sell_window + 1]
                max_high = window_data['High'].max()
                is_success = max_high >= bep
                
                # === 원금 회복 날짜 계산 (초기값 설정) ===
                recovery_date = "-"
                recovery_days_trading = "-"
                recovery_days_calendar = "-"
                
                if is_success:
                    profit_pct = (after_tax_div / buy_price) * 100
                else:
                    future_data = df.iloc[idx:] 
                    recovery_series = future_data[future_data['High'] >= bep]
                    
                    if not recovery_series.empty:
                        recovery_dt = recovery_series.index[0]
                        recovery_date = recovery_dt.strftime("%Y-%m-%d")
                        recovery_days_trading = f"{(df.index.get_loc(recovery_dt) - idx)}거래일"
                        recovery_days_calendar = f"{(recovery_dt - ex_date).days}일"
                    else:
                        # 🚨 핵심 수정 포인트: 원금 회복 불가 시 표기 변경
                        recovery_date = "회복불가"
                        recovery_days_trading = "회복불가"
                        recovery_days_calendar = "회복불가"
                    
                    sell_price = window_data.iloc[-1]['Close']
                    profit_pct = ((sell_price + after_tax_div - buy_price) / buy_price) * 100

                results.append({
                    "배당락일": ex_date.strftime("%Y-%m-%d"),
                    "매수가": round(buy_price, 2),
                    "세후배당금": round(after_tax_div, 4),
                    "손익분기점": round(bep, 4),
                    "성공여부": "성공" if is_success else "실패",
                    "수익률(%)": round(profit_pct, 2),
                    "원금 회복 날짜": recovery_date,
                    "소요 기간(거래일)": recovery_days_trading,
                    "소요 기간(달력)": recovery_days_calendar
                })
                
            res_df = pd.DataFrame(results)
            
            if len(res_df) > 0:
                st.success(f"총 {len(res_df)}회의 과거 배당 이벤트 분석 완료! (적용 세율: {tax_rate}%)")
                st.info(f"📅 백테스트 기간: {res_df['배당락일'].iloc[0]} ~ {res_df['배당락일'].iloc[-1]}")
                
                success_rate = (res_df['성공여부'] == '성공').mean() * 100
                avg_profit = res_df[res_df['성공여부'] == '성공']['수익률(%)'].mean()
                avg_loss = res_df[res_df['성공여부'] == '실패']['수익률(%)'].mean()
                expected_return = res_df['수익률(%)'].mean()
                
                tax_saving = (avg_profit / 100) * invest_capital * 0.22 if pd.notna(avg_profit) else 0
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("전략 승률", f"{success_rate:.1f}%")
                col2.metric("성공 평균수익률", f"{avg_profit:.2f}%" if pd.notna(avg_profit) else "0%")
                col3.metric("실패 평균손실률", f"{avg_loss:.2f}%" if pd.notna(avg_loss) else "0%")
                col4.metric("손익비", f"{abs(avg_profit/avg_loss):.2f}" if (pd.notna(avg_loss) and avg_loss != 0) else "∞")
                col5.metric("1회 기대수익률", f"{expected_return:.2f}%")
                col6.metric("1회 절세예상액", f"${tax_saving:.2f}")
                
                st.write("### 📈 수익률 분포 그래프")
                st.scatter_chart(res_df, x='배당락일', y='수익률(%)', color='성공여부')

                st.write("### 📊 회차별 상세 결과 (실패 시 원금 회복 정보 포함)")
                display_df = res_df[["배당락일", "매수가", "세후배당금", "손익분기점", "성공여부", "수익률(%)", "원금 회복 날짜", "소요 기간(거래일)", "소요 기간(달력)"]]
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("분석할 수 있는 데이터가 없습니다.")
