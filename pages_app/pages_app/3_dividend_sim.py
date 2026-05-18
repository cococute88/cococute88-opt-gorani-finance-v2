import streamlit as st
import yfinance as yf
import pandas as pd

st.title("💵 양도세치기 배당시뮬")

# 🛠️ 핵심 마법: 야후 파이낸스 데이터 기억(캐싱)
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

# 1. 왼쪽 사이드바(메뉴) 설정
st.sidebar.header("⚙️ 전략 설정")
ticker_input = st.sidebar.text_input("티커 (예: SCHD, ARCC)", "ARCC")
ticker = ticker_input.upper() 
invest_capital = st.sidebar.number_input("투자자금 (달러)", min_value=0, value=10000, step=1000)
buy_type = st.sidebar.selectbox("매수가 기준", ["D-1 종가", "D-1 시가", "D-2 종가", "D-2 시가"])
sell_window = st.sidebar.number_input("매도허용기간 (N거래일)예:0,5", min_value=0, max_value=600, value=0)
recent_5y_only = st.sidebar.checkbox("최근 5년 데이터만 보기", value=False)

# 2. 백테스트 실행
if st.sidebar.button("백테스트 실행!"):
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
                    
                after_tax_div = div_amount * 0.85
                bep = buy_price - after_tax_div
                
                # 설정된 매도허용기간 내 결과 분석
                window_data = df.iloc[idx : idx + sell_window + 1]
                max_high = window_data['High'].max()
                is_success = max_high >= bep
                
                # === 원금 회복 날짜 및 소요 기간 계산 로직 추가 ===
                recovery_date = "-"
                recovery_days_trading = "-"
                recovery_days_calendar = "-"
                
                if is_success:
                    profit_pct = (after_tax_div / buy_price) * 100
                else:
                    # 실패 시: 매도기간 이후 데이터에서 BEP를 회복하는 첫 날을 찾음
                    future_data = df.iloc[idx:] 
                    # 장중 고가가 BEP 이상이 되는 날들 중 가장 빠른 날
                    recovery_series = future_data[future_data['High'] >= bep]
                    
                    if not recovery_series.empty:
                        recovery_dt = recovery_series.index[0]
                        recovery_date = recovery_dt.strftime("%Y-%m-%d")
                        
                        # 소요 기간 계산
                        # 1. 거래일 기준 (배당락일부터 회복일까지의 행 개수)
                        recovery_days_trading = f"{(df.index.get_loc(recovery_dt) - idx)}거래일"
                        # 2. 달력 기준 (단순 날짜 차이)
                        recovery_days_calendar = f"{(recovery_dt - ex_date).days}일"
                    
                    sell_price = window_data.iloc[-1]['Close']
                    profit_pct = ((sell_price + after_tax_div - buy_price) / buy_price) * 100
                # ===============================================

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
                st.success(f"총 {len(res_df)}회의 과거 배당 이벤트 분석 완료!")
                st.info(f"📅 백테스트 기간: {res_df['배당락일'].iloc[0]} ~ {res_df['배당락일'].iloc[-1]}")
                
                # 지표 계산
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
                # 표의 가독성을 위해 열 순서 재배치
                display_df = res_df[["배당락일", "매수가", "세후배당금", "손익분기점", "성공여부", "수익률(%)", "원금 회복 날짜", "소요 기간(거래일)", "소요 기간(달력)"]]
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("분석할 수 있는 데이터가 없습니다.")