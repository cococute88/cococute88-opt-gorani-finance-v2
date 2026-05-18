# NAVIGATION MAP

## 기존 페이지 구조
- `pages_app/1_asset_simulator.py` → 자산 시뮬레이터
- `pages_app/2_asset_tracker.py` → 자산 트래커
- `pages_app/3_dividend_sim.py` → 양도세치기 배당시뮬
- `pages_app/4_conversion_analysis.py` → 매도전환계산기
- `modules/dividend_calendar.py` 함수 렌더 기반 → 배당 캘린더

## 기존 `st.navigation` 매핑
- 그룹: `🧸 자산 및 배당 관리`
- 5개 항목을 사이드바 내 네비게이션으로 제공

## 변경 후 라우팅 구조
- 상단: `streamlit-option-menu` 가로 메뉴(빠른 화면 이동)
- 본 라우팅: 기존 `st.navigation` 유지
- 배당 캘린더는 `pages_app/5_dividend_calendar.py` 래퍼 페이지를 통해 동일 렌더 함수 호출

## 새 상단 메뉴 항목
- 자산 시뮬레이터
- 자산 트래커
- 양도세치기 배당시뮬
- 매도전환계산기
- 배당 캘린더

## 사이드바 유지 항목
- 로그인 사용자 표시
- 자동저장 상태 표시
- 로그아웃 버튼
- 즐겨찾기 링크 관리/목록
- 기존 `st.navigation` 페이지 메뉴

## Firebase / session_state 영향
- Firebase 스키마 변경 없음
- 자동저장/저장 타이밍 변경 없음
- 기존 session_state key 변경 없음
- 상단 메뉴 이동 추적용 `_last_top_nav_page` 신규 key 1개 추가(기존 키 불변)
