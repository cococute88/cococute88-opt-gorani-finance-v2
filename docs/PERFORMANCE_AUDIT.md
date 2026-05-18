# PERFORMANCE AUDIT (1차 저위험 최적화)

## 범위 및 원칙
- 기능/계산/Firebase 스키마/저장 경로/자동저장 타이밍/UI 구조 유지.
- 이번 단계는 **Low risk** 항목만 반영.
- 사용자별 데이터(`asset_data`, `tracker_cfg`, `sim_cfg`, `div_calendar`, `favorite_links`)는 캐시 대상에서 제외.

## 병목 후보 분석 (파일/함수 단위)

### 1) `modules/dividend_calendar.py`
- `fetch_ticker_bundle()`:
  - yfinance + Finnhub + Polygon 연쇄 호출이 있어 네트워크 지연이 큼.
  - 이미 `st.cache_data(ttl=1h)` 적용되어 있으나, 내부 HTTP 연결 재사용 없음.
- `_safe_request()`:
  - 매 호출마다 새 TCP 연결 생성 가능성.
- `infer_frequency_months()`:
  - 동일 배당 시계열에서 반복 계산 가능.
- `project_future_dividends()`:
  - 동일 입력에서 이벤트 계산 반복 가능.

### 2) `core/firebase.py`
- 모듈 로드시 Firebase 초기화 수행.
- 재실행 환경에서 초기화 가드가 있어도 리소스 획득 경로를 명시적으로 캐시하면 안정성 향상 여지.

### 3) `app.py` / `core/sync.py`
- `load_all_data()`는 `data_loaded` 플래그로 중복 로드 방지 중.
- 사용자 데이터 캐시는 금지 대상이므로 현 구조 유지가 안전.

### 4) `pages_app/*`
- 변환 계산기/배당시뮬 일부는 이미 `st.cache_data` 사용.
- 위젯 key/session_state 구조는 안정성 중요 구간으로 변경 리스크 존재.

## 위험도 분류

### Low risk (이번 반영)
1. HTTP 세션 재사용(`st.cache_resource`)으로 외부 API 연결 비용 절감.
2. 순수 계산 함수 캐시 (`infer_frequency_months`) 추가.
3. 배당 이벤트 투영 계산 캐시(직렬화 dict 기반) 추가.
4. Firebase client 초기화 경로 `st.cache_resource`로 중복 초기화 방지 강화.

### Medium risk (테스트 필요, 이번 미반영)
1. 페이지별 대규모 lazy import 재배치 (초기 import 순서/사이드 이펙트 검증 필요).
2. 캘린더 렌더 경로의 데이터 구조 정규화 타이밍 조정.
3. 페이지 rerun 단위(폼/콜백) 변경.

### High risk (이번 미반영)
1. Firebase load/save 인터페이스/경로/스키마 변경.
2. 자동저장 트리거 시점 변경.
3. session_state 키/Streamlit widget key 변경.
4. 계산식(절세계산/시뮬레이터 공식) 변경.

## 실제 반영 내용

### A. 외부 API 네트워크 최적화
- `modules/dividend_calendar.py`
  - `_http_session()` 추가 (`st.cache_resource`) 후 `_safe_request()`에서 재사용.
  - 기대효과: 동일 세션 내 TCP/TLS 핸드셰이크 감소로 요청 지연 완화.

### B. 순수 계산 캐시
- `modules/dividend_calendar.py`
  - `infer_frequency_months()`에 `st.cache_data(ttl=6h)` 적용.
  - 이유: 입력 배당 시계열이 동일하면 결과가 동일한 순수 함수.

### C. 배당 이벤트 투영 캐시
- `modules/dividend_calendar.py`
  - 투영 계산부를 `_project_future_dividends_cached()`로 분리하고 `st.cache_data(ttl=3h)` 적용.
  - 캐시 저장 형태는 `DividendEvent` 객체 대신 직렬화 가능한 dict list.
  - 기존 외부 인터페이스 `project_future_dividends()`는 유지(결과 타입 동일: `List[DividendEvent]`).

### D. Firebase 리소스 초기화 안정화
- `core/firebase.py`
  - `_get_db_client()`를 `st.cache_resource`로 구성.
  - `save_data/load_data` 시 캐시된 client를 통해 reference 사용.
  - 기존 secrets 구조, 경로(`users/{uid}/{path}`), 함수 인터페이스 유지.

## TTL 정책 및 이유
- `fetch_ticker_bundle`: **1시간** (기존 유지)
  - 배당/실적/기초 가격 혼합 데이터로 실시간성-호출비용 절충.
- `infer_frequency_months`: **6시간**
  - 과거 배당 패턴 추론은 고빈도 변동 없음.
- `_project_future_dividends_cached`: **3시간**
  - 캘린더성 결과는 분 단위 갱신 필요성이 낮음.
- `get_historical_tax_saving`: **24시간** (기존 유지)
  - 과거 데이터 기반 통계.

## 건드리지 않은 위험 구간
- Firebase 저장 경로/스키마/동기화 플로우.
- 자동저장 함수 호출 타이밍.
- OAuth/로그인 처리.
- 페이지 파일명/네비게이션/UI 구조.
- 계산 공식 전반.
