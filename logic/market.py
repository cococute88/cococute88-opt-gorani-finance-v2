"""시장온도 탭용 순수 계산 함수 모음.

Streamlit / 외부 API 에 의존하지 않는 순수 함수만 둔다.
입력 데이터가 비어 있거나 결측치가 섞여 있어도 예외로 앱이 죽지 않도록
방어적으로 동작하는 것을 원칙으로 한다.
"""

from __future__ import annotations

import pandas as pd


def _coerce_close(close) -> pd.Series:
    """입력을 숫자형 종가 Series 로 정규화한다.

    - DataFrame 이면 첫 번째 컬럼을 사용한다.
    - 숫자로 변환할 수 없는 값은 NaN 으로 처리 후 제거한다.
    - 변환 불가능하거나 비어 있으면 빈 Series 를 반환한다.
    """
    if close is None:
        return pd.Series(dtype="float64")

    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            return pd.Series(dtype="float64")
        series = close.iloc[:, 0]
    elif isinstance(close, pd.Series):
        series = close
    else:
        try:
            series = pd.Series(close)
        except Exception:
            return pd.Series(dtype="float64")

    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    return series


def compute_rsi(close, period: int = 14) -> pd.Series:
    """Wilder 방식 RSI 를 pandas 만으로 직접 계산한다 (pandas_ta 미사용).

    Wilder 의 평활(RMA)은 ``alpha = 1/period`` 인 지수가중이동평균과 동일하다.
    데이터가 부족하면 입력 인덱스에 맞춘 NaN Series 를 반환한다.
    """
    series = _coerce_close(close)

    try:
        period = int(period)
    except (TypeError, ValueError):
        period = 14
    if period < 1:
        period = 14

    if series.empty or len(series) <= period:
        # 계산이 불가능하면 인덱스를 보존한 NaN Series 반환
        return pd.Series(index=series.index, dtype="float64")

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 손실 평균이 0 이면 (상승만 존재) RSI = 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    # 상승/하락이 모두 0 (완전 횡보) 이면 RSI = 50 으로 정의
    flat_mask = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.mask(flat_mask, 50.0)

    return rsi


def compute_drawdown_series(close) -> pd.Series:
    """고점(누적 최댓값) 대비 하락률 시계열을 반환한다.

    값은 비율(예: -0.25 == -25%) 이다. ``close / close.cummax() - 1``.
    데이터가 비어 있으면 빈 Series 를 반환한다.
    """
    series = _coerce_close(close)
    if series.empty:
        return pd.Series(dtype="float64")

    running_max = series.cummax()
    # running_max 가 0 이하인 비정상 구간은 0(하락 없음)으로 처리
    drawdown = (series / running_max) - 1.0
    drawdown = drawdown.where(running_max > 0, 0.0)
    return drawdown


def compute_mdd(close) -> dict:
    """최대 낙폭(MDD)과 고점일/저점일을 함께 계산한다.

    반환 형식::

        {"mdd": float|None, "peak_date": Timestamp|None, "trough_date": Timestamp|None}

    데이터가 비어 있거나 계산 불가하면 모든 값이 ``None`` 이다.
    ``mdd`` 는 비율(예: -0.3 == -30%) 이다.
    """
    empty_result = {"mdd": None, "peak_date": None, "trough_date": None}

    series = _coerce_close(close)
    if series.empty:
        return empty_result

    drawdown = compute_drawdown_series(series)
    if drawdown.empty or drawdown.isna().all():
        return empty_result

    trough_date = drawdown.idxmin()
    mdd_value = float(drawdown.loc[trough_date])

    # 고점일: 저점일 이전(포함) 구간에서 가격이 최대였던 날
    running = series.loc[:trough_date]
    if running.empty:
        return empty_result
    peak_date = running.idxmax()

    return {
        "mdd": mdd_value,
        "peak_date": peak_date,
        "trough_date": trough_date,
    }



# ──────────────────────────────────────────────
# STEP 3: 시장 심리(Fear & Greed / 고라니 시장온도) 순수 계산 함수
# ──────────────────────────────────────────────
def _to_float_or_none(value):
    """숫자로 변환 가능하면 float, 아니면(또는 NaN) None 을 반환한다."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def clip_score(value, low: float = 0.0, high: float = 100.0):
    """값을 [low, high] 범위로 제한한다. 변환 불가/NaN 이면 None."""
    result = _to_float_or_none(value)
    if result is None:
        return None
    return max(low, min(high, result))


def classify_fear_greed_score(score):
    """0~100 점수를 한국어 심리 라벨로 분류한다.

    0~25 극단적 공포 / 25~45 공포 / 45~55 중립 / 55~75 탐욕 / 75~100 극단적 탐욕.
    (구간 하한 포함 기준) 값이 없으면 None 을 반환한다.
    """
    result = _to_float_or_none(score)
    if result is None:
        return None
    if result < 25:
        return "극단적 공포"
    if result < 45:
        return "공포"
    if result < 55:
        return "중립"
    if result < 75:
        return "탐욕"
    return "극단적 탐욕"


def compute_distance_from_moving_average(close, window: int = 200):
    """종가의 (현재가 / window일 단순이동평균 - 1) 을 반환한다.

    값은 비율(예: 0.05 == 이동평균 대비 +5%). 데이터가 부족하거나 계산
    불가하면 None 을 반환한다.
    """
    series = _coerce_close(close)

    try:
        window = int(window)
    except (TypeError, ValueError):
        return None
    if window < 1 or series.empty or len(series) < window:
        return None

    ma = series.rolling(window).mean()
    latest_price = _to_float_or_none(series.iloc[-1])
    latest_ma = _to_float_or_none(ma.iloc[-1])
    if latest_price is None or latest_ma is None or latest_ma == 0:
        return None
    return latest_price / latest_ma - 1.0


def compute_gorani_market_temperature(
    qqq_rsi=None,
    spy_rsi=None,
    qqq_drawdown=None,
    spy_drawdown=None,
    spy_ma_distance=None,
    vix_level=None,
):
    """가용한 구성요소만으로 0~100 의 자체 "고라니 시장온도" 점수를 산출한다.

    방향성:
      - RSI 가 높을수록 탐욕(점수↑)
      - 고점대비 하락폭이 작을수록 탐욕(점수↑)
      - SPY 가 200일선 위에 있을수록 탐욕(점수↑)
      - VIX 가 낮을수록 탐욕(점수↑)

    구성요소가 하나도 없으면 score=None. CNN 7요소를 복제하지 않는 단순 합성.
    반환: {"score": float|None, "components": {이름: 0~100 점수}}.
    """
    components = {}

    rsi_qqq = clip_score(qqq_rsi)
    if rsi_qqq is not None:
        components["QQQ RSI"] = rsi_qqq

    rsi_spy = clip_score(spy_rsi)
    if rsi_spy is not None:
        components["SPY RSI"] = rsi_spy

    # 하락률(음수 비율): 0 → 100점, -50% 이하 → 0점
    dd_qqq = _to_float_or_none(qqq_drawdown)
    if dd_qqq is not None:
        components["QQQ 하락률"] = clip_score(100.0 + dd_qqq * 200.0)

    dd_spy = _to_float_or_none(spy_drawdown)
    if dd_spy is not None:
        components["SPY 하락률"] = clip_score(100.0 + dd_spy * 200.0)

    # 200일선 대비 위치: 0% → 50점, +20% → 100점, -20% → 0점
    ma_dist = _to_float_or_none(spy_ma_distance)
    if ma_dist is not None:
        components["SPY 200일선"] = clip_score(50.0 + ma_dist * 250.0)

    # VIX 레벨: 10 → 100점, 40 → 0점 (낮을수록 탐욕)
    vix = _to_float_or_none(vix_level)
    if vix is not None:
        components["VIX"] = clip_score(100.0 - (vix - 10.0) * (100.0 / 30.0))

    valid = {k: v for k, v in components.items() if v is not None}
    if not valid:
        return {"score": None, "components": {}}

    score = sum(valid.values()) / len(valid)
    return {"score": clip_score(score), "components": valid}


def _is_score_value(value) -> bool:
    """0~100 범위의 점수 후보 숫자인지 판별한다 (bool 제외)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != numeric:  # NaN
            return False
        return 0.0 <= numeric <= 100.0
    return False


def find_score_in_payload(
    payload,
    preferred_keys=("score", "value", "now", "current", "rating_value"),
    max_depth: int = 6,
):
    """JSON 유사 구조에서 0~100 사이의 점수 값을 방어적으로 재귀 탐색한다.

    우선 키(score/value/current/now 등)를 먼저 확인하고, 없으면 중첩 구조를
    재귀 탐색한다. 응답 구조가 바뀌어도 동작하도록 만든 폴백용 헬퍼이며,
    찾지 못하면 None 을 반환한다.
    """

    def _search(obj, depth):
        if depth > max_depth:
            return None
        if isinstance(obj, dict):
            # 1) 우선 키가 직접 점수 값을 가지면 즉시 반환
            for key in preferred_keys:
                if key in obj and _is_score_value(obj[key]):
                    return float(obj[key])
            # 2) 우선 키의 중첩 구조를 먼저 탐색
            for key in preferred_keys:
                if key in obj and isinstance(obj[key], (dict, list)):
                    found = _search(obj[key], depth + 1)
                    if found is not None:
                        return found
            # 3) 나머지 값을 재귀 탐색
            for nested in obj.values():
                if isinstance(nested, (dict, list)):
                    found = _search(nested, depth + 1)
                    if found is not None:
                        return found
            return None
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    found = _search(item, depth + 1)
                    if found is not None:
                        return found
            return None
        return None

    return _search(payload, 0)



# ──────────────────────────────────────────────
# STEP 4: MDD 계산기용 순수 계산 함수
# ──────────────────────────────────────────────
def _scalar_at(series, label):
    """라벨(날짜)에 해당하는 종가를 스칼라 float 로 안전하게 반환한다.

    중복 인덱스로 Series 가 반환되면 첫 값을 사용하고, 실패하면 None.
    """
    if series is None or label is None:
        return None
    try:
        value = series.loc[label]
    except Exception:
        return None
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    return _to_float_or_none(value)


def compute_recovery_date(close, peak_date, trough_date):
    """저점일 이후 가격이 고점일 가격 이상으로 처음 회복한 날짜를 반환한다.

    회복하지 못했거나 계산 불가하면 None 을 반환한다.
    """
    series = _coerce_close(close)
    if series.empty or peak_date is None or trough_date is None:
        return None

    peak_price = _scalar_at(series, peak_date)
    if peak_price is None:
        return None

    after = series.loc[series.index > trough_date]
    if after.empty:
        return None

    recovered = after[after >= peak_price]
    if recovered.empty:
        return None
    return recovered.index[0]


def compute_mdd_details(close) -> dict:
    """달러 기준 MDD 분석에 필요한 값들을 한 번에 계산한다.

    반환 키:
      current_price, period_high, current_drawdown(비율),
      mdd(비율), peak_date, trough_date, peak_price, trough_price,
      recovery_date, recovered(bool)

    데이터가 비어 있으면 가격/비율 값은 None, recovered=False.
    """
    base = {
        "current_price": None,
        "period_high": None,
        "current_drawdown": None,
        "mdd": None,
        "peak_date": None,
        "trough_date": None,
        "peak_price": None,
        "trough_price": None,
        "recovery_date": None,
        "recovered": False,
    }

    series = _coerce_close(close)
    if series.empty:
        return base

    current_price = _to_float_or_none(series.iloc[-1])
    period_high = _to_float_or_none(series.max())

    running_max = series.cummax()
    last_max = _to_float_or_none(running_max.iloc[-1])
    current_drawdown = None
    if current_price is not None and last_max not in (None, 0):
        current_drawdown = current_price / last_max - 1.0

    mdd_info = compute_mdd(series)
    peak_date = mdd_info["peak_date"]
    trough_date = mdd_info["trough_date"]

    peak_price = _scalar_at(series, peak_date)
    trough_price = _scalar_at(series, trough_date)

    recovery_date = None
    if peak_date is not None and trough_date is not None:
        recovery_date = compute_recovery_date(series, peak_date, trough_date)

    return {
        "current_price": current_price,
        "period_high": period_high,
        "current_drawdown": current_drawdown,
        "mdd": mdd_info["mdd"],
        "peak_date": peak_date,
        "trough_date": trough_date,
        "peak_price": peak_price,
        "trough_price": trough_price,
        "recovery_date": recovery_date,
        "recovered": recovery_date is not None,
    }
