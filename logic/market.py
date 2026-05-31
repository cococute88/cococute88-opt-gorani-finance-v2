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



# ──────────────────────────────────────────────
# STEP 5: 원화(KRW) 환산 / 날짜 병합 순수 함수
# ──────────────────────────────────────────────
def _clean_series_for_merge(series) -> pd.Series:
    """병합용 정규화: 숫자화·결측제거·중복 인덱스 제거·정렬."""
    cleaned = _coerce_close(series)
    if cleaned.empty:
        return cleaned
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    return cleaned.sort_index()


def align_and_convert_to_krw(usd_close, usdkrw_rate):
    """달러 종가를 USD/KRW 환율로 원화 환산한다.

    미국 거래일과 환율 날짜가 다를 수 있으므로, 환율을 (달러+환율) 합집합
    인덱스에 reindex 후 ``ffill`` 하여 달러 거래일에 맞춘다. (bfill 미사용)
    시작 구간 환율이 없어 환산 불가한 날은 자동으로 제외(dropna)된다.

    반환: ``(krw_close: Series, aligned_rate: Series)``.
    환산 가능한 데이터가 없으면 빈 Series 두 개를 반환한다.
    """
    usd = _clean_series_for_merge(usd_close)
    fx = _clean_series_for_merge(usdkrw_rate)

    empty = pd.Series(dtype="float64")
    if usd.empty or fx.empty:
        return empty, empty

    combined_index = usd.index.union(fx.index)
    fx_ffilled = fx.reindex(combined_index).ffill()
    aligned_rate = fx_ffilled.reindex(usd.index)

    krw_close = (usd * aligned_rate).dropna()
    if krw_close.empty:
        return empty, empty

    aligned_rate = aligned_rate.reindex(krw_close.index)
    return krw_close, aligned_rate



# ──────────────────────────────────────────────
# 고라니 시장온도 v2: CNN 유사 7요소 방식 순수 계산 함수
# ──────────────────────────────────────────────
import math as _math


def rolling_zscore(series, window: int = 252, min_periods: int = 80):
    """시계열의 rolling z-score 최신값을 반환한다.

    (latest - rolling_mean) / rolling_std. 데이터 부족/std==0 이면 None.
    """
    s = _coerce_close(series)
    if s.empty or len(s) < min_periods:
        return None
    rm = s.rolling(window, min_periods=min_periods).mean()
    rs = s.rolling(window, min_periods=min_periods).std()
    latest_val = _to_float_or_none(s.iloc[-1])
    latest_mean = _to_float_or_none(rm.iloc[-1])
    latest_std = _to_float_or_none(rs.iloc[-1])
    if latest_val is None or latest_mean is None or latest_std is None:
        return None
    if latest_std == 0:
        return 0.0
    return (latest_val - latest_mean) / latest_std


def sigmoid_score(z, k: float = 1.0, invert: bool = False):
    """z-score 를 0~100 점수로 변환한다 (로지스틱 함수).

    z=0 → 50, z↑ → 100, z↓ → 0.
    invert=True 이면 공포방향 지표의 부호를 반전한다.
    """
    val = _to_float_or_none(z)
    if val is None:
        return None
    if invert:
        val = -val
    # 극단값 방어 (exp overflow 방지)
    val = max(-10.0, min(10.0, val * k))
    try:
        score = 100.0 / (1.0 + _math.exp(-val))
    except OverflowError:
        score = 0.0 if val < 0 else 100.0
    return clip_score(score)


def _ratio_latest(series_a, series_b):
    """두 종가 Series 의 최신 비율 (a/b) 을 반환한다. 실패 시 None."""
    a = _coerce_close(series_a)
    b = _coerce_close(series_b)
    if a.empty or b.empty:
        return None
    la = _to_float_or_none(a.iloc[-1])
    lb = _to_float_or_none(b.iloc[-1])
    if la is None or lb is None or lb == 0:
        return None
    return la / lb


def _return_n(series, n: int = 20):
    """최근 n 거래일 단순 수익률 (latest/latest-n - 1). 실패 시 None."""
    s = _coerce_close(series)
    if s.empty or len(s) <= n:
        return None
    latest = _to_float_or_none(s.iloc[-1])
    past = _to_float_or_none(s.iloc[-1 - n])
    if latest is None or past is None or past == 0:
        return None
    return latest / past - 1.0


def compute_gorani_v2_components(
    spy_close=None,
    rsp_close=None,
    hyg_close=None,
    lqd_close=None,
    tlt_close=None,
    vix_close=None,
    pcr_close=None,
    vix3m_close=None,
    k: float = 1.0,
):
    """고라니 시장온도 v2 구성요소별 점수(0~100)를 계산한다.

    CNN 7요소 철학 근사: 252일 rolling z-score → sigmoid(0~100).
    각 구성요소에 계산 가능 여부와 점수를 dict 로 반환한다.
    반환: {"components": {이름: score|None}, "available_count": int}
    """
    components = {}

    # 1) Momentum: SPY / SPY 125일 MA - 1
    spy = _coerce_close(spy_close)
    if not spy.empty and len(spy) >= 125:
        ma125 = spy.rolling(125).mean()
        momentum_raw = spy / ma125 - 1.0
        z = rolling_zscore(momentum_raw, window=252, min_periods=80)
        components["Momentum"] = sigmoid_score(z, k=k, invert=False)
    else:
        components["Momentum"] = None

    # 2) Price Strength: SPY 252일 구간 내 위치
    if not spy.empty and len(spy) >= 252:
        window_252 = spy.tail(252)
        hi = _to_float_or_none(window_252.max())
        lo = _to_float_or_none(window_252.min())
        cur = _to_float_or_none(spy.iloc[-1])
        if hi is not None and lo is not None and cur is not None and hi != lo:
            raw_position = (cur - lo) / (hi - lo)  # 0~1
            # z-score 대신 직접 0~100 매핑 (이미 0~1 범위)
            components["Price Strength"] = clip_score(raw_position * 100.0)
        else:
            components["Price Strength"] = None
    else:
        components["Price Strength"] = None

    # 3) Breadth: RSP/SPY 비율의 20일 변화율
    rsp = _coerce_close(rsp_close)
    if not spy.empty and not rsp.empty:
        # 날짜 합집합 → 비율 계산
        combined = pd.concat([spy.rename("spy"), rsp.rename("rsp")], axis=1).dropna()
        if len(combined) >= 40:
            ratio = combined["rsp"] / combined["spy"]
            ratio_change = ratio / ratio.shift(20) - 1.0
            z = rolling_zscore(ratio_change.dropna(), window=252, min_periods=80)
            components["Breadth"] = sigmoid_score(z, k=k, invert=False)
        else:
            components["Breadth"] = None
    else:
        components["Breadth"] = None

    # 4) Put/Call: PCR (높을수록 공포 → invert)
    pcr = _coerce_close(pcr_close)
    if not pcr.empty and len(pcr) >= 10:
        pcr_5d = pcr.rolling(5).mean()
        z = rolling_zscore(pcr_5d.dropna(), window=252, min_periods=80)
        components["Put/Call"] = sigmoid_score(z, k=k, invert=True)
    else:
        # proxy: VIX / VIX3M (높을수록 단기 공포 우위 → invert)
        vix = _coerce_close(vix_close)
        vix3m = _coerce_close(vix3m_close)
        if not vix.empty and not vix3m.empty:
            combined_v = pd.concat([vix.rename("vix"), vix3m.rename("vix3m")], axis=1).dropna()
            if len(combined_v) >= 80:
                ratio_v = combined_v["vix"] / combined_v["vix3m"]
                z = rolling_zscore(ratio_v, window=252, min_periods=80)
                components["Put/Call"] = sigmoid_score(z, k=k, invert=True)
            else:
                components["Put/Call"] = None
        else:
            components["Put/Call"] = None

    # 5) Junk Bond Demand: HYG/LQD 비율
    hyg = _coerce_close(hyg_close)
    lqd = _coerce_close(lqd_close)
    if not hyg.empty and not lqd.empty:
        combined_jb = pd.concat([hyg.rename("hyg"), lqd.rename("lqd")], axis=1).dropna()
        if len(combined_jb) >= 80:
            ratio_jb = combined_jb["hyg"] / combined_jb["lqd"]
            z = rolling_zscore(ratio_jb, window=252, min_periods=80)
            components["Junk Bond"] = sigmoid_score(z, k=k, invert=False)
        else:
            components["Junk Bond"] = None
    else:
        components["Junk Bond"] = None

    # 6) Market Volatility: VIX / VIX 50일 MA (높을수록 공포 → invert)
    vix = _coerce_close(vix_close)
    if not vix.empty and len(vix) >= 80:
        vix_ma50 = vix.rolling(50).mean()
        vix_ratio = vix / vix_ma50
        z = rolling_zscore(vix_ratio.dropna(), window=252, min_periods=80)
        components["Volatility"] = sigmoid_score(z, k=k, invert=True)
    else:
        components["Volatility"] = None

    # 7) Safe Haven Demand: SPY 20일 수익률 - TLT 20일 수익률
    tlt = _coerce_close(tlt_close)
    if not spy.empty and not tlt.empty:
        spy_ret = _return_n(spy, 20)
        tlt_ret = _return_n(tlt, 20)
        if spy_ret is not None and tlt_ret is not None:
            # 전체 시계열의 z-score 필요 → 비율 시계열 생성
            combined_sh = pd.concat([spy.rename("spy"), tlt.rename("tlt")], axis=1).dropna()
            if len(combined_sh) >= 80:
                spy_roll_ret = combined_sh["spy"] / combined_sh["spy"].shift(20) - 1.0
                tlt_roll_ret = combined_sh["tlt"] / combined_sh["tlt"].shift(20) - 1.0
                diff = (spy_roll_ret - tlt_roll_ret).dropna()
                z = rolling_zscore(diff, window=252, min_periods=80)
                components["Safe Haven"] = sigmoid_score(z, k=k, invert=False)
            else:
                components["Safe Haven"] = None
        else:
            components["Safe Haven"] = None
    else:
        components["Safe Haven"] = None

    available = {name: score for name, score in components.items() if score is not None}
    return {"components": components, "available_count": len(available)}


def compute_gorani_market_temperature_v2(
    spy_close=None,
    rsp_close=None,
    hyg_close=None,
    lqd_close=None,
    tlt_close=None,
    vix_close=None,
    pcr_close=None,
    vix3m_close=None,
    k: float = 1.0,
    min_components: int = 4,
):
    """고라니 시장온도 v2 최종 점수를 계산한다.

    반환: {"score": float|None, "components": dict, "available_count": int, "ok": bool}
    ok=False 이면 min_components 미충족. score 는 가용 구성요소 평균.
    """
    result = compute_gorani_v2_components(
        spy_close=spy_close,
        rsp_close=rsp_close,
        hyg_close=hyg_close,
        lqd_close=lqd_close,
        tlt_close=tlt_close,
        vix_close=vix_close,
        pcr_close=pcr_close,
        vix3m_close=vix3m_close,
        k=k,
    )
    comps = result["components"]
    available = {n: s for n, s in comps.items() if s is not None}
    count = len(available)

    if count < min_components:
        return {
            "score": None,
            "components": comps,
            "available_count": count,
            "ok": False,
        }

    score = sum(available.values()) / count
    return {
        "score": clip_score(score),
        "components": comps,
        "available_count": count,
        "ok": True,
    }
