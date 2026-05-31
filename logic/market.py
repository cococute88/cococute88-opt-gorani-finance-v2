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
