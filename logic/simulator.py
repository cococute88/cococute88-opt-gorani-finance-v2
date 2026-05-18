"""자산 시뮬레이터 순수 로직 (UI와 완전 분리)"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# 데이터 모델
# ============================================================
@dataclass
class SimConfig:
    start_year: int = 2026
    sim_years: int = 30
    init_isa: float = 2000.0           # 만원
    init_pension: float = 11897.0
    init_general: float = 0.0
    return_rate: float = 0.06          # 연 6%
    inflation_rate: float = 0.03
    withdraw_rate: float = 0.035       # 2051년~ 인출률
    withdraw_increase: float = 0.03    # 인출금 연간 증액률
    withdraw_delay: int = 1            # 은퇴 후 인출 미룰 년수


@dataclass
class YearPlan:
    year: int
    monthly: float = 0.0               # 월 적립액 (만원)
    pension_check: bool = False
    isa_check: bool = False
    isa_transfer: bool = False
    status: str = "적립"               # 적립 / 은퇴 / 인출

    @property
    def annual(self) -> float:
        return self.monthly * 12


@dataclass
class YearResult:
    year: int
    status: str
    pension_deposit: float = 0.0
    pension_balance: float = 0.0
    isa_deposit: float = 0.0
    isa_balance: float = 0.0
    general_deposit: float = 0.0
    general_balance: float = 0.0
    total_balance: float = 0.0
    from_prev_general_for_pension: float = 0.0
    from_prev_general_for_isa: float = 0.0
    isa_transferred: float = 0.0
    total_pension_deposit: float = 0.0
    total_isa_deposit: float = 0.0
    # 수익률 반영 후
    pension_nominal: float = 0.0
    isa_nominal: float = 0.0
    general_nominal: float = 0.0
    total_nominal: float = 0.0


# ============================================================
# 상태 자동 결정 (월 적립액 < 1,000만원/년이 되는 해부터 은퇴)
# ============================================================
def assign_statuses(plans: List[YearPlan]) -> None:
    retire_found = False
    for p in plans:
        if not retire_found and p.annual < 1000:
            p.status = "은퇴"
            retire_found = True
        elif retire_found:
            p.status = "인출"
        else:
            p.status = "적립"


def find_retire_index(plans: List[YearPlan]) -> int:
    for i, p in enumerate(plans):
        if p.status == "은퇴":
            return i
    return -1


# ============================================================
# 적립 시뮬레이션 (ISA/연금 한도, ISA→연금 이전, 일반위탁 보충)
# ============================================================
def simulate_deposits(cfg: SimConfig, plans: List[YearPlan]) -> List[YearResult]:
    results: List[YearResult] = []
    prev_isa = cfg.init_isa
    prev_pension = cfg.init_pension
    prev_general = cfg.init_general
    total_pension_dep = 0.0
    total_isa_dep = 0.0

    for p in plans:
        is_retired_or_after = p.status in ("은퇴", "인출")
        is_after_retire = p.status == "인출"

        annual = p.annual
        pension_dep = 0.0
        isa_dep = 0.0
        general_dep = 0.0
        from_prev_g_pension = 0.0
        from_prev_g_isa = 0.0

        # ISA → 연금 이전
        isa_transferred = 0.0
        if p.isa_transfer and prev_isa > 0:
            isa_transferred = prev_isa
            prev_pension += prev_isa
            prev_isa = 0.0

        pension_limit = 3800.0 if p.isa_transfer else 1800.0
        isa_limit = 2000.0

        if not is_after_retire:
            # 연금저축 적립
            if p.pension_check and not is_retired_or_after:
                if annual >= pension_limit:
                    pension_dep = pension_limit
                    annual -= pension_limit
                else:
                    pension_dep = annual
                    needed = pension_limit - annual
                    if prev_general >= needed:
                        from_prev_g_pension = needed
                        pension_dep = pension_limit
                        prev_general -= needed
                    else:
                        from_prev_g_pension = prev_general
                        pension_dep += prev_general
                        prev_general = 0.0
                    annual = 0.0

            # ISA 적립
            if p.isa_check:
                if is_retired_or_after:
                    if prev_general >= isa_limit:
                        from_prev_g_isa = isa_limit
                        isa_dep = isa_limit
                        prev_general -= isa_limit
                    else:
                        from_prev_g_isa = prev_general
                        isa_dep = prev_general
                        prev_general = 0.0
                    remain = min(annual, isa_limit - isa_dep)
                    isa_dep += remain
                    annual -= remain
                else:
                    if annual >= isa_limit:
                        isa_dep = isa_limit
                        annual -= isa_limit
                    else:
                        isa_dep = annual
                        needed = isa_limit - annual
                        if prev_general >= needed:
                            from_prev_g_isa = needed
                            isa_dep = isa_limit
                            prev_general -= needed
                        else:
                            from_prev_g_isa = prev_general
                            isa_dep += prev_general
                            prev_general = 0.0
                        annual = 0.0
            general_dep = annual
        else:
            # 인출 단계 (적립 X, ISA만 일반위탁→ISA 이동 가능)
            if p.isa_check:
                if prev_general >= isa_limit:
                    from_prev_g_isa = isa_limit
                    isa_dep = isa_limit
                    prev_general -= isa_limit
                else:
                    from_prev_g_isa = prev_general
                    isa_dep = prev_general
                    prev_general = 0.0
                remain = min(annual, isa_limit - isa_dep)
                isa_dep += remain
                annual -= remain
            general_dep = annual

        pension_balance = prev_pension + pension_dep
        isa_balance = prev_isa + isa_dep
        general_balance = prev_general + general_dep

        total_pension_dep += pension_dep
        total_isa_dep += isa_dep

        results.append(YearResult(
            year=p.year, status=p.status,
            pension_deposit=pension_dep, pension_balance=pension_balance,
            isa_deposit=isa_dep, isa_balance=isa_balance,
            general_deposit=general_dep, general_balance=general_balance,
            total_balance=pension_balance + isa_balance + general_balance,
            from_prev_general_for_pension=from_prev_g_pension,
            from_prev_general_for_isa=from_prev_g_isa,
            isa_transferred=isa_transferred,
            total_pension_deposit=total_pension_dep,
            total_isa_deposit=total_isa_dep,
        ))

        prev_pension = pension_balance
        prev_isa = isa_balance
        prev_general = general_balance

    return results


# ============================================================
# 수익률 적용 (명목 잔고 누적)
# ============================================================
def apply_returns(cfg: SimConfig, results: List[YearResult]) -> None:
    pension_n = 0.0
    isa_n = 0.0
    general_n = 0.0

    for i, r in enumerate(results):
        if i == 0:
            pension_n = cfg.init_pension * (1 + cfg.return_rate) + r.pension_deposit
            isa_n = cfg.init_isa * (1 + cfg.return_rate) + r.isa_deposit
            general_n = cfg.init_general * (1 + cfg.return_rate) + r.general_deposit
            if r.isa_transferred > 0:
                pension_n += r.isa_transferred * (1 + cfg.return_rate)
                isa_n = r.isa_deposit
        else:
            pension_n = pension_n * (1 + cfg.return_rate) + r.pension_deposit
            isa_n = isa_n * (1 + cfg.return_rate) + r.isa_deposit
            general_n = general_n * (1 + cfg.return_rate) + r.general_deposit
            if r.isa_transferred > 0:
                pension_n += r.isa_transferred * (1 + cfg.return_rate)
                isa_n = r.isa_deposit

        r.pension_nominal = pension_n
        r.isa_nominal = isa_n
        r.general_nominal = general_n
        r.total_nominal = pension_n + isa_n + general_n


def get_real_balances(cfg: SimConfig, results: List[YearResult]) -> List[dict]:
    """명목 → 실질 환산 (물가상승률 적용)"""
    out = []
    cum = 1.0
    for r in results:
        cum *= (1 + cfg.inflation_rate)
        out.append({
            "year": r.year,
            "pension_real": r.pension_nominal / cum,
            "isa_real": r.isa_nominal / cum,
            "general_real": r.general_nominal / cum,
            "total_real": r.total_nominal / cum,
            "cum_inflation": cum,
        })
    return out


# ============================================================
# 인출 시뮬 - 이분탐색 최적화 (원본 JS 로직 그대로 이식)
# ============================================================
def _calc_first_by_limit(total_limit: float, years: int, eff_rate: float) -> float:
    if years <= 0:
        return 0.0
    if eff_rate == 0:
        return total_limit / years
    factor = ((1 + eff_rate) ** years - 1) / eff_rate
    return total_limit / factor


def _find_optimal(initial_balance: float, return_rate: float, eff_rate: float,
                  years: int, limit: float,
                  additional_deposits: Optional[List[float]] = None) -> float:
    """이분탐색으로 첫해 인출액 최적화 (50회 반복)
    제약: ① 잔고 ≥ 인출액  ② 누적 ≤ 한도  ③ 인출액 비감소"""
    if years <= 0:
        return 0.0
    high = _calc_first_by_limit(limit, years, eff_rate)
    low = 0.0
    optimal = 0.0

    for _ in range(50):
        mid = (low + high) / 2
        balance = initial_balance
        total_w = 0.0
        prev_w = 0.0
        valid = True

        for y in range(years):
            balance *= (1 + return_rate)
            if additional_deposits and y < len(additional_deposits):
                balance += additional_deposits[y]

            withdraw = mid * (1 + eff_rate) ** y
            if total_w + withdraw > limit:
                withdraw = max(0.0, limit - total_w)
            if withdraw > balance:
                valid = False
                break
            if withdraw < prev_w - 0.001:
                valid = False
                break

            balance -= withdraw
            total_w += withdraw
            prev_w = withdraw

        if valid:
            optimal = mid
            low = mid
        else:
            high = mid

    return optimal


# ============================================================
# 절세계좌 인출 시뮬레이션
# ============================================================
@dataclass
class WithdrawRow:
    year: int
    period_label: str           # "대기" / "~2050" / "2051~"
    is_delay: bool
    isa_gross: float = 0.0
    isa_net: float = 0.0
    isa_balance: float = 0.0
    isa_remaining_limit: Optional[float] = None
    pension_gross: float = 0.0
    pension_net: float = 0.0
    pension_balance: float = 0.0
    pension_remaining_limit: Optional[float] = None
    total_net: float = 0.0
    monthly_net: float = 0.0
    monthly_net_real: float = 0.0
    isa_tax_rate: float = 0.0
    pension_tax_rate: float = 0.0


@dataclass
class WithdrawPlan:
    retire_year: int
    actual_start_year: int
    years_until_2050: int
    isa_balance_at_start: float
    pension_balance_at_start: float
    isa_first_withdraw: float
    pension_first_withdraw: float
    isa_constraint: str         # "한도기준" / "잔고제약"
    pension_constraint: str
    pension_deposit_limit: float
    isa_limit_until_2050: float = 10000.0
    rows: List[WithdrawRow] = field(default_factory=list)
    # 합계
    total_gross_isa: float = 0.0
    total_gross_pension: float = 0.0
    total_net_isa: float = 0.0
    total_net_pension: float = 0.0
    final_isa_balance: float = 0.0
    final_pension_balance: float = 0.0


def simulate_tax_account_withdraw(cfg: SimConfig, results: List[YearResult],
                                  retire_idx: int) -> Optional[WithdrawPlan]:
    if retire_idx < 0:
        return None

    delay = max(1, min(15, cfg.withdraw_delay))
    actual_start_idx = retire_idx + delay
    if actual_start_idx >= len(results):
        return None

    retire_year = results[retire_idx].year
    actual_start_year = results[actual_start_idx].year

    # 인출 시작 시점 잔고 (지연 기간 동안 수익률 + 추가 ISA 적립)
    isa_at_start = results[retire_idx].isa_nominal or results[retire_idx].isa_balance
    pen_at_start = results[retire_idx].pension_nominal or results[retire_idx].pension_balance
    for d in range(retire_idx + 1, retire_idx + delay + 1):
        if d < len(results):
            isa_at_start *= (1 + cfg.return_rate)
            pen_at_start *= (1 + cfg.return_rate)
            isa_at_start += results[d].isa_deposit

    pension_deposit_limit = cfg.init_pension + results[retire_idx].total_pension_deposit
    isa_limit_until_2050 = 10000.0
    years_until_2050 = max(0, 2050 - actual_start_year + 1)

    isa_eff_rate = (1 + cfg.withdraw_increase) * (1 + cfg.inflation_rate) - 1
    pension_eff_rate = cfg.withdraw_increase

    # 인출 기간 동안 추가되는 ISA 적립
    isa_additional = []
    for i in range(actual_start_idx, len(results)):
        if results[i].year > 2050:
            break
        isa_additional.append(results[i].isa_deposit)

    isa_first = _find_optimal(isa_at_start, cfg.return_rate, isa_eff_rate,
                              years_until_2050, isa_limit_until_2050, isa_additional)
    pension_first = _find_optimal(pen_at_start, cfg.return_rate, pension_eff_rate,
                                  years_until_2050, pension_deposit_limit)

    isa_first_by_limit = _calc_first_by_limit(isa_limit_until_2050, years_until_2050, isa_eff_rate)
    pen_first_by_limit = _calc_first_by_limit(pension_deposit_limit, years_until_2050, pension_eff_rate)
    isa_constraint = "잔고제약" if isa_first < isa_first_by_limit * 0.99 else "한도기준"
    pen_constraint = "잔고제약" if pension_first < pen_first_by_limit * 0.99 else "한도기준"

    plan = WithdrawPlan(
        retire_year=retire_year,
        actual_start_year=actual_start_year,
        years_until_2050=years_until_2050,
        isa_balance_at_start=isa_at_start,
        pension_balance_at_start=pen_at_start,
        isa_first_withdraw=isa_first,
        pension_first_withdraw=pension_first,
        isa_constraint=isa_constraint,
        pension_constraint=pen_constraint,
        pension_deposit_limit=pension_deposit_limit,
        isa_limit_until_2050=isa_limit_until_2050,
    )

    # 연도별 실제 시뮬레이션
    pension_balance = results[retire_idx].pension_nominal or results[retire_idx].pension_balance
    isa_balance = results[retire_idx].isa_nominal or results[retire_idx].isa_balance
    total_w_isa = 0.0
    total_w_pen = 0.0
    isa_2051_base = 0.0
    pen_2051_base = 0.0
    prev_isa_w = 0.0
    prev_pen_w = 0.0

    cum_inflation = 1.0
    for i in range(retire_idx + 1):
        cum_inflation *= (1 + cfg.inflation_rate)

    for i in range(retire_idx + 1, len(results)):
        r = results[i]
        year = r.year
        cum_inflation *= (1 + cfg.inflation_rate)

        pension_balance *= (1 + cfg.return_rate)
        isa_balance *= (1 + cfg.return_rate)
        isa_balance += r.isa_deposit

        is_delay = i < actual_start_idx
        isa_g = pen_g = isa_n = pen_n = 0.0
        isa_tax = pen_tax = 0.0
        period = ""
        isa_remain = pen_remain = None

        if is_delay:
            period = "대기"
        elif year <= 2050:
            period = "~2050"
            yfs = i - actual_start_idx

            # ----------------------------------------------------
            # 1. ISA 계좌 인출액 산정 및 상한선 적용 로직 수정
            # ----------------------------------------------------
            isa_g = isa_first * (1 + isa_eff_rate) ** yfs
            if total_w_isa + isa_g > isa_limit_until_2050:
                isa_g = max(0.0, isa_limit_until_2050 - total_w_isa)
                
            # 비감소 규칙 적용 (전년도보다 적게 인출하지 않음)
            if isa_g < prev_isa_w and prev_isa_w > 0:
                isa_g = prev_isa_w
                
            # 💡 핵심 방어막: 설정된 인출률(cfg.withdraw_rate)을 절대 넘지 않도록 제한
            isa_g = min(isa_g, isa_balance * cfg.withdraw_rate)
            # 최종 잔고를 넘을 수 없음
            isa_g = min(isa_g, isa_balance)

            # ----------------------------------------------------
            # 2. 연금 계좌 인출액 산정 및 상한선 적용 로직 수정
            # ----------------------------------------------------
            pen_g = pension_first * (1 + pension_eff_rate) ** yfs
            if total_w_pen + pen_g > pension_deposit_limit:
                pen_g = max(0.0, pension_deposit_limit - total_w_pen)
                
            # 비감소 규칙 적용
            if pen_g < prev_pen_w and prev_pen_w > 0:
                pen_g = prev_pen_w
                
            # 💡 핵심 방어막: 설정된 인출률(cfg.withdraw_rate)을 절대 넘지 않도록 제한
            pen_g = min(pen_g, pension_balance * cfg.withdraw_rate)
            # 최종 잔고를 넘을 수 없음
            pen_g = min(pen_g, pension_balance)

            isa_n = isa_g
            pen_n = pen_g
            isa_remain = max(0.0, isa_limit_until_2050 - total_w_isa - isa_g)
            pen_remain = max(0.0, pension_deposit_limit - total_w_pen - pen_g)
        else:
            period = "2051~"
            isa_tax = 0.099
            pen_tax = 0.055
            yf2051 = year - 2051

            if yf2051 == 0:
                isa_2051_base = isa_balance * cfg.withdraw_rate
                pen_2051_base = pension_balance * cfg.withdraw_rate
                isa_g = isa_2051_base
                pen_g = pen_2051_base
            else:
                isa_g = isa_2051_base * (1 + cfg.withdraw_increase) ** yf2051
                pen_g = pen_2051_base * (1 + cfg.withdraw_increase) ** yf2051

            isa_g = min(isa_g, isa_balance)
            pen_g = min(pen_g, pension_balance)
            isa_n = isa_g * (1 - isa_tax)
            pen_n = pen_g * (1 - pen_tax)

        if not is_delay:
            prev_isa_w = isa_g
            prev_pen_w = pen_g

        isa_balance = max(0.0, isa_balance - isa_g)
        pension_balance = max(0.0, pension_balance - pen_g)
        total_w_isa += isa_g
        total_w_pen += pen_g

        total_net = isa_n + pen_n
        monthly_net = total_net / 12
        monthly_net_real = monthly_net / cum_inflation

        plan.rows.append(WithdrawRow(
            year=year, period_label=period, is_delay=is_delay,
            isa_gross=isa_g, isa_net=isa_n, isa_balance=isa_balance,
            isa_remaining_limit=isa_remain,
            pension_gross=pen_g, pension_net=pen_n, pension_balance=pension_balance,
            pension_remaining_limit=pen_remain,
            total_net=total_net,
            monthly_net=monthly_net, monthly_net_real=monthly_net_real,
            isa_tax_rate=isa_tax, pension_tax_rate=pen_tax,
        ))

        plan.total_gross_isa += isa_g
        plan.total_gross_pension += pen_g
        plan.total_net_isa += isa_n
        plan.total_net_pension += pen_n

    plan.final_isa_balance = isa_balance
    plan.final_pension_balance = pension_balance
    return plan


# ============================================================
# 전체자산 인출 시뮬 (Tab3)
# ============================================================
@dataclass
class TotalWithdrawRow:
    year: int
    total_nominal: float
    withdraw: float = 0.0
    monthly: float = 0.0
    after_balance: float = 0.0
    real_withdraw: float = 0.0
    is_withdraw: bool = False


def simulate_total_withdraw(cfg: SimConfig, results: List[YearResult],
                            retire_idx: int) -> List[TotalWithdrawRow]:
    rows: List[TotalWithdrawRow] = []
    if retire_idx < 0:
        retire_idx = len(results)

    first_w = 0.0
    cum_infl = 1.0
    for i, r in enumerate(results):
        cum_infl *= (1 + cfg.inflation_rate)
        is_w = i > retire_idx
        w = 0.0
        after = r.total_nominal
        real = 0.0

        if is_w:
            if first_w == 0:
                first_w = r.total_nominal * cfg.withdraw_rate
                w = first_w
            else:
                w = first_w * (1 + cfg.withdraw_increase) ** (i - retire_idx - 1)
            after = r.total_nominal - w
            real = w / cum_infl

        rows.append(TotalWithdrawRow(
            year=r.year, total_nominal=r.total_nominal,
            withdraw=w, monthly=w / 12,
            after_balance=after, real_withdraw=real, is_withdraw=is_w,
        ))
    return rows
