from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ContractTerms:
    alpha: float
    r_cap: float
    principal: float
    cost_of_capital: float
    target_repayment: float

@dataclass
class RepaymentResult:
    terms: ContractTerms
    schedule: list[float] = field(default_factory = list)
    periods_to_repay: int | None = None
    total_repaid: float = 0.0
    missed_or_low_weeks: int = 0

def compute_alpha(
            theta_hat: float,
            sigma_cf: float,
            alpha_base: float = 0.15,
            lambda1: float = -0.10,
            lambda2: float = 0.20,
            alpha_bounds: tuple[float, float] = (0.05, 0.45),
            ) -> float:
    alpha = alpha_base + lambda1 * theta_hat + lambda2 * sigma_cf
    lo, hi = alpha_bounds
    return float(np.clip(alpha, lo, hi))

def make_contract(
            theta_hat: float,
            sigma_cf: float,
            mean_income: float,
            principal: float,
            cost_of_capital: float = 0.18,
            cap_multiple: float = 2.2,
            alpha_base: float = 0.15,
            lambda1: float = -0.10,
            lambda2: float = 0.20
            ) -> ContractTerms:
     alpha = compute_alpha(theta_hat, sigma_cf, alpha_base, lambda1, lambda2)
     r_cap = cap_multiple * alpha * mean_income
     target_repayment = principal * (1 + cost_of_capital)
     return ContractTerms(
                alpha = alpha,
                r_cap = r_cap,
                principal = principal,
                cost_of_capital = cost_of_capital,
                target_repayment = target_repayment
             )

def simulate_repayment(
            terms: ContractTerms,
            income_stream: np.ndarray
            ) -> RepaymentResult:
    cumulative = 0.0
    schedule: list[float] = []
    low_weeks = 0
    periods_to_repay = None
    typical_payment = terms.alpha * float(np.mean(income_stream))

    for t, y in enumerate(income_stream, start = 1):
        payment = min(terms.alpha * y, terms.r_cap)
        schedule.append(payment)
        cumulative += payment
        if payment < 0.5 * typical_payment:
            low_weeks += 1
        if cumulative >= terms.target_repayment and periods_to_repay is None:
            periods_to_repay = t
            break

    return RepaymentResult(
                terms = terms,
                schedule = schedule,
                periods_to_repay = periods_to_repay,
                total_repaid = cumulative,
                missed_or_low_weeks = low_weeks
            )

def compare_fixed_debt(
            principal: float,
            cost_of_capital: float,
            fixed_payment: float,
            income_stream: np.ndarray
            ) -> dict:
    target = principal * (1 + cost_of_capital)
    cumulative = 0.0
    missed = 0
    periods_to_repay = None
    for t, y in enumerate(income_stream, start = 1):
        can_pay = y >= fixed_payment
        payment = fixed_payment if can_pay else y
        if not can_pay:
            missed += 1
        cumulative += payment
        if cumulative >= target and periods_to_repay is None:
            periods_to_repay = t
            break

    return {
                "periods_to_repay": periods_to_repay,
                "missed_payments": missed,
                "total_repaid": cumulative
            }

