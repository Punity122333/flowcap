from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from credit_engine.simulate import simulate_borrowers, save_dataset
from credit_engine.dml import fit_plr_dml, fit_naive_ols
from credit_engine.contract import make_contract, simulate_repayment, compare_fixed_debt


def cmd_simulate(args: argparse.Namespace) -> None:
    save_dataset(
        out_prefix=args.out,
        n=args.n,
        p=args.p,
        theta_true=args.theta_true,
        periods=args.periods,
        base_income=args.base_income,
        seed=args.seed,
    )
    print(f"Wrote {args.out}_features.csv and {args.out}_income.csv "
          f"({args.n} borrowers, {args.p} covariates, {args.periods} weeks)")


def cmd_estimate(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.features_csv)
    x_cols = [c for c in df.columns if c.startswith("X")]
    X = df[x_cols].to_numpy()
    D = df["D"].to_numpy()
    Y = df["Y"].to_numpy()

    naive = fit_naive_ols(D, Y)
    result = fit_plr_dml(X, D, Y, n_folds=args.folds, seed=args.seed)

    print("=== Naive estimate (Y ~ D, ignoring confounders X) ===")
    print(f"theta_naive = {naive: .5f}\n")
    print("=== Neyman-orthogonal Double ML (PLR, cross-fitted) ===")
    print(result.summary())

    if args.save_theta:
        with open(args.save_theta, "w") as f:
            f.write(str(result.theta))
        print(f"\nSaved theta_hat to {args.save_theta}")


def cmd_contract(args: argparse.Namespace) -> None:
    income = np.array([float(x) for x in args.income.split(",")])
    sigma_cf = float(np.std(income) / (np.mean(income) + 1e-9)) 

    terms = make_contract(
        theta_hat=args.theta,
        sigma_cf=sigma_cf,
        mean_income=float(np.mean(income)),
        principal=args.principal,
        cost_of_capital=args.gamma,
        cap_multiple=args.cap_multiple,
    )
    result = simulate_repayment(terms, income)

    print("=== Contract terms ===")
    print(f"alpha (income share)     = {terms.alpha:.3f}")
    print(f"R_cap (per-period cap)   = {terms.r_cap:.2f}")
    print(f"principal (K_bar)        = {terms.principal:.2f}")
    print(f"target repayment         = {terms.target_repayment:.2f}")
    print()
    print("=== Repayment simulation (income-sharing contract) ===")
    if result.periods_to_repay:
        print(f"Repaid in {result.periods_to_repay} periods "
              f"(total repaid: {result.total_repaid:.2f})")
    else:
        print(f"NOT fully repaid within {len(income)} periods "
              f"(total repaid: {result.total_repaid:.2f} / {terms.target_repayment:.2f})")
    print(f"Low-payment weeks (<50% of typical payment): {result.missed_or_low_weeks}")

    if args.compare_fixed:
        fixed_payment = terms.target_repayment / max(len(income), 1)
        baseline = compare_fixed_debt(
            principal=args.principal,
            cost_of_capital=args.gamma,
            fixed_payment=fixed_payment,
            income_stream=income,
        )
        print()
        print("=== Baseline: traditional fixed-repayment debt ===")
        print(f"fixed payment/period = {fixed_payment:.2f}")
        print(f"missed payments      = {baseline['missed_payments']}")
        if baseline["periods_to_repay"]:
            print(f"repaid in {baseline['periods_to_repay']} periods")
        else:
            print(f"NOT fully repaid within {len(income)} periods "
                  f"(total repaid: {baseline['total_repaid']:.2f})")


def cmd_pipeline(args: argparse.Namespace) -> None:
    print(f"[1/3] Simulating {args.n} synthetic borrowers "
          f"(true causal effect = {args.theta_true})...")
    df, income_panel = simulate_borrowers(
        n=args.n, p=args.p, theta_true=args.theta_true,
        periods=args.periods, seed=args.seed,
    )
    x_cols = [c for c in df.columns if c.startswith("X")]
    X = df[x_cols].to_numpy()
    D = df["D"].to_numpy()
    Y = df["Y"].to_numpy()

    print("[2/3] Estimating causal effect theta_0 via naive OLS and DML...")
    naive = fit_naive_ols(D, Y)
    result = fit_plr_dml(X, D, Y, n_folds=args.folds, seed=args.seed)
    print(f"      true theta   = {args.theta_true: .5f}")
    print(f"      naive theta  = {naive: .5f}  (bias = {naive - args.theta_true: .5f})")
    print(f"      DML theta    = {result.theta: .5f}  "
          f"(bias = {result.theta - args.theta_true: .5f}, se = {result.se:.5f})")

    borrower_idx = args.borrower_id if args.borrower_id is not None else 0
    income = income_panel[borrower_idx]
    sigma_cf = float(np.std(income) / (np.mean(income) + 1e-9))

    print(f"\n[3/3] Building income-sharing contract for borrower {borrower_idx} "
          f"(income CV = {sigma_cf:.3f})...")
    terms = make_contract(
        theta_hat=result.theta,
        sigma_cf=sigma_cf,
        mean_income=float(np.mean(income)),
        principal=args.principal,
        cost_of_capital=args.gamma,
    )
    rep = simulate_repayment(terms, income)
    fixed_payment = terms.target_repayment / len(income)
    baseline = compare_fixed_debt(args.principal, args.gamma, fixed_payment, income)

    print(f"      alpha = {terms.alpha:.3f}, R_cap = {terms.r_cap:.2f}, "
          f"target repayment = {terms.target_repayment:.2f}")
    print(f"      Income-sharing contract: "
          f"{'repaid in ' + str(rep.periods_to_repay) + ' periods' if rep.periods_to_repay else 'not repaid in horizon'}, "
          f"{rep.missed_or_low_weeks} low-payment weeks")
    print(f"      Fixed-debt baseline:    "
          f"{'repaid in ' + str(baseline['periods_to_repay']) + ' periods' if baseline['periods_to_repay'] else 'not repaid in horizon'}, "
          f"{baseline['missed_payments']} missed payments")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-cli",
        description="Causal-ML income-sharing credit underwriting toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="Generate synthetic borrower + income data")
    p_sim.add_argument("--out", default="data/synthetic", help="Output file prefix")
    p_sim.add_argument("--n", type=int, default=2000, help="Number of borrowers")
    p_sim.add_argument("--p", type=int, default=10, help="Number of covariates")
    p_sim.add_argument("--theta-true", type=float, default=-0.35,
                        help="True causal effect used to generate data")
    p_sim.add_argument("--periods", type=int, default=26, help="Weeks of income history")
    p_sim.add_argument("--base-income", type=float, default=100.0)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.set_defaults(func=cmd_simulate)

    p_est = sub.add_parser("estimate", help="Estimate theta_0 via Double ML")
    p_est.add_argument("features_csv", help="Path to *_features.csv from `simulate`")
    p_est.add_argument("--folds", type=int, default=5)
    p_est.add_argument("--seed", type=int, default=42)
    p_est.add_argument("--save-theta", default=None,
                        help="Optional path to save theta_hat as plain text")
    p_est.set_defaults(func=cmd_estimate)

    p_con = sub.add_parser("contract", help="Build & simulate an income-sharing contract")
    p_con.add_argument("--theta", type=float, required=True,
                        help="theta_hat from `estimate`")
    p_con.add_argument("--income", type=str, required=True,
                        help="Comma-separated income stream, e.g. '90,110,40,130,...'")
    p_con.add_argument("--principal", type=float, default=1000.0)
    p_con.add_argument("--gamma", type=float, default=0.18, help="Cost of capital")
    p_con.add_argument("--cap-multiple", type=float, default=2.2)
    p_con.add_argument("--compare-fixed", action="store_true",
                        help="Also simulate an equivalent fixed-repayment loan")
    p_con.set_defaults(func=cmd_contract)

    p_pipe = sub.add_parser("pipeline", help="Run simulate -> estimate -> contract end-to-end")
    p_pipe.add_argument("--n", type=int, default=2000)
    p_pipe.add_argument("--p", type=int, default=10)
    p_pipe.add_argument("--theta-true", type=float, default=-0.35)
    p_pipe.add_argument("--periods", type=int, default=26)
    p_pipe.add_argument("--folds", type=int, default=5)
    p_pipe.add_argument("--seed", type=int, default=42)
    p_pipe.add_argument("--borrower-id", type=int, default=None)
    p_pipe.add_argument("--principal", type=float, default=1000.0)
    p_pipe.add_argument("--gamma", type=float, default=0.18)
    p_pipe.set_defaults(func=cmd_pipeline)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
