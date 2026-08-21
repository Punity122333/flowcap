from __future__ import annotations

import numpy as np

import pandas as pd

def _g0(X: np.ndarray) -> np.ndarray:
    return (
            0.6 * np.sin(X[:, 0]) + 0.4 * X[:, 1] ** 2 - 0.3 * X[:, 2] * X[:, 3] + 0.2 * np.log1p(np.abs(X[:, 4]))
            )

def _m0(X: np.ndarray) -> np.ndarray:
    return (
        0.5 * X[:, 0]
        + 0.3 * np.cos(X[:, 1])
        + 0.2 * X[:, 2]
        - 0.1 * X[:, 3] ** 2
    )


def simulate_borrowers(
        n: int = 2000,
        p: int = 10,
        theta_true: float = -0.35,
        periods: int = 26,
        base_income: float = 100.0,
        seed: int = 42,
        ) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    m0_X = _m0(X)
    V = rng.normal(scale=1.0, size=n)
    D = m0_X + V
    g0_X = _g0(X)
    U = rng.normal(scale=1.0, size=n)
    Y = theta_true * D + g0_X + U
    
    cols = {f"X{j}": X[:, j] for j in range(p)}
    df = pd.DataFrame(cols)
    df.insert(0, "borrower_id", np.arange(n))
    df["D"] = D
    df["Y"] = Y
    volatility = 0.25 + 0.20 * (1 / (1 + np.exp(-df["Y"].to_numpy())))
    drift = base_income * (1 + 0.05 * np.tanh(df["X0"].to_numpy()))
    shocks = rng.normal(0, 1, size=(n, periods))
    income_panel = drift[:, None] * np.exp(
                volatility[:, None] * shocks - 0.5 * volatility[:, None] ** 2
            )
    dropout_mask = rng.random(size=(n, periods)) < 0.06
    income_panel = np.where(dropout_mask, income_panel * 0.1, income_panel)
    
    return df, income_panel
    

def save_dataset(
            out_prefix: str,
            n: int,
            p: int,
            theta_true: float,
            periods: int,
            base_income: float,
            seed: int
        ) -> None:
    df, income_panel = simulate_borrowers(
                n = n,
                p = p,
                theta_true = theta_true,
                periods = periods,
                base_income = base_income,
                seed = seed,
            )
    df.to_csv(f"{out_prefix}_features.csv", index = False)
    income_df = pd.DataFrame(
                income_panel, columns = [f"week_{t + 1}" for t in range(periods)]
            )
    income_df.insert(0, "borrower_id", df["borrower_id"])
    income_df.to_csv(f"{out_prefix}_income.csv", index = False)
