from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

@dataclass
class DMLResult:
    theta: float
    se: float
    ci_low: float
    ci_high: float
    n_obs: float
    n_folds: float

    def summary(self) -> str:
        return (
                    f"theta_hat = {self.theta: .5f}\n"
                    f"std. error = {self.se: .5f}\n"
                    f"95% CI = {self.ci_low: .5f}, {self.ci_high: .5f}\n"
                    f"n_obs / n_folds = {self.n_obs} / {self.n_folds}"
                )

def _cross_fit_predict(model, X: np.ndarray, target: np.ndarray, folds: KFold) -> np.ndarray:
    preds = np.zeros_like(target, dtype=float)
    for train_idx, test_idx in folds.split(X):
        m = clone(model)
        m.fit(X[train_idx], target[train_idx])
        preds[test_idx] = m.predict(X[test_idx])

    return preds

def fit_plr_dml(
            X: np.ndarray,
            D: np.ndarray,
            Y: np.ndarray,
            n_folds: int = 5,
            seed: int = 42,
            outcome_model = None,
            treatment_model = None,
            ) -> DMLResult:
    n = X.shape[0]
    folds = KFold(n_splits = n_folds, shuffle = True, random_state = seed)

    if outcome_model is None:
        outcome_model = GradientBoostingRegressor(
                    n_estimators = 300,
                    max_depth = 3,
                    learning_rate = 0.05,
                    random_state = seed
                )

    if treatment_model is None:
        treatment_model = RandomForestRegressor(
                    n_estimators = 400,
                    max_depth = 6,
                    random_state = seed,
                    n_jobs = -1
                )

    g_hat = _cross_fit_predict(outcome_model, X, Y, folds)
    Y_tilde = Y - g_hat
    m_hat = _cross_fit_predict(treatment_model, X, D, folds)
    D_tilde = D - m_hat

    denom = float(np.mean(D_tilde ** 2))
    theta_hat = float(np.mean(D_tilde * Y_tilde) / denom)

    scores = (Y_tilde - theta_hat * D_tilde) * D_tilde
    sigma2 = float(np.mean(scores ** 2)) / (denom ** 2)
    se = float(np.sqrt(sigma2 / n))

    z = 1.959963984540054

    return DMLResult(
                theta = theta_hat,
                se = se,
                ci_low = theta_hat - z * se,
                ci_high = theta_hat + z * se,
                n_obs = n,
                n_folds = n_folds
            )

def fit_naive_ols(D: np.ndarray, Y: np.ndarray) -> float:
    D = D.reshape(-1, 1)
    D1 = np.hstack([np.ones_like(D), D])
    beta, *_ = np.linalg.lstsq(D1, Y, rcond = None)
    return float(beta[1])







