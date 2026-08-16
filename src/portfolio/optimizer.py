"""
Convex Portfolio Optimizer using cvxpy.
Enforces direction/conviction bounds, budget constraint ||w||_1 <= 1, and 14 Barra factor bounds.
"""
import numpy as np
import cvxpy as cp
from config.settings import INDUSTRY_FACTOR_BOUND, STYLE_FACTOR_BOUND
from src.portfolio.shrinkage import compute_ledoit_wolf_constant_correlation

def solve_portfolio_optimization(
    asset_names: list[str],
    mean_returns: np.ndarray,
    Sigma_shrunk: np.ndarray,
    views: list[dict],
    factor_exposures: np.ndarray,  # Shape (N, 14)
    risk_aversion_lambda: float = 1.0
) -> np.ndarray:
    """
    Solves convex optimization:
        Maximize mu^T w - lambda * w^T Sigma_shrunk w
        s.t.
        direction & conviction bounds per asset
        ||w||_1 <= 1
        | sum_i w_i X_{i, k} | <= bound_k for 14 Barra factors
    """
    N = len(asset_names)
    if N == 0:
        return np.array([])

    # Check edge case: all assets flat
    all_flat = all(v.get("direction", "flat") == "flat" for v in views)
    if all_flat:
        return np.zeros(N)

    w = cp.Variable(N)
    
    # 1. Objective: mu^T w - lambda * w^T Sigma w
    # Ensure Sigma is positive semi-definite
    Sigma_psd = cp.psd_wrap(Sigma_shrunk)
    objective = cp.Maximize(mean_returns @ w - risk_aversion_lambda * cp.quad_form(w, Sigma_psd))

    constraints = []

    # 2. Direction & Conviction Bounds
    for i, v in enumerate(views):
        direction = v.get("direction", "flat").lower()
        conviction = float(v.get("conviction", 0.0))
        conviction = max(0.0, min(1.0, conviction))

        if direction == "long":
            constraints.append(w[i] >= 0.0)
            constraints.append(w[i] <= conviction)
        elif direction == "short":
            constraints.append(w[i] >= -conviction)
            constraints.append(w[i] <= 0.0)
        else:  # flat
            constraints.append(w[i] == 0.0)

    # 3. Budget Constraint: ||w||_1 <= 1
    constraints.append(cp.norm(w, 1) <= 1.0)

    # 4. Barra Factor Exposure Constraints (14 factors)
    if factor_exposures is not None and factor_exposures.shape == (N, 14):
        # 10 Industry Division factors (index 0..9), 4 Style factors (index 10..13)
        for k in range(14):
            X_k = factor_exposures[:, k]
            b_k = INDUSTRY_FACTOR_BOUND if k < 10 else STYLE_FACTOR_BOUND
            constraints.append(cp.abs(X_k @ w) <= b_k)

    # Solve problem using cvxpy
    prob = cp.Problem(objective, constraints)
    
    # Try solvers in order of preference
    for solver in [cp.CLARABEL, cp.OSQP, cp.ECOS, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                return np.asarray(w.value, dtype=float)
        except Exception:
            continue

    # Fallback to zero weights if solver fails
    return np.zeros(N)
