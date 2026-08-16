"""
Portfolio Breadth and Realized Factor Exposure reporting module.
Fully compliant with Step 6 specification in coding-plan.md.
"""
import numpy as np

def compute_portfolio_breadth(w: np.ndarray, Sigma_shrunk: np.ndarray) -> float:
    """
    Computes portfolio breadth = 1 / (w_hat^T R_shrunk w_hat)
    where w_hat = w / ||w||_1.
    On an all-cash day (||w||_1 == 0), returns None (undefined).
    """
    w_arr = np.asarray(w, dtype=float)
    gross_exp = np.sum(np.abs(w_arr))
    if gross_exp == 0.0:
        return None

    w_hat = w_arr / gross_exp

    # Derives correlation matrix R_shrunk from Sigma_shrunk
    diag_std = np.sqrt(np.diag(Sigma_shrunk))
    diag_std_safe = np.where(diag_std == 0, 1e-12, diag_std)
    outer_std = np.outer(diag_std_safe, diag_std_safe)
    R_shrunk = Sigma_shrunk / outer_std

    denom = float(w_hat.T @ R_shrunk @ w_hat)
    if denom <= 0:
        return None
    return float(1.0 / denom)

def compute_realized_factor_exposures(w: np.ndarray, factor_exposures: np.ndarray) -> dict[str, float]:
    """
    Computes realized factor exposures E_k = sum_i w_i X_{i, k} for the 14 Barra factors.
    factor_exposures shape: (N, 14).
    """
    w_arr = np.asarray(w, dtype=float)
    N = len(w_arr)
    
    if factor_exposures is None or factor_exposures.shape != (N, 14):
        return {}

    factor_names = [
        "Ind_A", "Ind_B", "Ind_C", "Ind_D", "Ind_E",
        "Ind_F", "Ind_G", "Ind_H", "Ind_I", "Ind_J",
        "Style_Size", "Style_Value", "Style_Momentum", "Style_Volatility"
    ]

    realized = {}
    for k, name in enumerate(factor_names):
        val = float(np.dot(w_arr, factor_exposures[:, k]))
        realized[name] = val

    return realized

def compute_peer_concentration_exposure(
    w: np.ndarray,
    asset_names: list[str],
    traded_equities: list[str],
    visible_edges: list[dict],
) -> dict[str, float]:
    """
    Peer-concentration exposure diagnostic (Step 9).

    For each traded equity, returns the sum of current portfolio weight held in every OTHER traded
    equity connected to it by a competitor edge, counting edges in EITHER direction (competitor is
    symmetric: both this equity's own outgoing competitor edges and any other equity's incoming
    competitor edge naming this one count).

    `visible_edges` must already be point-in-time-filtered for the decision day. This is a reported
    diagnostic only and must not become a hard constraint in the portfolio optimizer.
    """
    w_arr = np.asarray(w, dtype=float)
    index_of = {name: i for i, name in enumerate(asset_names)}
    traded = set(traded_equities)

    # Build the symmetric competitor adjacency among traded equities from the visible edges.
    competitor_pairs = set()
    for e in visible_edges:
        if e.get("relationship_type") != "competitor":
            continue
        s, tgt = e.get("source"), e.get("target")
        if s in traded and tgt in traded and s != tgt:
            competitor_pairs.add((s, tgt))
            competitor_pairs.add((tgt, s))

    result = {}
    for eq in traded_equities:
        if eq not in index_of:
            result[eq] = 0.0
            continue
        peers = {t for (s, t) in competitor_pairs if s == eq}
        result[eq] = float(sum(w_arr[index_of[p]] for p in peers if p in index_of))
    return result
