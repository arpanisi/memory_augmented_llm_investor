"""
Ledoit-Wolf constant-correlation covariance shrinkage derivation.
Fully compliant with Step 5 specification in coding-plan.md.
"""
import numpy as np

def compute_ledoit_wolf_constant_correlation(returns_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Computes Ledoit-Wolf constant-correlation shrunk covariance matrix.
    returns_matrix: shape (T, N) where T = 252 days, N = number of assets.
    Returns: (Sigma_shrunk, delta)
    """
    T, N = returns_matrix.shape
    if T < 2 or N < 1:
        raise ValueError("Returns matrix must have at least 2 days and 1 asset.")

    # 1. Sample covariance matrix S
    S = np.cov(returns_matrix, rowvar=False, bias=False)
    if N == 1:
        return S.reshape((1, 1)), 0.0

    s = np.sqrt(np.diag(S))
    s_outer = np.outer(s, s)
    s_outer_safe = np.where(s_outer == 0, 1e-12, s_outer)
    R = S / s_outer_safe

    # 2. Average pairwise correlation r_bar
    upper_tri_indices = np.triu_indices(N, k=1)
    r_bar = float(np.mean(R[upper_tri_indices])) if len(upper_tri_indices[0]) > 0 else 0.0

    # 3. Target matrix F
    F = r_bar * s_outer
    np.fill_diagonal(F, np.diag(S))

    # Demeaned returns
    X = returns_matrix - np.mean(returns_matrix, axis=0)

    # 4. Compute pi_hat
    pi_hat_mat = np.zeros((N, N))
    for t in range(T):
        x_t = X[t, :]
        dev = np.outer(x_t, x_t) - S
        pi_hat_mat += dev ** 2
    pi_hat_mat /= T
    pi_hat = float(np.sum(pi_hat_mat))

    # 5. Compute gamma_hat
    gamma_hat = float(np.sum((F - S) ** 2))

    # Degenerate case check
    if gamma_hat < 1e-12:
        return S, 0.0

    # 6. Compute rho_hat
    rho_hat = float(np.sum(np.diag(pi_hat_mat)))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                s_i = max(s[i], 1e-12)
                s_j = max(s[j], 1e-12)
                
                # theta_ii_ij
                dev_ii = X[:, i] ** 2 - S[i, i]
                dev_ij = X[:, i] * X[:, j] - S[i, j]
                theta_ii_ij = float(np.mean(dev_ii * dev_ij))
                
                # theta_jj_ij
                dev_jj = X[:, j] ** 2 - S[j, j]
                theta_jj_ij = float(np.mean(dev_jj * dev_ij))
                
                term = (r_bar / 2.0) * ((s_j / s_i) * theta_ii_ij + (s_i / s_j) * theta_jj_ij)
                rho_hat += term

    # 7. Compute shrinkage intensity delta
    delta = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / float(T)))

    # 8. Shrunk covariance matrix
    Sigma_shrunk = delta * F + (1.0 - delta) * S
    return Sigma_shrunk, delta
