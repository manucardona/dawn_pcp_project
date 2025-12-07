# pcp.py

import numpy as np

def soft_threshold(X, tau):
    """
    Apply element-wise soft-thresholding to a matrix or vector.

    Soft-thresholding is the proximal operator of the l1 norm and is defined as:
        S_tau(x) = sign(x) * max(|x| − tau, 0)

    This operation shrinks values toward zero and sets them exactly to zero
    when |x| <= tau. It is commonly used in optimization algorithms involving
    sparsity penalties, such as LASSO, Robust PCA / PCP, and compressed sensing.
    """
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def singular_value_thresholding(X, tau):
    """
    Apply Singular Value Thresholding (SVT) to a matrix.

    SVT computes the proximal operator of the nuclear norm. Given a matrix X,
    the SVT operator solves the optimization problem:

        minimize_M   0.5 * ||X - M||_F^2  +  tau * ||M||_*

    where ||M||_* denotes the nuclear norm (sum of singular values).
    The solution is obtained by performing SVD on X and shrinking each singular
    value by `tau`:

        X = U diag(s) V^T
        M* = U diag(max(s - tau, 0)) V^T

    This operation promotes low-rank structure by reducing or eliminating small
    singular values. It is a key step in algorithms for Robust PCA /
    Principal Component Pursuit (PCP), matrix completion, and related convex
    optimization problems.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    rank_eff = np.sum(s_thresh > 0)
    return U @ np.diag(s_thresh) @ Vt, s, s_thresh, rank_eff

def pcp_inexact_alm(
    X,
    lam=None,
    mu=None,
    max_iters=1000,
    tol=1e-7,
    verbose=False,
):
    """
    Principal Component Pursuit via inexact ALM (Candès et al. 2011).

    Parameters
    X : ndarray, shape (m, n)
        Input data matrix (e.g., grayscale image).
    lam : float or None
        Sparsity weight lambda. Defaults to 1/sqrt(max(m, n)).
    mu : float or None
        Augmented Lagrangian parameter. If None, chosen heuristically.
    max_iters : int
        Maximum number of iterations.
    tol : float
        Relative tolerance on ||X - L - S||_F / ||X||_F.
    verbose : bool
        If True, print progress.

    Returns
    L : ndarray
        Low-rank component.
    S : ndarray
        Sparse component.
    info : dict
        Contains convergence info (errors, ranks, iterations, etc.).
    """
    X = X.astype(np.float64)
    m, n = X.shape

    # Default lambda
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    # Norm of X
    norm_X = np.linalg.norm(X, 'fro')

    # Initialize variables
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)

    # mu parameter heuristic (from literature / robust PCA code)
    if mu is None:
        mu = (m * n) / (4.0 * np.linalg.norm(X, 1))
        # Guard against pathologies
        if not np.isfinite(mu) or mu <= 0:
            mu = 1.0

    mu_bar = mu * 1e7    # upper bound to avoid explosion
    rho = 1.5            # growth factor for mu

    errors = []
    ranks = []
    sparsities = []

    for it in range(max_iters):
        # 1) Update L via SVT
        temp_L = X - S + (1.0 / mu) * Y
        L, s, s_thresh, rank_eff = singular_value_thresholding(temp_L, 1.0 / mu)

        # 2) Update S via soft-thresholding
        temp_S = X - L + (1.0 / mu) * Y
        S = soft_threshold(temp_S, lam / mu)

        # 3) Dual variable update
        residual = X - L - S
        Y = Y + mu * residual

        # Convergence metrics
        err = np.linalg.norm(residual, 'fro') / (norm_X + 1e-12)
        errors.append(err)
        ranks.append(rank_eff)
        sparsities.append(np.mean(np.abs(S) > 1e-6))

        if verbose and (it % 10 == 0 or it == 0 or err < tol):
            print(f"[iter {it:4d}] rel_error={err:.3e}, rank(L)={rank_eff}, sparsity(S)={sparsities[-1]:.3f}")

        if err < tol:
            break

        # Increase mu (inexact ALM strategy)
        mu = min(mu * rho, mu_bar)

    info = {
        "errors": np.array(errors),
        "ranks": np.array(ranks),
        "sparsities": np.array(sparsities),
        "iterations": it + 1,
        "final_error": errors[-1],
        "lambda": lam,
    }
    return L, S, info
