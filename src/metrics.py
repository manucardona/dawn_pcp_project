# metrics.py
import numpy as np
from pcp import pcp_inexact_alm

def reconstruction_error(X, L, S, norm='fro'):
    """
    Compute the relative reconstruction error of a Robust PCA decomposition.

    Given an observed matrix X and its decomposition into a low-rank component L
    and a sparse component S, this function computes:

        error = ||X − L − S|| / ||X||

    using either the Frobenius norm or the ℓ₂ norm. The result measures how well
    the decomposition reconstructs the original image or data matrix.

    Parameters
    X : ndarray
        Original data matrix.
    L : ndarray
        Low-rank component obtained from PCP or related decomposition.
    S : ndarray
        Sparse component.
    norm : {'fro', 'l2'}, optional
        Norm used to compute the error:
        - 'fro' : Frobenius norm (default)
        - 'l2'  : ℓ₂ norm applied to the vectorized matrix

    Returns
    float
        Relative reconstruction error in the chosen norm. A value close to zero
        indicates a good decomposition.
    """
    residual = X - L - S
    if norm == 'fro':
        num = np.linalg.norm(residual, 'fro')
        den = np.linalg.norm(X, 'fro') + 1e-12
    elif norm == 'l2':
        num = np.linalg.norm(residual.ravel(), 2)
        den = np.linalg.norm(X.ravel(), 2) + 1e-12
    else:
        raise ValueError("Unknown norm type")
    return num / den

import numpy as np

def sparse_energy(X, S):
    """
    Relative energy of the sparse component:
        ||S||_F / ||X||_F
    """
    num = np.linalg.norm(S, "fro")
    den = np.linalg.norm(X, "fro") + 1e-12
    return num / den


def sparsity(S, threshold=1e-3):
    """
    Fraction of pixels where |S_ij| > threshold.
    """
    return np.mean(np.abs(S) > threshold)


def effective_rank_from_singular_values(s, threshold=1e-2):
    """
    Count how many singular values are >= threshold * max(s).
    """
    if len(s) == 0:
        return 0
    return int(np.sum(s >= threshold * np.max(s)))


def singular_value_spectrum(X):
    """
    Return singular values of X, sorted descending.
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    return s

def summarize_category(images, pcp_kwargs=None):
    """
    Run PCP on each image in a list and aggregate stats.

    Returns
    -------
    results : list of dict
        For each image: {'L', 'S', 'info', 'sv_original', 'sv_L', 'recon_error', ...}
    """
    if pcp_kwargs is None:
        pcp_kwargs = {}

    results = []
    for i, X in enumerate(images):
        L, S, info = pcp_inexact_alm(X, **pcp_kwargs)
        err = reconstruction_error(X, L, S)
        sv_X = singular_value_spectrum(X)
        sv_L = singular_value_spectrum(L)

        results.append({
            "X": X,
            "L": L,
            "S": S,
            "info": info,
            "sv_X": sv_X,
            "sv_L": sv_L,
            "recon_error": err,
        })

    return results
