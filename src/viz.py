# viz.py

import matplotlib.pyplot as plt
import numpy as np

def show_decomposition(X, L, S, clim=(0, 1), cmap='gray', title_prefix=""):
    """
    Plot original, low-rank, sparse components side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, img, title in zip(
        axes,
        [X, L, S],
        ["Original", "Low-rank L", "Sparse S"]
    ):
        im = ax.imshow(img, cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax.set_title(f"{title_prefix}{title}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_singular_value_spectrum(s_values_list, labels, log_scale=True, title="Singular value spectra"):
    """
    Plot average or representative singular value curves for each category.

    s_values_list : list of 1D arrays
    labels        : list of str
    """
    plt.figure(figsize=(6, 4))
    for s, label in zip(s_values_list, labels):
        plt.plot(s, label=label)

    if log_scale:
        plt.yscale('log')
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_vs_rank(ranks, errors, label=None, title="Reconstruction error vs. rank(L)"):
    plt.figure(figsize=(6, 4))
    plt.scatter(ranks, errors, alpha=0.6, label=label)
    plt.xlabel("rank(L)")
    plt.ylabel("Reconstruction error")
    plt.title(title)
    if label is not None:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_decomposition(
    X,
    L,
    S,
    sparse_threshold=1e-3,
    cmap_main="gray",
    figsize=(14, 6),
    title_prefix="",
):
    """
    Rich visualization of PCP decomposition:
    - Original image
    - Low-rank component L
    - Raw sparse component S (gray)
    - Sparse S rescaled to [0,1]
    - Sparse S with tight symmetric range (seismic)
    - Original image with sparse-mask overlay

    Parameters
    X, L, S : 2D ndarrays
        Original image, low-rank, and sparse components.
    sparse_threshold : float
        Threshold for creating the sparse mask: |S| > sparse_threshold.
    cmap_main : str
        Colormap for X and L.
    figsize : tuple
        Figure size.
    title_prefix : str
        Optional prefix to prepend to all subplot titles.
    """
    X = np.asarray(X)
    L = np.asarray(L)
    S = np.asarray(S)

    # Basic stats for S
    max_abs_S = np.max(np.abs(S)) + 1e-12
    mask = np.abs(S) > sparse_threshold

    # S rescaled to [0,1] for grayscale visualization
    S_min = S.min()
    S_max = S.max()
    S_rescaled = (S - S_min) / (S_max - S_min + 1e-12)

    # Choose a tighter range around 0 for seismic view
    v = 0.5 * max_abs_S  # you can tweak this factor
    v = max(v, 1e-6)     # avoid zero range

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Row 1: Original, Low-rank, Raw Sparse (gray)
    ax = axes[0, 0]
    im = ax.imshow(X, cmap=cmap_main)
    ax.set_title(f"{title_prefix}Original X")
    ax.axis("off")

    ax = axes[0, 1]
    im = ax.imshow(L, cmap=cmap_main)
    ax.set_title(f"{title_prefix}Low-rank L")
    ax.axis("off")

    ax = axes[0, 2]
    im = ax.imshow(S, cmap="gray")
    ax.set_title(f"{title_prefix}Sparse S (raw gray)")
    ax.axis("off")

    # Row 2: S rescaled, S seismic, Original + mask
    ax = axes[1, 0]
    im = ax.imshow(S_rescaled, cmap="gray")
    ax.set_title(f"{title_prefix}Sparse S (rescaled [0,1])")
    ax.axis("off")

    ax = axes[1, 1]
    im = ax.imshow(S, cmap="seismic", vmin=-v, vmax=v)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{title_prefix}Sparse S (seismic, tight range)")
    ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(X, cmap=cmap_main)
    ax.imshow(mask, cmap="autumn", alpha=0.5)
    ax.set_title(f"{title_prefix}X with sparse mask (|S|>{sparse_threshold:g})")
    ax.axis("off")

    plt.tight_layout()
    plt.show()
