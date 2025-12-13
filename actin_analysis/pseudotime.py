"""Pseudotime-style ordering using lightweight PCA-based embedding.

In constrained environments (no sklearn), we approximate diffusion-style
pseudotime by projecting data onto the first two PCA axes and min–max scaling
the first axis to [0, 1]. This retains a continuous ordering for downstream
visualization.
"""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    return (X - means) / stds, means, stds


def _pca_2d(X: np.ndarray) -> np.ndarray:
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:2]
    eigvecs = eigvecs[:, order]
    return X @ eigvecs


def compute_pseudotime(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Iterable[str],
    outdir: Path,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Embed cells and assign a simple PCA-based pseudotime coordinate."""

    feature_list = list(feature_cols)
    X = df[feature_list].values.astype(float)
    Xz, _, _ = _standardize(X)
    emb = _pca_2d(Xz)

    # Avoid degenerate case
    if emb.shape[1] == 1:
        emb = np.hstack([emb, np.zeros((emb.shape[0], 1))])

    # NumPy 2.0 removed ndarray.ptp; use the functional form for compatibility.
    pt_range = np.ptp(emb[:, 0])
    pt = (emb[:, 0] - emb[:, 0].min()) / (pt_range + 1e-9)
    emb_df = pd.DataFrame(emb, columns=["PC1", "PC2"])
    emb_df["pseudotime"] = pt
    emb_df[label_col] = df[label_col].values
    emb_df.to_csv(outdir / "pseudotime_embedding.csv", index=False)

    # Scatter colored by label
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=emb_df, x="PC1", y="PC2", hue=label_col, s=45, ax=ax)
    ax.set_title("PCA embedding colored by label")
    fig.tight_layout()
    fig.savefig(outdir / "embedding_by_label.png", dpi=300)
    plt.close(fig)

    # Scatter colored by pseudotime
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(emb_df["PC1"], emb_df["PC2"], c=pt, cmap="viridis", s=45)
    ax.set_title("PCA embedding colored by pseudotime")
    fig.colorbar(scatter, ax=ax, label="pseudotime")
    fig.tight_layout()
    fig.savefig(outdir / "embedding_by_pseudotime.png", dpi=300)
    plt.close(fig)

    # Trajectories of each index along pseudotime
    ordered_idx = np.argsort(pt)
    for feat in feature_list:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(pt[ordered_idx], df[feat].values[ordered_idx], marker="o", linestyle="-")
        ax.set_xlabel("pseudotime (scaled PC1)")
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} vs pseudotime")
        fig.tight_layout()
        fig.savefig(outdir / f"{feat}_pseudotime.png", dpi=300)
        plt.close(fig)

    return emb_df, pt
