"""Numpy-based PCA utilities (no external dependencies required).

This module standardizes features, computes PCA via eigen-decomposition of the
covariance matrix, and emits publication-ready plots (scatter, variance, and
loadings). It deliberately avoids heavy dependencies to remain runnable in
restricted environments.
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
    Xz = (X - means) / stds
    return Xz, means, stds


def _pca_eig(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]
    return eigvals, eigvecs


def run_pca(
    df: pd.DataFrame, label_col: str, feature_cols: Iterable[str], outdir: Path
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run PCA without scikit-learn and save plots/tables."""

    feature_list = list(feature_cols)
    X = df[feature_list].values.astype(float)
    Xz, means, stds = _standardize(X)
    n_components = min(5, len(feature_list))
    eigvals, eigvecs = _pca_eig(Xz, n_components)

    pcs = Xz @ eigvecs
    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, columns=pc_cols)
    pca_df[label_col] = df[label_col].values

    total_var = eigvals.sum()
    evr = eigvals / total_var if total_var != 0 else np.zeros_like(eigvals)

    # Scatter PC1 vs PC2
    if pcs.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=label_col,
            alpha=0.7,
            s=40,
            edgecolor="none",
        )
        ax.set_title("PCA of indices (PC1 vs PC2)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(outdir / "pca_pc1_pc2_scatter.png", dpi=300)
        plt.close(fig)

    # Explained variance ratio
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(range(1, len(evr) + 1), evr * 100)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA explained variance")
    fig.tight_layout()
    fig.savefig(outdir / "pca_explained_variance.png", dpi=300)
    plt.close(fig)

    # Loadings heatmap (indices vs PCs)
    loadings = pd.DataFrame(
        eigvecs,
        index=feature_list,
        columns=pc_cols,
    )
    fig, ax = plt.subplots(
        figsize=(max(6, 0.5 * len(pc_cols)), max(5, 0.4 * len(feature_list)))
    )
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("PCA loadings (contribution of each index)")
    fig.tight_layout()
    fig.savefig(outdir / "pca_loadings_heatmap.png", dpi=300)
    plt.close(fig)

    loadings.to_csv(outdir / "pca_loadings.csv")
    pca_df.to_csv(outdir / "pca_scores.csv", index=False)

    return eigvals, pca_df, loadings
