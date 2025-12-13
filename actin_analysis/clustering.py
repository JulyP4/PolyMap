"""Unsupervised clustering of actin index profiles without sklearn.

Uses a light PCA implementation (variance-maximizing eigenvectors) and a simple
k-means routine to discover clusters in index space. Figures compare cluster
assignments to existing labels.
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


def _pca_scores(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvecs = eigvecs[:, order]
    scores = X @ eigvecs
    return scores, eigvecs


def _kmeans(X: np.ndarray, k: int, iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.vstack([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids
    return labels, centroids


def cluster_states(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Iterable[str],
    outdir: Path,
    n_clusters: int = 4,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run PCA + k-means, save cluster assignments and QC figures."""

    feature_list = list(feature_cols)
    X = df[feature_list].values.astype(float)
    Xz, _, _ = _standardize(X)
    n_components = min(5, len(feature_list))
    scores, eigvecs = _pca_scores(Xz, n_components)

    pc_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    pca_df = pd.DataFrame(scores, columns=pc_cols)
    pca_df[label_col] = df[label_col].values

    clusters, centroids = _kmeans(scores, n_clusters)
    pca_df["cluster"] = clusters
    pca_df.to_csv(outdir / "cluster_assignments.csv", index=False)

    if scores.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="cluster",
            palette="tab10",
            s=45,
            ax=ax,
        )
        ax.set_title("PCA space colored by k-means clusters")
        fig.tight_layout()
        fig.savefig(outdir / "clusters_pc1_pc2.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=label_col,
            s=45,
            ax=ax,
        )
        ax.set_title("PCA space colored by existing labels")
        fig.tight_layout()
        fig.savefig(outdir / "labels_pc1_pc2.png", dpi=300)
        plt.close(fig)

    # Compactness metric
    compactness = np.mean(
        [np.linalg.norm(scores[clusters == i] - centroids[i], axis=1).mean() for i in range(n_clusters)]
    )
    pd.DataFrame({"compactness": [compactness]}).to_csv(
        outdir / "cluster_compactness.csv", index=False
    )

    loadings = pd.DataFrame(eigvecs, index=feature_list, columns=pc_cols)
    loadings.to_csv(outdir / "pca_component_loadings.csv")

    return pca_df, centroids
