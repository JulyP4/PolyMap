from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.style.use("ggplot")


def run_pca(
    df: pd.DataFrame, label_col: str, feature_cols: Iterable[str], outdir: Path
) -> Tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Run PCA on standardized indices and save plots and tables."""

    feature_list = list(feature_cols)
    X = df[feature_list].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=min(5, len(feature_list)))
    pcs = pca.fit_transform(Xz)

    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, columns=pc_cols)
    pca_df[label_col] = df[label_col].values

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
    evr = pca.explained_variance_ratio_
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
        pca.components_.T,
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

    return pca, pca_df, loadings
