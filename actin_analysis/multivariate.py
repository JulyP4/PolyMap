"""Multivariate group comparison without external dependencies.

Implements MANOVA-style Pillai trace and Linear Discriminant Analysis using
pure NumPy. These tools assess whether labels differ in the multi-index space
and visualize the axes that best separate them.
"""
"""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")


def _centered_group_arrays(
    df: pd.DataFrame, label_col: str, feature_cols: Iterable[str]
) -> Tuple[np.ndarray, dict]:
    X = df[list(feature_cols)].values.astype(float)
    labels = df[label_col].values
    unique = np.unique(labels)
    groups = {}
    for lab in unique:
        groups[lab] = X[labels == lab]
    return X, groups


def run_manova(
    df: pd.DataFrame, label_col: str, feature_cols: Iterable[str], outdir: Path
) -> pd.DataFrame:
    """Compute Pillai's trace to test multivariate label differences."""

    X, groups = _centered_group_arrays(df, label_col, feature_cols)
    overall_mean = X.mean(axis=0)

    # Between-group (H) and within-group (E) scatter matrices
    H = np.zeros((X.shape[1], X.shape[1]))
    E = np.zeros_like(H)
    for lab, arr in groups.items():
        n_g = arr.shape[0]
        mean_g = arr.mean(axis=0)
        diff_mean = (mean_g - overall_mean).reshape(-1, 1)
        H += n_g * (diff_mean @ diff_mean.T)
        centered = arr - mean_g
        E += centered.T @ centered

    HE_inv = np.linalg.pinv(H + E)
    pillai_trace = np.trace(HE_inv @ H)
    pillai_df = pd.DataFrame({"pillai_trace": [pillai_trace]})
    pillai_df.to_csv(outdir / "manova_pillai_trace.csv", index=False)

    with (outdir / "manova_summary.txt").open("w", encoding="utf-8") as f:
        f.write("MANOVA-style test (Pillai trace)\n")
        f.write(f"Groups: {list(groups.keys())}\n")
        f.write(f"Pillai trace: {pillai_trace:.4f}\n")

    return pillai_df


def run_lda(
    df: pd.DataFrame, label_col: str, feature_cols: Iterable[str], outdir: Path
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fisher-style Linear Discriminant Analysis via scatter matrices."""

    feature_list = list(feature_cols)
    X, groups = _centered_group_arrays(df, label_col, feature_list)
    labels = df[label_col].values
    overall_mean = X.mean(axis=0)

    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros_like(Sw)
    for lab, arr in groups.items():
        mean_g = arr.mean(axis=0)
        centered = arr - mean_g
        Sw += centered.T @ centered
        diff_mean = (mean_g - overall_mean).reshape(-1, 1)
        Sb += arr.shape[0] * (diff_mean @ diff_mean.T)

    # Solve generalized eigenproblem (Sb v = lambda Sw v)
    Sw_inv = np.linalg.pinv(Sw)
    eigvals, eigvecs = np.linalg.eigh(Sw_inv @ Sb)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    ld_scores = X @ eigvecs
    ld_labels = [f"LD{i+1}" for i in range(ld_scores.shape[1])]
    ld_df = pd.DataFrame(ld_scores[:, :2], columns=ld_labels[:2])
    ld_df[label_col] = labels

    # Plot LD1 vs LD2 if available
    if ld_scores.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=ld_df, x="LD1", y="LD2", hue=label_col, s=50, ax=ax)
        ax.set_title("LDA: label separation (LD1 vs LD2)")
        fig.tight_layout()
        fig.savefig(outdir / "lda_scatter_ld1_ld2.png", dpi=300)
        plt.close(fig)

    # One-dimensional density along LD1
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.kdeplot(data=ld_df, x="LD1", hue=label_col, fill=True, common_norm=False)
    ax.set_title("LDA: distribution along LD1")
    fig.tight_layout()
    fig.savefig(outdir / "lda_ld1_density.png", dpi=300)
    plt.close(fig)

    # Loadings
    loadings = pd.DataFrame(eigvecs[:, : len(ld_labels)], index=feature_list, columns=ld_labels)
    loadings.to_csv(outdir / "lda_loadings.csv")

    # Simple nearest-centroid accuracy in LD space
    centroids = {lab: ld_scores[labels == lab, :2].mean(axis=0) for lab in groups}
    preds = []
    for row in ld_scores[:, :2]:
        dists = {lab: np.linalg.norm(row - c) for lab, c in centroids.items()}
        preds.append(min(dists, key=dists.get))
    acc = np.mean(np.array(preds) == labels)
    pd.DataFrame({"nearest_centroid_accuracy": [acc]}).to_csv(
        outdir / "lda_nearest_centroid_accuracy.csv", index=False
    )

    return eigvecs, loadings
