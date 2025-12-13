"""Plotting utilities for actin index exploration.

This module focuses on quick univariate and pairwise visualisations: histograms,
violin plots, correlation heatmaps, and radar profiles. Callers are encouraged
to pass subdirectories (e.g. ``analysis_output/univariate``) so different plot
families remain neatly separated on disk.
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("ggplot")


def plot_univariate(
    df, label_col: str, feature_cols: Iterable[str], outdir: Path
) -> None:
    """Plot histograms and violin plots for each feature."""

    for feat in feature_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df[feat].dropna(), kde=True, ax=ax)
        ax.set_title(f"{feat} – overall distribution")
        ax.set_xlabel(feat)
        fig.tight_layout()
        fig.savefig(outdir / f"{feat}_hist.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.violinplot(data=df, x=label_col, y=feat, inner="box", cut=0)
        ax.set_title(f"{feat} by {label_col}")
        ax.set_xlabel(label_col)
        ax.set_ylabel(feat)
        fig.tight_layout()
        fig.savefig(outdir / f"{feat}_by_label_violin.png", dpi=300)
        plt.close(fig)


def plot_correlation(df, feature_cols: Iterable[str], outdir: Path):
    """Compute correlation matrix and save heatmap + CSV."""

    corr = df[feature_cols].corr(method="pearson")
    fig, ax = plt.subplots(
        figsize=(max(6, 0.5 * len(feature_cols)), max(5, 0.4 * len(feature_cols)))
    )
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation between indices (Pearson)")
    fig.tight_layout()
    fig.savefig(outdir / "correlation_heatmap.png", dpi=300)
    plt.close(fig)
    corr.to_csv(outdir / "correlation_matrix.csv")
    return corr


def profile_per_label(df, label_col: str, feature_cols: Iterable[str], outdir: Path):
    """Create radar-style profile of mean index values per label."""
    # Convert to list once to keep ordering stable
    feature_list = list(feature_cols)
    if len(feature_list) > 15:
        return
    mean_table = df.groupby(label_col)[feature_list].mean()
    labels_order = mean_table.index.tolist()

    import math

    N = len(feature_list)
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for lab in labels_order:
        values = mean_table.loc[lab].values
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, label=str(lab))
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_list, fontsize=8)
    ax.set_title("Mean index profile per label (radar plot)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "per_label_radar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
