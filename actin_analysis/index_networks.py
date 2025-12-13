"""Partial-correlation index networks for actin measurements.

This module builds a Gaussian graphical model (Graphical Lasso) from the index
matrix to highlight conditionally dependent relationships among indices. Outputs
live under ``<outdir>/index_networks``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.covariance import GraphicalLassoCV

plt.style.use("ggplot")


def run_index_network(df: pd.DataFrame, feature_cols, outdir: Path):
    """Fit Graphical Lasso, save partial correlations, and plot heatmap."""

    X = df[feature_cols].values
    model = GraphicalLassoCV()
    model.fit(X)

    prec = pd.DataFrame(model.precision_, index=feature_cols, columns=feature_cols)
    partial = -prec / np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
    np.fill_diagonal(partial.values, 1.0)
    partial_df = pd.DataFrame(partial, index=feature_cols, columns=feature_cols)
    partial_df.to_csv(outdir / "partial_correlations.csv")

    fig, ax = plt.subplots(
        figsize=(max(6, 0.5 * len(feature_cols)), max(5, 0.4 * len(feature_cols)))
    )
    sns.heatmap(partial_df, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Partial correlation network (Graphical Lasso)")
    fig.tight_layout()
    fig.savefig(outdir / "partial_corr_heatmap.png", dpi=300)
    plt.close(fig)


__all__ = ["run_index_network"]
