"""Network-style analysis of relationships between actin indices.

Uses plain NumPy to compute inverse covariance (precision) and derive partial
correlations. This provides a first-pass view of conditional dependencies
between indices without relying on external solvers.
"""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")


def partial_corr_network(
    df: pd.DataFrame, feature_cols: Iterable[str], outdir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate precision and partial correlations via covariance inversion."""

    feature_list = list(feature_cols)
    X = df[feature_list].values.astype(float)
    cov = np.cov(X, rowvar=False)
    precision = np.linalg.pinv(cov)

    denom = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial_corr = -precision / denom
    np.fill_diagonal(partial_corr, 1.0)

    precision_df = pd.DataFrame(precision, index=feature_list, columns=feature_list)
    partial_corr_df = pd.DataFrame(partial_corr, index=feature_list, columns=feature_list)
    precision_df.to_csv(outdir / "precision_matrix.csv")
    partial_corr_df.to_csv(outdir / "partial_correlation_matrix.csv")

    fig, ax = plt.subplots(
        figsize=(max(6, 0.5 * len(feature_list)), max(5, 0.4 * len(feature_list)))
    )
    sns.heatmap(partial_corr_df, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Partial correlation (precision-based)")
    fig.tight_layout()
    fig.savefig(outdir / "partial_corr_heatmap.png", dpi=300)
    plt.close(fig)

    cond_number = np.linalg.cond(cov)
    pd.DataFrame({"cov_condition_number": [cond_number]}).to_csv(
        outdir / "covariance_condition_number.csv", index=False
    )

    return precision_df, partial_corr_df
