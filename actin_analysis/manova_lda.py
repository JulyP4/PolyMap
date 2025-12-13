"""MANOVA + Linear Discriminant Analysis on actin index tables.

This module compares multivariate index profiles across labels using MANOVA
and visualises group separation via LDA. Results are written into a dedicated
subdirectory (``<outdir>/manova_lda``) created by the caller.

Outputs
-------
* ``manova_stats.csv``: Wilks' lambda, Pillai's trace, Hotelling–Lawley, Roy's
  greatest root with p-values.
* ``lda_scatter.png``: 2D scatter of the first two discriminant axes.
* ``lda_loadings.csv`` and ``lda_loadings_heatmap.png``: index contributions to
  LD1/LD2.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from actin_analysis.advanced_common import standardize_features

plt.style.use("ggplot")


def run_manova(df: pd.DataFrame, label_col: str, feature_cols, outdir: Path) -> pd.DataFrame:
    """Fit MANOVA and save the multivariate test table."""

    try:
        from statsmodels.multivariate.manova import MANOVA
    except ImportError:
        note = outdir / "_manova_skipped_missing_dependency.txt"
        note.write_text(
            "MANOVA skipped because 'statsmodels' is not installed.\n"
            "Install it via 'pip install statsmodels' to enable this module.\n"
        )
        return pd.DataFrame()

    formula_terms = " + ".join(feature_cols)
    formula = f"{formula_terms} ~ {label_col}"
    mv = MANOVA.from_formula(formula, data=df)
    mv_res = mv.mv_test()
    stats_df = mv_res.results[label_col]["stat"].copy()
    stats_df.to_csv(outdir / "manova_stats.csv")
    return stats_df


def run_lda(df: pd.DataFrame, label_col: str, feature_cols, outdir: Path):
    """Fit LDA, project data, and save scatter + loadings."""

    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except ImportError:
        note = outdir / "_lda_skipped_missing_dependency.txt"
        note.write_text(
            "LDA skipped because 'scikit-learn' is not installed.\n"
            "Install it via 'pip install scikit-learn' to enable this module.\n"
        )
        return None

    Xz, _ = standardize_features(df, feature_cols)
    y = df[label_col].values
    lda = LinearDiscriminantAnalysis(n_components=min(len(np.unique(y)) - 1, 2))
    proj = lda.fit_transform(Xz, y)

    proj_df = pd.DataFrame(proj, columns=["LD1", "LD2"][: proj.shape[1]])
    proj_df[label_col] = y

    if proj.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=proj_df,
            x="LD1",
            y="LD2",
            hue=label_col,
            palette="tab10",
            s=35,
            edgecolor="none",
            alpha=0.8,
            ax=ax,
        )
        ax.set_title("LDA: LD1 vs LD2")
        fig.tight_layout()
        fig.savefig(outdir / "lda_scatter.png", dpi=300)
        plt.close(fig)

    n_comp = proj.shape[1]
    # ``lda.scalings_`` always has ``n_classes - 1`` columns; slice to the
    # projected dimensionality (often 1 or 2) to avoid shape mismatch when the
    # number of classes exceeds the requested components.
    scalings = lda.scalings_[:, :n_comp] if lda.scalings_ is not None else None
    loadings = pd.DataFrame(
        scalings,
        index=feature_cols,
        columns=[f"LD{i+1}" for i in range(n_comp)],
    )
    loadings.to_csv(outdir / "lda_loadings.csv")

    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * loadings.shape[1]), max(5, 0.4 * loadings.shape[0]))
    )
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("LDA loadings")
    fig.tight_layout()
    fig.savefig(outdir / "lda_loadings_heatmap.png", dpi=300)
    plt.close(fig)


__all__ = ["run_manova", "run_lda"]
