"""Publication-oriented plotting utilities for actin index exploration.

This module standardises univariate and pairwise visualisations used throughout
the actin-analysis workflow. The routines emphasize publication-ready output by
applying a shared scientific theme, automatically resizing plots to accommodate
long category labels, and formatting dense heatmaps for readability.

Callers are encouraged to pass subdirectories (e.g. ``analysis_output/univariate``)
so different plot families remain neatly separated on disk.
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from actin_analysis.plot_style import (
    apply_scientific_theme,
    estimate_figsize,
    format_category_axis,
    format_heatmap_axes,
    save_figure,
)

apply_scientific_theme()


def plot_univariate(
    df, label_col: str, feature_cols: Iterable[str], outdir: Path
) -> None:
    """Plot histograms and violin plots for each feature."""

    label_values = df[label_col].dropna().unique()
    label_order = [str(label) for label in label_values]
    label_count = len(label_order)
    max_label_len = max((len(label) for label in label_order), default=0)

    summary_rows = []
    for feat in feature_cols:
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        sns.histplot(
            df[feat].dropna(),
            kde=True,
            ax=ax,
            color="#4C72B0",
            stat="density",
            edgecolor="white",
        )
        ax.set_title(f"{feat} – overall distribution")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        save_figure(fig, outdir / f"{feat}_hist.png")

        fig, ax = plt.subplots(
            figsize=estimate_figsize(label_count, max_label_len, base_size=(6.0, 3.6))
        )
        sns.violinplot(
            data=df,
            x=label_col,
            y=feat,
            inner="quartile",
            cut=0,
            linewidth=1.0,
            scale="width",
            width=0.8,
            ax=ax,
            palette="Set2",
            order=label_order,
        )
        sns.stripplot(
            data=df,
            x=label_col,
            y=feat,
            color="black",
            size=2,
            alpha=0.35,
            jitter=0.2,
            ax=ax,
            order=label_order,
        )
        sns.pointplot(
            data=df,
            x=label_col,
            y=feat,
            order=label_order,
            ci=95,
            join=False,
            color="black",
            markers="D",
            errwidth=1.2,
            capsize=0.2,
            ax=ax,
        )
        ax.set_title(f"{feat} by {label_col}")
        ax.set_xlabel(label_col)
        ax.set_ylabel(feat)
        format_category_axis(ax, label_order)
        save_figure(fig, outdir / f"{feat}_by_label_violin.png")

        for label in label_order:
            values = df.loc[df[label_col].astype(str) == label, feat].dropna()
            if values.empty:
                continue
            mean = values.mean()
            std = values.std()
            median = values.median()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            cv = std / mean if mean else np.nan
            skewness = stats.skew(values, nan_policy="omit")
            summary_rows.append(
                {
                    "label": label,
                    "feature": feat,
                    "count": len(values),
                    "mean": mean,
                    "std": std,
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "cv": cv,
                    "skewness": skewness,
                }
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(outdir / "univariate_summary_stats.csv", index=False)


def plot_correlation(df, feature_cols: Iterable[str], outdir: Path):
    """Compute correlation matrix and save heatmap + CSV."""

    corr = df[feature_cols].corr(method="pearson")
    n_features = len(feature_cols)
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * n_features), max(6, 0.38 * n_features)))
    show_annotations = n_features <= 18
    sns.heatmap(
        corr,
        annot=show_annotations,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation between indices (Pearson)")
    format_heatmap_axes(ax, feature_cols, feature_cols)
    save_figure(fig, outdir / "correlation_heatmap.png")
    corr.to_csv(outdir / "correlation_matrix.csv")
    pvals = pd.DataFrame(
        np.ones((n_features, n_features)), index=feature_cols, columns=feature_cols
    )
    for i, feat_i in enumerate(feature_cols):
        for j, feat_j in enumerate(feature_cols):
            if j < i:
                continue
            values = df[[feat_i, feat_j]].dropna()
            if values.empty:
                continue
            try:
                _, p_value = stats.pearsonr(
                    values[feat_i].to_numpy().ravel(),
                    values[feat_j].to_numpy().ravel(),
                )
            except ValueError:
                p_value = np.nan
            p_value = np.squeeze(p_value)
            p_value = float(p_value) if np.size(p_value) == 1 else np.nan
            pvals.loc[feat_i, feat_j] = p_value
            pvals.loc[feat_j, feat_i] = p_value
    pvals.to_csv(outdir / "correlation_pvalues.csv")
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

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for lab in labels_order:
        values = mean_table.loc[lab].values
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, label=str(lab))
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_list, fontsize=8)
    ax.set_title("Mean index profile per label (radar plot)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.grid(False)
    save_figure(fig, outdir / "per_label_radar.png")
