"""Shared plotting utilities for consistent, publication-ready figures.

This module centralizes figure styling, sizing heuristics, and tick-label
formatting to ensure all actin-analysis plots look coherent and avoid crowded
axes. The helpers are intentionally lightweight and can be used by any plotting
module in this package without introducing heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def apply_scientific_theme() -> None:
    """Apply a consistent, publication-friendly theme and typography."""

    sns.set_theme(style="white", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": False,
        }
    )


def estimate_figsize(
    n_categories: int,
    max_label_len: int,
    base_size: tuple[float, float] = (6.0, 3.8),
    min_size: tuple[float, float] = (4.5, 3.2),
    per_category: float = 0.5,
    per_char: float = 0.06,
) -> tuple[float, float]:
    """Heuristically estimate figure size based on category count and label length."""

    width = max(min_size[0], base_size[0] + n_categories * per_category)
    width = max(width, base_size[0] + max_label_len * per_char)
    height = max(min_size[1], base_size[1] + max_label_len * 0.03)
    return (width, height)


def format_category_axis(ax, labels: Iterable[str]) -> None:
    """Format categorical tick labels with rotation/size tweaks to prevent overlaps."""

    labels = [str(label) for label in labels]
    max_len = max((len(label) for label in labels), default=0)
    rotation = 0
    ha = "center"
    font_size = 9

    if max_len >= 18:
        rotation = 45
        ha = "right"
        font_size = 8
    elif max_len >= 12:
        rotation = 30
        ha = "right"
        font_size = 9

    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.tick_params(axis="x", labelsize=font_size)


def format_heatmap_axes(ax, labels_x: Iterable[str], labels_y: Iterable[str]) -> None:
    """Apply consistent formatting to heatmap axes with dense labels."""

    labels_x = [str(label) for label in labels_x]
    labels_y = [str(label) for label in labels_y]
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_xticklabels(labels_x, rotation=45, ha="right")
    ax.set_yticklabels(labels_y, rotation=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


def save_figure(fig, path: Path) -> None:
    """Save figures with tight layout and close them to free resources."""

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_legend(ax, labels: Iterable[str] | None = None) -> None:
    """Place legends sensibly based on the number/length of labels."""

    handles, legend_labels = ax.get_legend_handles_labels()
    if labels is not None:
        legend_labels = [str(label) for label in labels]
    if not handles or not legend_labels:
        return

    max_len = max((len(label) for label in legend_labels), default=0)
    if len(legend_labels) > 6 or max_len > 12:
        ax.legend(
            handles,
            legend_labels,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0,
            frameon=True,
        )
    else:
        ax.legend(handles, legend_labels, frameon=True)
