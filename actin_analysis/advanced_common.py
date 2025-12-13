"""Shared helpers for advanced actin-index analyses.

This module centralises utilities so the advanced pipelines can stay focused
on statistical logic while reusing consistent I/O and output handling.

Provided helpers
----------------
load_dataset
    Wraps :func:`actin_analysis.io.load_per_cell_table` to read the standard
    per-cell CSV and return ``(df, id_col, label_col, feature_cols)``.
prepare_output_tree
    Ensures a dedicated subfolder under a base output directory for each
    analysis, keeping figures and tables tidy (e.g. ``analysis_output/manova_lda``).
standardize_features
    Z-score selected numeric columns and return both the scaled array and the
    fitted :class:`sklearn.preprocessing.StandardScaler`.

Notes
-----
* Any range computation uses :func:`numpy.ptp` explicitly to avoid the removed
  ndarray method in NumPy 2.0.
* The helpers remain dependency-light and reuse existing actin_analysis IO
  utilities for consistency across scripts.
"""

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from actin_analysis.io import ensure_outdir, load_per_cell_table


def load_dataset(csv_path: Path, label_col: str = "label"):
    """Load a per-cell table using the shared IO helper.

    Parameters
    ----------
    csv_path:
        Path to a CSV that follows ``per_cell_template.csv``.
    label_col:
        Column name containing categorical labels.
    """

    return load_per_cell_table(csv_path, label_col=label_col)


def prepare_output_tree(base_outdir: Path, analysis_name: str) -> Path:
    """Create a structured output directory for a specific analysis."""

    ensure_outdir(base_outdir)
    analysis_dir = base_outdir / analysis_name
    ensure_outdir(analysis_dir)
    return analysis_dir


def standardize_features(
    df: pd.DataFrame, feature_cols: Iterable[str]
) -> Tuple[np.ndarray, StandardScaler]:
    """Z-score the given numeric columns and return the scaled matrix."""

    feature_list = list(feature_cols)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_list].values)
    return scaled, scaler


__all__ = ["load_dataset", "prepare_output_tree", "standardize_features"]
