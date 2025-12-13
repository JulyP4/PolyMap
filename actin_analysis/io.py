"""I/O helpers for the actin analysis pipeline.

This module focuses on loading the per-cell table and preparing a consistent
output directory tree. The goal is to keep the main analysis script light while
ensuring every analysis component writes its results into a predictable
location (e.g., ``outdir/summary``, ``outdir/pca``), so the outputs remain
organized even as the pipeline grows more complex.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_outdir(outdir: Path) -> None:
    """Create the output directory if it does not exist."""

    outdir.mkdir(parents=True, exist_ok=True)


def ensure_subdir(outdir: Path, name: str) -> Path:
    """Create and return a named subdirectory within ``outdir``."""

    subdir = outdir / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def build_output_tree(outdir: Path) -> Dict[str, Path]:
    """Create a standard set of subfolders for different analysis modules.

    The returned mapping can be passed directly to downstream functions, keeping
    plots and tables grouped by theme.
    """

    ensure_outdir(outdir)
    subfolders = {
        "summary": ensure_subdir(outdir, "summary"),
        "univariate": ensure_subdir(outdir, "univariate"),
        "correlation": ensure_subdir(outdir, "correlation"),
        "pca": ensure_subdir(outdir, "pca"),
        "manova": ensure_subdir(outdir, "manova"),
        "lda": ensure_subdir(outdir, "lda"),
        "clustering": ensure_subdir(outdir, "clustering"),
        "pseudotime": ensure_subdir(outdir, "pseudotime"),
        "network": ensure_subdir(outdir, "network"),
        "model": ensure_subdir(outdir, "model"),
    }
    return subfolders


def load_per_cell_table(
    csv_path: Path, label_col: str = "label"
) -> Tuple[pd.DataFrame, Optional[str], str, List[str]]:
    """
    Load the per-cell table while preserving key metadata.

    Lines starting with ``#`` are treated as comments (like the template header).

    Parameters
    ----------
    csv_path:
        CSV file with columns: id, label, index_1, index_2, ...
    label_col:
        Name of the label column.

    Returns
    -------
    df, id_col, label_col, feature_cols
        The dataframe, the inferred id column name (if any), the label column
        name, and a list of numeric feature columns.
    """

    df = pd.read_csv(csv_path, comment="#")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Columns: {df.columns.tolist()}")

    # Drop completely empty columns if any
    df = df.dropna(axis=1, how="all")

    # Infer id column (optional)
    id_col = None
    for col in df.columns:
        if col.lower() in {"id", "cell_id", "roi_id"}:
            id_col = col
            break

    # Feature columns = all numeric columns except id
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if id_col is not None and id_col in feature_cols:
        feature_cols.remove(id_col)

    if not feature_cols:
        raise ValueError("No numeric index columns found.")

    return df, id_col, label_col, feature_cols
