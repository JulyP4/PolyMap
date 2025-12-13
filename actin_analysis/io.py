from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_outdir(outdir: Path) -> None:
    """Create the output directory if it does not exist."""

    outdir.mkdir(parents=True, exist_ok=True)


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
