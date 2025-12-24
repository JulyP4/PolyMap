"""Input/output helpers for per-cell actin index tables.

This module centralises file handling so scripts can assume a consistent
interface for loading ``per_cell_template.csv``-style data and creating output
folders. Key behaviours:

* Lines beginning with ``#`` in the CSV are treated as comments (mirroring the
  provided template header) and automatically skipped.
* :func:`ensure_outdir` makes sure nested output directories exist before we
  write figures or tables.
* The loader returns the dataframe plus the inferred ID column, label column,
  and numeric feature columns, keeping the rest of the pipeline simple.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_schema(schema_path: Optional[Path]) -> Dict[str, object]:
    """Load an optional JSON schema that describes feature metadata."""

    if schema_path is None:
        return {}

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    data = pd.read_json(schema_path, typ="series")
    if not isinstance(data, pd.Series):
        raise ValueError("Schema JSON must decode to a JSON object.")
    return data.to_dict()


def apply_schema_renames(df: pd.DataFrame, schema: Dict[str, object]) -> pd.DataFrame:
    """Apply optional column renames from schema."""

    rename_map = schema.get("feature_rename", {})
    if not isinstance(rename_map, dict):
        raise ValueError("Schema 'feature_rename' must be a JSON object.")
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def ensure_outdir(outdir: Path) -> None:
    """Create the output directory if it does not exist."""

    outdir.mkdir(parents=True, exist_ok=True)


def load_per_cell_table(
    csv_path: Path,
    label_col: str = "label",
    schema_path: Optional[Path] = None,
    include_non_normalized: bool = False,
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
    schema_path:
        Optional JSON schema for renaming or excluding features.
    include_non_normalized:
        Whether to keep non-normalized features listed in the schema.

    Returns
    -------
    df, id_col, label_col, feature_cols
        The dataframe, the inferred id column name (if any), the label column
        name, and a list of numeric feature columns.
    """

    df = pd.read_csv(csv_path, comment="#")
    schema = load_schema(schema_path)
    df = apply_schema_renames(df, schema)

    label_col = schema.get("label_col", label_col)
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Columns: {df.columns.tolist()}")

    # Drop completely empty columns if any
    df = df.dropna(axis=1, how="all")

    # Infer id column (optional)
    id_col = schema.get("id_col")
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"Schema id_col '{id_col}' not found in columns.")
    if id_col is None:
        for col in df.columns:
            if col.lower() in {"id", "cell_id", "roi_id"}:
                id_col = col
                break

    # Feature columns = all numeric columns except id
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if id_col is not None and id_col in feature_cols:
        feature_cols.remove(id_col)
    if label_col in feature_cols:
        feature_cols.remove(label_col)

    exclude_features = schema.get("exclude_features", [])
    if not isinstance(exclude_features, list):
        raise ValueError("Schema 'exclude_features' must be a JSON array.")
    non_normalized = schema.get("non_normalized_features", [])
    if not isinstance(non_normalized, list):
        raise ValueError("Schema 'non_normalized_features' must be a JSON array.")

    exclude_set = set(exclude_features)
    if not include_non_normalized:
        exclude_set.update(non_normalized)
    if exclude_set:
        feature_cols = [col for col in feature_cols if col not in exclude_set]

    if not feature_cols:
        raise ValueError("No numeric index columns found.")

    return df, id_col, label_col, feature_cols
