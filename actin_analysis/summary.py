"""Descriptive summaries for per-cell actin index tables.

This module produces quick-look CSVs for global and per-label distributions so
downstream notebooks and scripts can reuse consistent aggregates. The summaries
are designed for reporting pipelines and should be written into a dedicated
subfolder (e.g. ``analysis_output/summary``) created by the caller.
"""

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def summary_tables(
    df: pd.DataFrame, label_col: str, feature_cols: Iterable[str], outdir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Save global and per-label summary statistics (mean, std, quartiles)."""

    global_summary = pd.DataFrame(df[feature_cols].describe().T)
    global_summary.to_csv(outdir / "summary_global.csv")

    per_label_summary = (
        df.groupby(label_col)[feature_cols]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )
    per_label_summary.to_csv(outdir / "summary_per_label.csv")

    return global_summary, per_label_summary
