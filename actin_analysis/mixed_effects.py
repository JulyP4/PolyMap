"""Linear mixed-effects models for actin indices.

This module fits per-index mixed models to partition variance between fixed
effects (e.g., zone/label) and random effects (e.g., plant/root IDs). Outputs
include per-index model summaries and information criteria tables saved under
``<outdir>/mixed_effects`` for downstream reporting.
"""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def run_mixed_effects(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Iterable[str],
    outdir: Path,
    random_effect: Optional[str] = None,
):
    """Fit a simple mixed model for each feature and save summary tables."""

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        note = outdir / "_mixed_effects_skipped_missing_dependency.txt"
        note.write_text(
            "Mixed-effects analysis skipped because 'statsmodels' is not installed.\n"
            "Install it via 'pip install statsmodels' to enable this module.\n"
        )
        return None

    rows = []
    for feat in feature_cols:
        formula = f"{feat} ~ C({label_col})"
        if random_effect and random_effect in df.columns:
            formula += f" + (1|{random_effect})"

        # statsmodels mixedlm expects a different syntax; use patsy-like tuples
        if random_effect and random_effect in df.columns:
            model = smf.mixedlm(f"{feat} ~ C({label_col})", df, groups=df[random_effect])
        else:
            model = smf.ols(f"{feat} ~ C({label_col})", data=df)
        fit = model.fit()

        rows.append({"index": feat, "AIC": fit.aic, "BIC": fit.bic})
        with open(outdir / f"mixed_effects_{feat}.txt", "w") as f:
            f.write(str(fit.summary()))

    summary = pd.DataFrame(rows).set_index("index")
    summary.to_csv(outdir / "mixed_effects_model_ic.csv")
    return summary


__all__ = ["run_mixed_effects"]
