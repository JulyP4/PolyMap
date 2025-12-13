from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


def per_index_stat_tests(
    df: pd.DataFrame, label_col: str, feature_cols: Iterable[str], outdir: Path
) -> Optional[pd.DataFrame]:
    """Run ANOVA and Kruskal–Wallis tests per index and save results."""

    labels = df[label_col].unique()
    labels.sort()
    rows = []
    if len(labels) < 2:
        return None

    for feat in feature_cols:
        groups = [df.loc[df[label_col] == lab, feat].dropna().values for lab in labels]
        # Skip if any group has <2 points
        if any(len(g) < 2 for g in groups):
            rows.append(
                {
                    "index": feat,
                    "anova_F": np.nan,
                    "anova_p": np.nan,
                    "kruskal_H": np.nan,
                    "kruskal_p": np.nan,
                }
            )
            continue

        F, p_anova = stats.f_oneway(*groups)
        H, p_kruskal = stats.kruskal(*groups)

        rows.append(
            {
                "index": feat,
                "anova_F": F,
                "anova_p": p_anova,
                "kruskal_H": H,
                "kruskal_p": p_kruskal,
            }
        )

    res = pd.DataFrame(rows).set_index("index")
    res.to_csv(outdir / "per_index_tests.csv")
    return res
