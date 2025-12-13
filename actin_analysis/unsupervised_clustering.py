"""Unsupervised clustering of actin index profiles.

This module performs UMAP embedding followed by HDBSCAN (density-based)
clustering and reports how clusters map onto the provided label column.
Outputs are written to ``<outdir>/clustering``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

from actin_analysis.advanced_common import standardize_features

plt.style.use("ggplot")


def run_clustering(
    df: pd.DataFrame,
    label_col: str,
    feature_cols,
    outdir: Path,
    min_cluster_size: int = 15,
):
    """Embed with UMAP, cluster with HDBSCAN, and save plots/tables."""

    try:
        from umap import UMAP
        from hdbscan import HDBSCAN
    except ImportError as exc:
        note = outdir / "_clustering_skipped_missing_dependency.txt"
        note.write_text(
            "Clustering analysis skipped because optional dependencies are missing.\n"
            "Required: 'umap-learn' and 'hdbscan'.\n"
            f"Encountered: {exc}\n"
        )
        return None

    Xz, _ = standardize_features(df, feature_cols)
    reducer = UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(Xz)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = clusterer.fit_predict(emb)

    out = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"])
    out["cluster"] = clusters
    out[label_col] = df[label_col].values
    out.to_csv(outdir / "umap_clusters.csv", index=False)

    ari = adjusted_rand_score(df[label_col], clusters)
    with open(outdir / "clustering_metrics.txt", "w") as f:
        f.write(f"Adjusted Rand Index vs {label_col}: {ari:.3f}\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=out, x="UMAP1", y="UMAP2", hue="cluster", palette="tab20", s=30, ax=ax)
    ax.set_title("UMAP + HDBSCAN clusters")
    fig.tight_layout()
    fig.savefig(outdir / "clusters_scatter.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=out, x="UMAP1", y="UMAP2", hue=label_col, s=30, ax=ax)
    ax.set_title("UMAP coloured by label")
    fig.tight_layout()
    fig.savefig(outdir / "umap_by_label.png", dpi=300)
    plt.close(fig)


__all__ = ["run_clustering"]
