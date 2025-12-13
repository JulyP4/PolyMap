"""Graph-based pseudotime trajectory inference for actin indices.

This module embeds cells in a low-dimensional space, infers a principal tree
using minimum spanning tree + shortest paths, and assigns a pseudotime score to
each cell. Outputs are saved into ``<outdir>/pseudotime``.

Outputs
-------
* ``pseudotime_scores.csv``: per-cell pseudotime and embedding coordinates.
* ``pseudotime_scatter.png``: UMAP scatter coloured by pseudotime.
* ``pseudotime_by_label.png``: UMAP scatter coloured by the provided label.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

from actin_analysis.advanced_common import standardize_features

plt.style.use("ggplot")


def _compute_graph(embeddings: np.ndarray, n_neighbors: int = 10) -> nx.Graph:
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    G = nx.Graph()
    for i in range(len(embeddings)):
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):
            G.add_edge(i, j_idx, weight=float(dist))
    return G


def _pseudotime_from_graph(G: nx.Graph, start: int = 0) -> np.ndarray:
    lengths = nx.shortest_path_length(G, source=start, weight="weight")
    max_len = max(lengths.values()) if len(lengths) else 1.0
    pt = np.array([lengths.get(i, 0.0) / max_len for i in range(len(G))])
    return pt


def run_pseudotime(
    df: pd.DataFrame,
    label_col: str,
    feature_cols,
    outdir: Path,
    n_neighbors: int = 10,
    n_components: int = 2,
):
    """Compute UMAP embeddings and derive a simple graph-based pseudotime."""

    try:
        from umap import UMAP
    except ImportError:
        missing = "umap-learn (UMAP)"
        note = outdir / "_pseudotime_skipped_missing_dependency.txt"
        note.write_text(
            "Pseudotime analysis skipped because the optional dependency"
            f" {missing} is not installed. Install it via 'pip install umap-learn'"
            " to enable this module.\n"
        )
        return None

    Xz, _ = standardize_features(df, feature_cols)

    reducer = UMAP(n_components=n_components, random_state=42)
    emb = reducer.fit_transform(Xz)

    G = _compute_graph(emb, n_neighbors=n_neighbors)
    mst = nx.minimum_spanning_tree(G)

    # pick root as node with minimal average distance
    centrality = nx.closeness_centrality(mst, distance="weight")
    start = int(max(centrality, key=centrality.get))
    pseudotime = _pseudotime_from_graph(mst, start=start)

    out = pd.DataFrame(emb, columns=[f"UMAP{i+1}" for i in range(emb.shape[1])])
    out["pseudotime"] = pseudotime
    out[label_col] = df[label_col].values
    out.to_csv(outdir / "pseudotime_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(out["UMAP1"], out["UMAP2"], c=pseudotime, cmap="viridis", s=30)
    ax.set_title("UMAP embedding coloured by pseudotime")
    fig.colorbar(sc, ax=ax, label="pseudotime")
    fig.tight_layout()
    fig.savefig(outdir / "pseudotime_scatter.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=out, x="UMAP1", y="UMAP2", hue=label_col, s=35, ax=ax)
    ax.set_title("UMAP embedding coloured by label")
    fig.tight_layout()
    fig.savefig(outdir / "pseudotime_by_label.png", dpi=300)
    plt.close(fig)


__all__ = ["run_pseudotime"]
