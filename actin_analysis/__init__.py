"""Utilities for multidimensional actin analysis.

The package organizes I/O, statistical calculations, plotting, PCA helpers,
and advanced analyses (MANOVA, clustering, pseudotime, networks, ML) used by
``actin_multidim_analysis.py``.
"""

from .io import build_output_tree, ensure_outdir, ensure_subdir, load_per_cell_table
from .summary import summary_tables
from .stats import per_index_stat_tests
from .plots import plot_univariate, plot_correlation, profile_per_label
from .pca import run_pca
from .multivariate import run_manova, run_lda
from .clustering import cluster_states
from .pseudotime import compute_pseudotime
from .network import partial_corr_network
from .modeling import train_label_predictor

__all__ = [
    "build_output_tree",
    "ensure_outdir",
    "ensure_subdir",
    "load_per_cell_table",
    "summary_tables",
    "per_index_stat_tests",
    "plot_univariate",
    "plot_correlation",
    "profile_per_label",
    "run_pca",
    "run_manova",
    "run_lda",
    "cluster_states",
    "compute_pseudotime",
    "partial_corr_network",
    "train_label_predictor",
]
