"""Utilities for multidimensional actin analysis.

This package groups the reusable pieces that power ``actin_multidim_analysis``:
I/O helpers, descriptive statistics, plotting, PCA, and a suite of advanced
pipelines (MANOVA/LDA, pseudotime, clustering, index networks, mixed-effects
models, ML+SHAP). Each module writes outputs to caller-provided subdirectories
so figures and tables stay neatly organised on disk.
"""

from .io import ensure_outdir, load_per_cell_table
from .summary import summary_tables
from .stats import per_index_stat_tests
from .plots import plot_univariate, plot_correlation, profile_per_label
from .pca import run_pca
from .advanced_common import prepare_output_tree, standardize_features
from .manova_lda import run_manova, run_lda
from .pseudotime_trajectory import run_pseudotime
from .unsupervised_clustering import run_clustering
from .index_networks import run_index_network
from .mixed_effects import run_mixed_effects
from .ml_shap import run_ml_shap

__all__ = [
    "ensure_outdir",
    "load_per_cell_table",
    "summary_tables",
    "per_index_stat_tests",
    "plot_univariate",
    "plot_correlation",
    "profile_per_label",
    "run_pca",
    "prepare_output_tree",
    "standardize_features",
    "run_manova",
    "run_lda",
    "run_pseudotime",
    "run_clustering",
    "run_index_network",
    "run_mixed_effects",
    "run_ml_shap",
]
