"""End-to-end multidimensional analysis for per-cell actin index tables.

This command-line entry point orchestrates the full analytical workflow used in
actin-index studies. It takes a ``per_cell_template.csv``-style table, validates
the schema, and then writes a structured results tree with publication-quality
figures and statistical summaries. Outputs are grouped into logical subfolders
so basic summaries, multivariate models, and advanced analyses stay tidy:

* ``summary/`` – global and per-label descriptive tables.
* ``per_index_tests/`` – ANOVA and Kruskal–Wallis results per index.
* ``univariate/`` – histograms and violin plots with density overlays.
* ``correlation/`` – correlation heatmap and matrix.
* ``pca/`` – PCA scatter, explained variance, and loadings.
* ``profiles/`` – per-label radar plots (when feature count is manageable).
* ``advanced/`` – subfolders for MANOVA/LDA, pseudotime, clustering, index
  networks, mixed-effects models, and ML+SHAP interpretations.

All directories are created automatically; each plotting/statistics module just
needs the folder path to write its own figures and tables. This script is the
recommended entry point for reproducible reporting workflows.
"""

import argparse
from pathlib import Path

from actin_analysis import (
    ensure_outdir,
    load_per_cell_table,
    per_index_stat_tests,
    plot_correlation,
    plot_univariate,
    profile_per_label,
    run_pca,
    summary_tables,
    prepare_output_tree,
    run_manova,
    run_lda,
    run_pseudotime,
    run_clustering,
    run_index_network,
    run_mixed_effects,
    run_ml_shap,
)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive per-label analysis of actin indices."
    )
    parser.add_argument("csv", type=str, help="Input CSV (same format as per_cell_template.csv)")
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of the label column (default: 'label')",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Optional JSON schema defining feature metadata and exclusions",
    )
    parser.add_argument(
        "--include-non-normalized",
        action="store_true",
        help="Include non-normalized features listed in the schema",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="analysis_output",
        help="Output directory for tables and figures",
    )
    parser.add_argument(
        "--random-effect",
        type=str,
        default=None,
        help="Optional column name to use as random effect for mixed models",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    schema_path = Path(args.schema) if args.schema else None
    df, id_col, label_col, feature_cols = load_per_cell_table(
        csv_path,
        label_col=args.label_col,
        schema_path=schema_path,
        include_non_normalized=args.include_non_normalized,
    )
    print(f"Loaded {len(df)} rows.")
    print(f"Label column: {label_col}")
    print(f"ID column: {id_col}")
    print(f"Index columns ({len(feature_cols)}): {feature_cols}")

    # 1. Summary statistics
    summary_dir = prepare_output_tree(outdir, "summary")
    summary_tables(df, label_col, feature_cols, summary_dir)
    print("Saved global and per-label summary tables.")

    # 2. Per-index statistical tests
    tests_dir = prepare_output_tree(outdir, "per_index_tests")
    tests = per_index_stat_tests(df, label_col, feature_cols, tests_dir)
    if tests is not None:
        print("Saved per-index ANOVA/Kruskal–Wallis tests.")

    # 3. Univariate plots
    univariate_dir = prepare_output_tree(outdir, "univariate")
    plot_univariate(df, label_col, feature_cols, univariate_dir)
    print("Saved univariate histograms and violin plots.")

    # 4. Correlation analysis
    correlation_dir = prepare_output_tree(outdir, "correlation")
    plot_correlation(df, feature_cols, correlation_dir)
    print("Saved correlation heatmap and matrix.")

    # 5. PCA / multivariate structure
    pca_dir = prepare_output_tree(outdir, "pca")
    run_pca(df, label_col, feature_cols, pca_dir)
    print("Saved PCA scatter, explained variance, and loadings.")

    # 6. Per-label radar profile (if not too many indices)
    profile_dir = prepare_output_tree(outdir, "profiles")
    profile_per_label(df, label_col, feature_cols, profile_dir)
    print("Saved per-label radar profile (if number of indices is reasonable).")

    # 7. Advanced analyses in structured subfolders
    advanced_root = prepare_output_tree(outdir, "advanced")

    manova_dir = prepare_output_tree(advanced_root, "manova_lda")
    run_manova(df, label_col, feature_cols, manova_dir)
    run_lda(df, label_col, feature_cols, manova_dir)
    print("Saved MANOVA table and LDA projections.")

    pseudotime_dir = prepare_output_tree(advanced_root, "pseudotime")
    run_pseudotime(df, label_col, feature_cols, pseudotime_dir)
    print("Saved UMAP embeddings and pseudotime scores.")

    clustering_dir = prepare_output_tree(advanced_root, "clustering")
    run_clustering(df, label_col, feature_cols, clustering_dir)
    print("Saved UMAP embeddings and HDBSCAN clustering outputs.")

    network_dir = prepare_output_tree(advanced_root, "index_networks")
    run_index_network(df, feature_cols, network_dir)
    print("Saved partial-correlation network heatmap and table.")

    mixed_dir = prepare_output_tree(advanced_root, "mixed_effects")
    run_mixed_effects(
        df, label_col, feature_cols, mixed_dir, random_effect=args.random_effect
    )
    print("Saved mixed-effects model summaries per index.")

    ml_dir = prepare_output_tree(advanced_root, "ml_shap")
    run_ml_shap(df, label_col, feature_cols, ml_dir)
    print("Saved XGBoost performance metrics and SHAP explanations.")

    print(f"All output written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
