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
        "--outdir",
        type=str,
        default="analysis_output",
        help="Output directory for tables and figures",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df, id_col, label_col, feature_cols = load_per_cell_table(
        csv_path, label_col=args.label_col
    )
    print(f"Loaded {len(df)} rows.")
    print(f"Label column: {label_col}")
    print(f"ID column: {id_col}")
    print(f"Index columns ({len(feature_cols)}): {feature_cols}")

    # 1. Summary statistics
    summary_tables(df, label_col, feature_cols, outdir)
    print("Saved global and per-label summary tables.")

    # 2. Per-index statistical tests
    tests = per_index_stat_tests(df, label_col, feature_cols, outdir)
    if tests is not None:
        print("Saved per-index ANOVA/Kruskal–Wallis tests.")

    # 3. Univariate plots
    plot_univariate(df, label_col, feature_cols, outdir)
    print("Saved univariate histograms and violin plots.")

    # 4. Correlation analysis
    plot_correlation(df, feature_cols, outdir)
    print("Saved correlation heatmap and matrix.")

    # 5. PCA / multivariate structure
    run_pca(df, label_col, feature_cols, outdir)
    print("Saved PCA scatter, explained variance, and loadings.")

    # 6. Per-label radar profile (if not too many indices)
    profile_per_label(df, label_col, feature_cols, outdir)
    print("Saved per-label radar profile (if number of indices is reasonable).")

    print(f"All output written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
