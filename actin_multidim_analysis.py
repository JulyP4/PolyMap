import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.style.use("ggplot")


def load_per_cell_table(csv_path: Path, label_col: str = "label") -> pd.DataFrame:
    """
    Load the per-cell table.
    Lines starting with '#' are treated as comments (like the template header).

    Parameters
    ----------
    csv_path : Path
        CSV file with columns: id, label, index_1, index_2, ...
    label_col : str
        Name of the label column.

    Returns
    -------
    df : DataFrame
    """
    df = pd.read_csv(csv_path, comment="#")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Columns: {df.columns.tolist()}")

    # Drop completely empty columns if any
    df = df.dropna(axis=1, how="all")

    # Infer id column (optional)
    id_col = None
    for c in df.columns:
        if c.lower() in {"id", "cell_id", "roi_id"}:
            id_col = c
            break

    # Feature columns = all numeric columns except id
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if id_col is not None and id_col in feature_cols:
        feature_cols.remove(id_col)

    if not feature_cols:
        raise ValueError("No numeric index columns found.")

    return df, id_col, label_col, feature_cols


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def summary_tables(df: pd.DataFrame, label_col: str, feature_cols, outdir: Path):
    """
    Global + per-label summary statistics (mean, std, quartiles) for each index.
    Saved as CSV.
    """
    global_summary = df[feature_cols].describe().T
    global_summary.to_csv(outdir / "summary_global.csv")

    per_label_summary = (
        df.groupby(label_col)[feature_cols]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )
    per_label_summary.to_csv(outdir / "summary_per_label.csv")

    return global_summary, per_label_summary


def per_index_stat_tests(df: pd.DataFrame, label_col: str, feature_cols, outdir: Path):
    """
    For each index, test whether its distribution differs between labels.

    We run:
    - One-way ANOVA (parametric, assumes roughly normal within each label)
    - Kruskal–Wallis test (non-parametric, rank-based)

    Results are saved to a CSV.
    """
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


def plot_univariate(df: pd.DataFrame, label_col: str, feature_cols, outdir: Path):
    """
    For each index, make:
    - Histogram (all cells)
    - Box/violin by label
    """
    for feat in feature_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df[feat].dropna(), kde=True, ax=ax)
        ax.set_title(f"{feat} – overall distribution")
        ax.set_xlabel(feat)
        fig.tight_layout()
        fig.savefig(outdir / f"{feat}_hist.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.violinplot(data=df, x=label_col, y=feat, inner="box", cut=0)
        ax.set_title(f"{feat} by {label_col}")
        ax.set_xlabel(label_col)
        ax.set_ylabel(feat)
        fig.tight_layout()
        fig.savefig(outdir / f"{feat}_by_label_violin.png", dpi=300)
        plt.close(fig)


def plot_correlation(df: pd.DataFrame, feature_cols, outdir: Path):
    """
    Correlation matrix of indices.
    """
    corr = df[feature_cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(feature_cols)), max(5, 0.4 * len(feature_cols))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation between indices (Pearson)")
    fig.tight_layout()
    fig.savefig(outdir / "correlation_heatmap.png", dpi=300)
    plt.close(fig)
    corr.to_csv(outdir / "correlation_matrix.csv")
    return corr


def run_pca(df: pd.DataFrame, label_col: str, feature_cols, outdir: Path):
    """
    PCA on standardized indices.
    """
    X = df[feature_cols].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=min(5, len(feature_cols)))
    PCs = pca.fit_transform(Xz)

    pc_cols = [f"PC{i+1}" for i in range(PCs.shape[1])]
    pca_df = pd.DataFrame(PCs, columns=pc_cols)
    pca_df[label_col] = df[label_col].values

    # Scatter PC1 vs PC2
    if PCs.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=label_col,
            alpha=0.7,
            s=40,
            edgecolor="none",
        )
        ax.set_title("PCA of indices (PC1 vs PC2)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(outdir / "pca_pc1_pc2_scatter.png", dpi=300)
        plt.close(fig)

    # Explained variance ratio
    evr = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(range(1, len(evr) + 1), evr * 100)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA explained variance")
    fig.tight_layout()
    fig.savefig(outdir / "pca_explained_variance.png", dpi=300)
    plt.close(fig)

    # Loadings heatmap (indices vs PCs)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=pc_cols,
    )
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(pc_cols)), max(5, 0.4 * len(feature_cols))))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("PCA loadings (contribution of each index)")
    fig.tight_layout()
    fig.savefig(outdir / "pca_loadings_heatmap.png", dpi=300)
    plt.close(fig)

    loadings.to_csv(outdir / "pca_loadings.csv")
    pca_df.to_csv(outdir / "pca_scores.csv", index=False)

    return pca, pca_df, loadings


def profile_per_label(df: pd.DataFrame, label_col: str, feature_cols, outdir: Path):
    """
    Radar-style profile: mean index values per label.
    Skipped if too many indices (>15).
    """
    if len(feature_cols) > 15:
        return

    mean_table = df.groupby(label_col)[feature_cols].mean()
    labels_order = mean_table.index.tolist()

    import math

    N = len(feature_cols)
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for lab in labels_order:
        values = mean_table.loc[lab].values
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, label=str(lab))
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, fontsize=8)
    ax.set_title("Mean index profile per label (radar plot)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "per_label_radar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive per-label analysis of actin indices.")
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

    df, id_col, label_col, feature_cols = load_per_cell_table(csv_path, label_col=args.label_col)
    print(f"Loaded {len(df)} rows.")
    print(f"Label column: {label_col}")
    print(f"ID column: {id_col}")
    print(f"Index columns ({len(feature_cols)}): {feature_cols}")

    # 1. Summary statistics
    global_summary, per_label_summary = summary_tables(df, label_col, feature_cols, outdir)
    print("Saved global and per-label summary tables.")

    # 2. Per-index statistical tests
    tests = per_index_stat_tests(df, label_col, feature_cols, outdir)
    if tests is not None:
        print("Saved per-index ANOVA/Kruskal–Wallis tests.")

    # 3. Univariate plots
    plot_univariate(df, label_col, feature_cols, outdir)
    print("Saved univariate histograms and violin plots.")

    # 4. Correlation analysis
    corr = plot_correlation(df, feature_cols, outdir)
    print("Saved correlation heatmap and matrix.")

    # 5. PCA / multivariate structure
    pca, pca_df, loadings = run_pca(df, label_col, feature_cols, outdir)
    print("Saved PCA scatter, explained variance, and loadings.")

    # 6. Per-label radar profile (if not too many indices)
    profile_per_label(df, label_col, feature_cols, outdir)
    print("Saved per-label radar profile (if number of indices is reasonable).")

    print(f"All output written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
