"""Supervised label prediction using a simple nearest-centroid classifier.

Without external ML libraries, we implement a lightweight classifier: compute
the mean index vector for each label and classify samples by nearest centroid.
Cross-validated accuracy and confusion matrix are saved alongside feature
contributions (variance explained per feature).
"""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    return (X - means) / stds, means, stds


def _confusion_matrix(true: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def _kfold_indices(n_samples: int, n_splits: int):
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        yield train_idx, test_idx
        current = stop


def train_label_predictor(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Iterable[str],
    outdir: Path,
    n_splits: int = 5,
) -> Tuple[pd.Series, np.ndarray]:
    """Nearest-centroid classifier with simple cross-validation."""

    feature_list = list(feature_cols)
    X = df[feature_list].values.astype(float)
    y = df[label_col].values
    labels_unique = np.unique(y)

    Xz, _, _ = _standardize(X)

    preds = np.empty_like(y, dtype=object)
    for train_idx, test_idx in _kfold_indices(len(y), min(n_splits, len(y))):
        train_X, train_y = Xz[train_idx], y[train_idx]
        centroids = {lab: train_X[train_y == lab].mean(axis=0) for lab in labels_unique}
        for idx in test_idx:
            dists = {lab: np.linalg.norm(Xz[idx] - c) for lab, c in centroids.items()}
            preds[idx] = min(dists, key=dists.get)

    accuracy = np.mean(preds == y)
    pd.DataFrame({"cv_accuracy": [accuracy]}).to_csv(outdir / "nc_cv_accuracy.csv", index=False)

    cm = _confusion_matrix(y, preds, labels_unique)
    cm_df = pd.DataFrame(cm, index=labels_unique, columns=labels_unique)
    cm_df.to_csv(outdir / "nc_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Nearest-centroid confusion matrix")
    fig.tight_layout()
    fig.savefig(outdir / "nc_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Feature variance contribution (normalized variance per feature)
    variances = X.var(axis=0)
    contributions = variances / variances.sum()
    contrib_series = pd.Series(contributions, index=feature_list, name="variance_fraction")
    contrib_series.sort_values(ascending=False).to_csv(outdir / "feature_variance_fraction.csv")

    return contrib_series, cm
