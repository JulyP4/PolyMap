"""Tree-based classification/regression with SHAP explanations.

This module trains a gradient boosting model to predict labels from actin
indices, reports performance, and computes SHAP values for interpretability.
Results are saved under ``<outdir>/ml_shap``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

plt.style.use("ggplot")


def run_ml_shap(
    df: pd.DataFrame,
    label_col: str,
    feature_cols,
    outdir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train XGBoost classifier and compute SHAP summaries."""

    try:
        import shap
        from xgboost import XGBClassifier
    except ImportError as exc:
        note = outdir / "_ml_shap_skipped_missing_dependency.txt"
        note.write_text(
            "ML+SHAP analysis skipped because optional dependencies are missing.\n"
            "Required: 'xgboost' and 'shap'.\n"
            f"Encountered: {exc}\n"
        )
        return None

    X = df[feature_cols].values
    y = df[label_col].values

    n_samples = len(df)
    n_classes = len(np.unique(y))
    min_test_fraction = n_classes / max(n_samples, 1)
    adjusted_test_size = max(test_size, min_test_fraction)

    if adjusted_test_size >= 1:
        note = outdir / "_ml_shap_skipped_split_too_small.txt"
        note.write_text(
            "ML+SHAP skipped because dataset is too small for a hold-out split "
            f"with {n_classes} classes and {n_samples} samples.\n"
        )
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=adjusted_test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        # Fall back to non-stratified split if class counts are too small.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=adjusted_test_size, random_state=random_state, stratify=None
        )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    with open(outdir / "ml_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.3f}\n\n")
        f.write(report)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary.png", dpi=300)
    plt.close()

    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(outdir / "shap_importance_bar.png", dpi=300)
    plt.close()

    importance = pd.Series(clf.feature_importances_, index=feature_cols)
    importance.sort_values(ascending=False).to_csv(outdir / "xgb_feature_importance.csv")


__all__ = ["run_ml_shap"]
