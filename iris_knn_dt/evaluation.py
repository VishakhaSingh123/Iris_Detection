"""
evaluation.py
-------------
Evaluates KNN and Decision Tree models.
Outputs:
  - Accuracy & F1 score
  - Confusion matrix (plotted)
  - Cross-validation scores
  - Feature importance (Decision Tree only)
  - Side-by-side comparison plot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# ── Color palette ─────────────────────────────────────────────────────────────
KNN_COLOR = "#378ADD"
DT_COLOR  = "#1D9E75"
PALETTE   = ["#378ADD", "#1D9E75", "#D85A30"]   # for 3 Iris classes


def evaluate_model(model, X_test, y_test, class_names, model_name):
    """
    Compute and print accuracy, F1, and full classification report.

    Returns dict with accuracy and f1.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'─'*45}")
    print(f"  {model_name}")
    print(f"{'─'*45}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Score  : {f1:.4f}  (weighted)")
    print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

    return {"model_name": model_name, "accuracy": acc, "f1": f1, "y_pred": y_pred}


def cross_validate_models(knn, dt, data, cv=5):
    """
    Run k-fold cross-validation on both models.
    KNN uses scaled data, DT uses raw data.

    Cross-validation = more reliable than a single train/test split.
    Trains on (cv-1) folds, tests on 1 fold, rotates through all folds.
    """
    knn_scores = cross_val_score(
        knn, data["X_train_scaled"], data["y_train"], cv=cv, scoring="accuracy"
    )
    dt_scores = cross_val_score(
        dt, data["X_train"], data["y_train"], cv=cv, scoring="accuracy"
    )

    print(f"\n{'─'*45}")
    print(f"  {cv}-Fold Cross-Validation")
    print(f"{'─'*45}")
    print(f"  KNN scores      : {np.round(knn_scores, 4)}")
    print(f"  KNN mean ± std  : {knn_scores.mean():.4f} ± {knn_scores.std():.4f}")
    print(f"\n  DT  scores      : {np.round(dt_scores, 4)}")
    print(f"  DT  mean ± std  : {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")

    return knn_scores, dt_scores


def plot_confusion_matrices(knn_results, dt_results, class_names):
    """
    Plot side-by-side confusion matrices for KNN and Decision Tree.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices — KNN vs Decision Tree", fontsize=14, fontweight="bold")

    for ax, results, color in zip(axes, [knn_results, dt_results], [KNN_COLOR, DT_COLOR]):
        cm = confusion_matrix(results["y_test"], results["y_pred"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(color, as_cmap=True),
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
        )
        ax.set_title(results["model_name"], fontsize=12, pad=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: confusion_matrices.png")


def plot_feature_importance(dt_model, feature_names):
    """
    Plot feature importances from the Decision Tree.

    Importance = total Gini impurity reduction attributed to each feature
    across all splits in the tree. Higher = more discriminative.
    """
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        sorted_names[::-1],
        importances[indices[::-1]],
        color=DT_COLOR,
        edgecolor="white",
        height=0.55,
    )

    for bar, val in zip(bars, importances[indices[::-1]]):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=10
        )

    ax.set_title("Feature Importance — Decision Tree (Gini)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance score")
    ax.set_xlim(0, max(importances) + 0.12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: feature_importance.png")


def plot_cv_comparison(knn_scores, dt_scores):
    """
    Box plot comparing cross-validation score distributions for both models.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    bp = ax.boxplot(
        [knn_scores, dt_scores],
        labels=["KNN", "Decision Tree"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=2),
    )
    bp["boxes"][0].set_facecolor(KNN_COLOR)
    bp["boxes"][1].set_facecolor(DT_COLOR)

    # Overlay individual fold scores
    for i, scores in enumerate([knn_scores, dt_scores], start=1):
        ax.scatter(
            [i] * len(scores), scores,
            color="white", s=40, zorder=3, edgecolors="gray", linewidths=0.5
        )

    ax.set_title("Cross-Validation Accuracy Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.8, 1.02)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("cv_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: cv_comparison.png")


def plot_accuracy_f1_comparison(knn_results, dt_results):
    """
    Bar chart comparing Accuracy and F1 score side by side.
    """
    metrics     = ["Accuracy", "F1 Score (weighted)"]
    knn_vals    = [knn_results["accuracy"], knn_results["f1"]]
    dt_vals     = [dt_results["accuracy"],  dt_results["f1"]]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width/2, knn_vals, width, label="KNN",           color=KNN_COLOR, edgecolor="white")
    b2 = ax.bar(x + width/2, dt_vals,  width, label="Decision Tree",  color=DT_COLOR,  edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_title("Model Comparison — Accuracy & F1 Score", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("accuracy_f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: accuracy_f1_comparison.png")
