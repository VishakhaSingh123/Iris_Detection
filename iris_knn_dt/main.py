"""
main.py
-------
Entry point for the Iris KNN vs Decision Tree project.

Run with:
    python main.py

Pipeline:
    1. Load & preprocess data
    2. EDA (plots)
    3. Train KNN and Decision Tree (with hyperparameter tuning)
    4. Evaluate — Accuracy, F1, Confusion Matrix, Cross-Validation
    5. Feature Importance (Decision Tree)
    6. Final comparison plots
"""

import warnings
warnings.filterwarnings("ignore")

from data_loader import load_data
from eda         import run_eda
from models      import train_knn, train_decision_tree
from evaluation  import (
    evaluate_model,
    cross_validate_models,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_cv_comparison,
    plot_accuracy_f1_comparison,
)


def main():
    print("=" * 50)
    print("  Iris Dataset — KNN vs Decision Tree")
    print("=" * 50)

    # ── 1. Load data ──────────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    data = load_data(test_size=0.2, random_state=42)
    print(f"      Train: {len(data['X_train'])} samples | Test: {len(data['X_test'])} samples")

    # ── 2. EDA ────────────────────────────────────────
    print("\n[2/5] Running EDA...")
    run_eda(data)

    # ── 3. Train models ───────────────────────────────
    print("\n[3/5] Training models with GridSearchCV...")
    knn = train_knn(data["X_train_scaled"], data["y_train"])
    dt  = train_decision_tree(data["X_train"], data["y_train"])

    # ── 4. Evaluate on test set ───────────────────────
    print("\n[4/5] Evaluating on test set...")
    knn_results = evaluate_model(
        knn, data["X_test_scaled"], data["y_test"],
        data["class_names"], "KNN"
    )
    dt_results = evaluate_model(
        dt, data["X_test"], data["y_test"],
        data["class_names"], "Decision Tree"
    )

    # Attach y_test for confusion matrix plotting
    knn_results["y_test"] = data["y_test"]
    dt_results["y_test"]  = data["y_test"]

    # ── 5. Cross-validation ───────────────────────────
    print("\n[5/5] Cross-validation (5-fold)...")
    knn_cv_scores, dt_cv_scores = cross_validate_models(knn, dt, data, cv=5)

    # ── Plots ─────────────────────────────────────────
    print("\nGenerating plots...")
    plot_confusion_matrices(knn_results, dt_results, data["class_names"])
    plot_feature_importance(dt, data["feature_names"])
    plot_cv_comparison(knn_cv_scores, dt_cv_scores)
    plot_accuracy_f1_comparison(knn_results, dt_results)

    # ── Final summary ─────────────────────────────────
    print("\n" + "=" * 50)
    print("  Final Summary")
    print("=" * 50)
    winner = "KNN" if knn_results["accuracy"] >= dt_results["accuracy"] else "Decision Tree"
    print(f"  KNN           — Accuracy: {knn_results['accuracy']:.4f}  F1: {knn_results['f1']:.4f}")
    print(f"  Decision Tree — Accuracy: {dt_results['accuracy']:.4f}  F1: {dt_results['f1']:.4f}")
    print(f"\n  Best model on test set : {winner}")
    print("=" * 50)
    print("\nAll plots saved as .png in current directory.")


if __name__ == "__main__":
    main()
