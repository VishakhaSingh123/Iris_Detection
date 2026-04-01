"""
models.py
---------
Trains KNN and Decision Tree classifiers on the Iris dataset.
- Includes hyperparameter tuning via GridSearchCV
- Returns fitted model objects
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def train_knn(X_train_scaled, y_train, cv=5):
    """
    Train KNN with GridSearchCV to find the best K.

    KNN is a lazy learner — it stores the training data and
    at prediction time finds the K nearest points by Euclidean distance.

    Why scaled data?
      Features have different units (cm). Without scaling, features with
      larger numeric ranges dominate the distance calculation unfairly.

    Args:
        X_train_scaled : scaled training features
        y_train        : training labels
        cv             : number of cross-validation folds

    Returns:
        best fitted KNeighborsClassifier
    """
    param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    best_k = grid.best_params_["n_neighbors"]
    print(f"[KNN] Best K = {best_k}  (CV accuracy = {grid.best_score_:.4f})")

    return grid.best_estimator_


def train_decision_tree(X_train, y_train, cv=5):
    """
    Train Decision Tree with GridSearchCV to find the best max_depth.

    The tree splits features using Gini Impurity:
        Gini = 1 - Σ(pᵢ²)
    It picks the feature + threshold that reduces impurity the most.

    Why unscaled data?
      Trees split on thresholds, not distances. Scaling doesn't change
      which threshold is best, so it's unnecessary.

    Args:
        X_train  : raw (unscaled) training features
        y_train  : training labels
        cv       : number of cross-validation folds

    Returns:
        best fitted DecisionTreeClassifier
    """
    param_grid = {
        "max_depth":        [2, 3, 4, 5, None],
        "min_samples_leaf": [1, 2, 4],
        "criterion":        ["gini", "entropy"]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print(f"[Decision Tree] Best params = {best_params}")
    print(f"[Decision Tree] CV accuracy = {grid.best_score_:.4f}")

    return grid.best_estimator_


if __name__ == "__main__":
    from data_loader import load_data

    data = load_data()

    print("Training KNN...")
    knn = train_knn(data["X_train_scaled"], data["y_train"])

    print("\nTraining Decision Tree...")
    dt = train_decision_tree(data["X_train"], data["y_train"])
