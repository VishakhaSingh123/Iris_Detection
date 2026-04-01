"""
data_loader.py
--------------
Loads and preprocesses the Iris dataset.
- Loads from sklearn
- Splits into train/test (stratified)
- Scales features (needed for KNN)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_data(test_size=0.2, random_state=42):
    """
    Load Iris dataset and return train/test splits.
    Scaling is applied for KNN (distance-based).
    Decision Tree gets the unscaled version (tree splits don't need scaling).

    Returns:
        dict with keys: X_train, X_test, y_train, y_test,
                        X_train_scaled, X_test_scaled,
                        feature_names, class_names, df
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # Train/test split (stratified keeps class proportions in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Standard scaling: mean=0, std=1 — important for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit only on train!
    X_test_scaled  = scaler.transform(X_test)        # apply same scale to test

    # Also build a DataFrame for EDA convenience
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [class_names[i] for i in y]

    return {
        "X_train":        X_train,
        "X_test":         X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "y_train":        y_train,
        "y_test":         y_test,
        "feature_names":  list(feature_names),
        "class_names":    list(class_names),
        "df":             df,
    }


if __name__ == "__main__":
    data = load_data()
    print("Dataset loaded successfully.")
    print(f"  Training samples : {len(data['X_train'])}")
    print(f"  Test samples     : {len(data['X_test'])}")
    print(f"  Features         : {data['feature_names']}")
    print(f"  Classes          : {data['class_names']}")
    print("\nClass distribution in training set:")
    for i, name in enumerate(data["class_names"]):
        count = (data["y_train"] == i).sum()
        print(f"  {name}: {count} samples")
