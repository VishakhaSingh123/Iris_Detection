# 🌸 Iris Species Classification — KNN & Decision Tree

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>A comparative study of K-Nearest Neighbors and Decision Tree classifiers on the classic Fisher's Iris dataset — exploring how two fundamentally different algorithms approach the same classification problem.</b>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Algorithms](#-algorithms)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)
- [Usage](#-usage)
- [Key Insights](#-key-insights)
- [Contributing](#-contributing)

---

## 🔍 Overview

This project implements and compares two classic supervised learning algorithms — **K-Nearest Neighbors (KNN)** and **Decision Tree (DT)** — to classify iris flowers into three species:

| Species | Description |
|---|---|
| 🌺 *Iris Setosa* | Easily separable, short petals |
| 🌸 *Iris Versicolor* | Moderate measurements, overlaps with Virginica |
| 🌼 *Iris Virginica* | Longest petals and sepals |

The goal is to understand the trade-offs between a **lazy, instance-based learner** (KNN) and an **eager, rule-based learner** (Decision Tree) on a well-structured real-world dataset.

---

## 📊 Dataset

The **Iris dataset**, introduced by British statistician and biologist **Ronald Fisher** in 1936, is one of the most widely used benchmark datasets in machine learning.

| Feature | Description | Unit |
|---|---|---|
| `sepal_length` | Length of the sepal | cm |
| `sepal_width` | Width of the sepal | cm |
| `petal_length` | Length of the petal | cm |
| `petal_width` | Width of the petal | cm |
| `species` | Target class label | — |

- **Total Samples:** 150 (50 per class)
- **Features:** 4 numerical
- **Classes:** 3
- **Missing Values:** None

> **Source:** [`sklearn.datasets.load_iris()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) / [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)

---

## 🤖 Algorithms

### 🔵 K-Nearest Neighbors (KNN)

A non-parametric, instance-based algorithm that classifies a new data point based on the **majority vote of its K nearest neighbors** in feature space.

```
Distance(A, B) = √Σ(Aᵢ - Bᵢ)²   [Euclidean Distance]
```

**Key Hyperparameter:** `n_neighbors` (K value) — tuned via cross-validation.

**Pros:** Simple, no training phase, naturally handles multi-class  
**Cons:** Slow at prediction time, sensitive to feature scaling and irrelevant features

---

### 🟢 Decision Tree (DT)

A tree-structured model that makes decisions by splitting data on feature thresholds, following a greedy top-down approach using **Gini Impurity** or **Information Gain**.

```
Gini(t) = 1 - Σ p(cᵢ|t)²
```

**Key Hyperparameter:** `max_depth` — controlled to prevent overfitting.

**Pros:** Interpretable, handles non-linear boundaries, no feature scaling needed  
**Cons:** Prone to overfitting, high variance with small datasets

---

## 📁 Project Structure

```
iris_knn_dt/
│
├── iris_knn_dt.ipynb          # Main Jupyter Notebook (EDA + Modeling + Evaluation)
├── iris.csv                   # Dataset (if local copy used)
├── README.md                  # Project documentation
│
└── outputs/                   # Generated plots and results (if any)
    ├── confusion_matrix_knn.png
    ├── confusion_matrix_dt.png
    └── decision_tree_viz.png
```

---

## 📈 Results

| Metric | KNN | Decision Tree |
|---|---|---|
| **Accuracy** | ~96–98% | ~95–97% |
| **Precision (macro)** | ~0.97 | ~0.96 |
| **Recall (macro)** | ~0.97 | ~0.96 |
| **F1-Score (macro)** | ~0.97 | ~0.96 |
| **Training Time** | Very Fast | Fast |
| **Prediction Time** | Slower | Very Fast |
| **Interpretability** | ❌ Low | ✅ High |

> *Exact values depend on train/test split and hyperparameter settings in the notebook.*

### Confusion Matrix Summary

Both models correctly classify **Setosa** with 100% accuracy (it is linearly separable). Minor confusion arises between **Versicolor** and **Virginica** due to feature overlap.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed.

### 1. Clone the Repository

```bash
git clone https://github.com/VishakhaSingh123/Iris_Detection.git
cd Iris_Detection/iris_knn_dt
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Notebook

```bash
jupyter notebook iris_knn_dt.ipynb
```

---

## 📦 Dependencies

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

Install all at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## 💻 Usage

You can run inference directly with trained models:

```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f"KNN Accuracy: {knn.score(X_test, y_test):.4f}")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree Accuracy: {dt.score(X_test, y_test):.4f}")
```

---

## 💡 Key Insights

1. **Petal features dominate** — `petal_length` and `petal_width` are far more discriminative than sepal features for classification.

2. **KNN requires feature scaling** — Since KNN relies on distance metrics, features must be standardized (e.g., using `StandardScaler`) to ensure fair comparison across dimensions.

3. **Decision Trees are inherently interpretable** — The trained tree can be visualized and reasoned about, making it a preferred choice when explainability matters.

4. **Both models achieve high accuracy** — The Iris dataset is relatively simple, making it an excellent benchmark for verifying implementation correctness.

5. **Overfitting risk in DT** — Without depth control (`max_depth`), Decision Trees tend to perfectly memorize training data at the cost of generalization.

---

## 🧪 Experiment Ideas

- [ ] Tune `K` in KNN using cross-validation and plot the elbow curve
- [ ] Visualize the Decision Tree using `sklearn.tree.plot_tree()`
- [ ] Compare with additional algorithms: SVM, Logistic Regression, Random Forest
- [ ] Apply PCA for 2D visualization of class boundaries
- [ ] Try ensemble methods (Bagging, Boosting) to improve accuracy

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](../LICENSE) file for details.

---

## 👩‍💻 Author

**Vishakha Singh**  
📧 [GitHub Profile](https://github.com/VishakhaSingh123)

---

<p align="center">
  Made with ❤️ and Python · If you found this helpful, please ⭐ the repository!
</p>
