# 🌸 Iris Species Detection: KNN vs Decision Tree

An end-to-end Machine Learning pipeline and interactive web dashboard built to classify Iris flowers. This project compares two popular classification algorithms: **K-Nearest Neighbors (KNN)** and a **Decision Tree**.

## ✨ Features
- **Exploratory Data Analysis (EDA):** Scripts that automatically generate insights into the dataset, producing beautiful Violin Plots, Pairplots, and Correlation Heatmaps to map out class separability.
- **Model Training & Tuning:** Automated hyperparameter tuning via `GridSearchCV` to find the optimal configurations for both models (e.g., discovering the best `K` for KNN).
- **Direct Comparison:** Side-by-side performance evaluation measuring Accuracy, F1 Scores, Cross-Validation distributions, and mapping feature importances.
- **Interactive UI Dashboard:** A fully functional Streamlit frontend that allows users to explore the data, view the model metrics, and use dynamic sliders to instantly classify their own custom flower measurements!

## 🚀 How to Run Locally

### 1. Requirements
Ensure you have Python installed, then install the required dependencies:
```bash
pip install -r iris_knn_dt/requirements.txt
```

### 2. View the Interactive Dashboard
To launch the Streamlit web application:
```bash
cd iris_knn_dt
streamlit run app.py
```
*This will open a new tab in your web browser with the interactive playground!*

### 3. Run the ML Pipeline directly
If you want to train the models and generate the evaluation plots directly from the terminal without the web interface:
```bash
cd iris_knn_dt
python main.py
```

## 📁 Project Structure
- `iris_knn_dt/app.py`: The main Streamlit web interface.
- `iris_knn_dt/data_loader.py`: Handles loading the sklearn dataset, splitting it into training/testing sets, and applying necessary feature scaling.
- `iris_knn_dt/eda.py`: Generates all statistical visualizations and plots.
- `iris_knn_dt/models.py`: Defines the training pipeline and GridSearchCV optimization for both algorithms.
- `iris_knn_dt/evaluation.py`: Calculates model metrics (Accuracy/F1), plots confusion matrices, and extracts feature importances.
- `iris_knn_dt/main.py`: The master script to run the entire backend pipeline.

## 📊 Dataset
This project uses the famous **[Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)** (often credited to Ronald Fisher). It includes 150 samples across 3 different species of Iris (*Setosa*, *Versicolor*, and *Virginica*) measured by 4 distinct features:
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)
