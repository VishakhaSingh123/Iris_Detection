import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# Import existing functions
from data_loader import load_data
from models import train_knn, train_decision_tree
from evaluation import evaluate_model

# Page config
st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸", layout="wide")

# ==========================================
# 1. Caching Data and Models
# ==========================================
@st.cache_resource
def get_data_and_models():
    """Load data and train models once in the background to make the app snappy."""
    data = load_data(test_size=0.2, random_state=42)
    # Train Models
    knn = train_knn(data["X_train_scaled"], data["y_train"])
    dt = train_decision_tree(data["X_train"], data["y_train"])
    # Evaluate Models for Metrics
    knn_res = evaluate_model(knn, data["X_test_scaled"], data["y_test"], data["class_names"], "KNN")
    dt_res = evaluate_model(dt, data["X_test"], data["y_test"], data["class_names"], "Decision Tree")
    return data, knn, dt, knn_res, dt_res

data, knn, dt, knn_res, dt_res = get_data_and_models()

# Helper function to safely load generated plots
def load_safe_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

# ==========================================
# 2. Sidebar Navigation
# ==========================================
st.sidebar.title("Navigation")
st.sidebar.markdown("Navigate through the project steps:")
page = st.sidebar.radio(
    "",
    ["Explore Data 📊", "Model Evaluation ⚙️", "Make Prediction 🔮"]
)

# ==========================================
# 3. Main Pages
# ==========================================
if page == "Explore Data 📊":
    st.title("Explore the Iris Dataset 📊")
    st.markdown("Welcome! This dashboard explains the classic Iris classification problem comparing **K-Nearest Neighbors (KNN)** and a **Decision Tree** algorithm.")
    
    st.subheader("Raw Dataset Sample")
    st.dataframe(data["df"].head(15), use_container_width=True)
    
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown("These plots reveal how features correlate and how separable the classes are:")
    
    col1, col2 = st.columns(2)
    
    img_class_dist = load_safe_image("eda_class_distribution.png")
    img_violin     = load_safe_image("eda_violin.png")
    img_pairplot   = load_safe_image("eda_pairplot.png")
    img_corr       = load_safe_image("eda_correlation.png")
    
    if img_class_dist: col1.image(img_class_dist, caption="Class Distribution", use_container_width=True)
    if img_violin:     col2.image(img_violin, caption="Feature Distributions by Class", use_container_width=True)
    if img_pairplot:   st.image(img_pairplot, caption="Feature Pairplot", use_container_width=True)
    if img_corr:       st.image(img_corr, caption="Correlation Matrix Heatmap", use_container_width=True)
    
    if not (img_class_dist or img_violin or img_pairplot or img_corr):
        st.warning("⚠️ High-resolution EDA plots not found! Please run `python main.py` in your terminal to generate them.")

elif page == "Model Evaluation ⚙️":
    st.title("Model Metrics & Comparison ⚙️")
    st.markdown(f"The models were trained on {len(data['X_train'])} samples and evaluated on {len(data['X_test'])} unseen samples.")
    
    # Metrics display
    col_k, col_d = st.columns(2)
    with col_k:
        st.info("### K-Nearest Neighbors (KNN)")
        st.metric(label="Accuracy", value=f"{knn_res['accuracy']*100:.2f}%")
        st.metric(label="F1 Score", value=f"{knn_res['f1']:.4f}")
    with col_d:
        st.success("### Decision Tree")
        st.metric(label="Accuracy", value=f"{dt_res['accuracy']*100:.2f}%")
        st.metric(label="F1 Score", value=f"{dt_res['f1']:.4f}")

    st.markdown("---")
    st.subheader("Evaluation Visualizations")
    
    img_cm       = load_safe_image("confusion_matrices.png")
    img_fi       = load_safe_image("feature_importance.png")
    img_cv_comp  = load_safe_image("cv_comparison.png")
    img_acc_comp = load_safe_image("accuracy_f1_comparison.png")
    
    if img_acc_comp: st.image(img_acc_comp, use_container_width=True)
    if img_cm:       st.image(img_cm, use_container_width=True)
    
    col1, col2 = st.columns(2)
    if img_fi:       col1.image(img_fi, caption="Gini Feature Importance (DT)", use_container_width=True)
    if img_cv_comp:  col2.image(img_cv_comp, caption="5-Fold Cross Validation Distributions", use_container_width=True)
    
    if not (img_cm or img_fi or img_cv_comp or img_acc_comp):
        st.warning("⚠️ Evaluation plots not found! Please run `python main.py` in your terminal to generate them.")

elif page == "Make Prediction 🔮":
    st.title("Predict Iris Species 🔮")
    st.markdown("Use the sliders below to enter custom flower measurements, and the models will classify it instantly!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Features")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
        petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.3, 0.1)
        
    with col2:
        st.subheader("Live Predictions")
        
        # Prepare the input for prediction
        input_raw = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # We must scale the input for the KNN model exactly how the training data was scaled.
        # DataLoader fitted a scaler implicitly during load_data. However, we only have 'X_train_scaled'.
        # Since we need the scaler transform, we should recreate it or re-fit it exactly on training data.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(data["X_train"])  # Fit fresh scaler to recreate the identical transform
        input_scaled = scaler.transform(input_raw)
        
        # Predict using KNN
        knn_pred_idx = knn.predict(input_scaled)[0]
        knn_pred_probs = knn.predict_proba(input_scaled)[0]
        
        # Predict using Decision Tree
        dt_pred_idx = dt.predict(input_raw)[0]
        dt_pred_probs = dt.predict_proba(input_raw)[0]
        
        classes = data["class_names"]
        
        st.info(f"**KNN Prediction:** `{classes[knn_pred_idx].capitalize()}`\n\nProbability: {knn_pred_probs[knn_pred_idx]*100:.1f}%")
        st.success(f"**Decision Tree Prediction:** `{classes[dt_pred_idx].capitalize()}`\n\nProbability: {dt_pred_probs[dt_pred_idx]*100:.1f}%")
        
        # Fun image (streamlit logo or flower emoji)
        st.markdown(f"### The flower is likely a... **{classes[dt_pred_idx].upper()}**!")
