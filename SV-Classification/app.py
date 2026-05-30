import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from src.eda import show_eda

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("Breast Cancer Classification using SVC")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if not os.path.exists("data/cancer.csv"):
    from src.data_preprocessing import load_and_save_data
    df = load_and_save_data()
else:
    df = pd.read_csv("data/cancer.csv")

# --------------------------------------------------
# CLEAN DATA TYPES
# --------------------------------------------------
for col in df.columns:
    if col != "target":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------
st.header("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# DATASET INFO
# --------------------------------------------------
st.header("Dataset Information")

st.write("Shape:", df.shape)

st.text("Column Data Types")
st.text(df.dtypes.to_string())

# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------
st.header("Missing Values")
st.text(df.isnull().sum().to_string())

# --------------------------------------------------
# EDA
# --------------------------------------------------
try:
    show_eda(df)
except Exception as e:
    st.error(f"EDA Error: {e}")

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
st.header("Feature Engineering")

X = df.drop("target", axis=1)
y = df["target"]

st.write("Feature Shape:", X.shape)
st.write("Target Shape:", y.shape)

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# SCALING
# --------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Training Shape:", X_train_scaled.shape)
st.write("Testing Shape:", X_test_scaled.shape)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
model = SVC(
    kernel="rbf",
    C=1.0,
    probability=True
)

model.fit(X_train_scaled, y_train)

st.success("Model Training Completed")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
y_pred = model.predict(X_test_scaled)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
st.header("Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{accuracy:.4f}")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------
st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/svc_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

st.success("Model and Scaler Saved")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
st.header("Predict New Data")

mean_radius = st.number_input(
    "Mean Radius",
    value=14.0
)

mean_texture = st.number_input(
    "Mean Texture",
    value=19.0
)

mean_perimeter = st.number_input(
    "Mean Perimeter",
    value=90.0
)

mean_area = st.number_input(
    "Mean Area",
    value=650.0
)

mean_smoothness = st.number_input(
    "Mean Smoothness",
    value=0.10
)

# --------------------------------------------------
# CREATE INPUT ROW
# --------------------------------------------------
input_data = X.mean().values.reshape(1, -1)

input_data[0, 0] = mean_radius
input_data[0, 1] = mean_texture
input_data[0, 2] = mean_perimeter
input_data[0, 3] = mean_area
input_data[0, 4] = mean_smoothness

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------
if st.button("Predict"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.error("Prediction: Malignant Cancer")
    else:
        st.success("Prediction: Benign Cancer")