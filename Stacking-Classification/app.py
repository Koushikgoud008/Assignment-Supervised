import streamlit as st
import pandas as pd
import os
import joblib

from src.data_preprocessing import load_data
from src.eda import show_eda
from src.feature_engineering import process_features
from src.train_model import train_stacking_classifier
from src.evaluate_model import evaluate_and_display
from src.predict import predict_new_instance

st.set_page_config(page_title="Stacking Classification Hub", layout="wide")

st.title("Stacking Classification: Cancer Predictor")

df = load_data()

st.header("1. Dataset Overview")
# st.table is safe and bypasses Arrow for dataframe rendering
st.table(df.head())

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.subheader("Data Information")
    st.text(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    # FIX: Using st.text forces raw console output, entirely bypassing PyArrow
    st.text(df.dtypes)
with col_info2:
    st.subheader("Missing Values")
    # FIX: Applying st.text here as well to prevent identical Arrow crashes on the Series
    st.text(df.isnull().sum())

st.header("2. Exploratory Data Analysis")
with st.expander("View EDA Plots"):
    show_eda(df)

st.header("3. Model Training & Evaluation")

X = df.drop('target', axis=1)
y = df['target']

if st.button("Train Stacking Classifier"):
    with st.spinner("Processing features and training models..."):
        X_train_s, X_test_s, y_train, y_test = process_features(X, y)
        model = train_stacking_classifier(X_train_s, y_train)
        
        st.success("Model trained and saved successfully!")
        st.header("Evaluation Metrics")
        evaluate_and_display(model, X_test_s, y_test)
elif os.path.exists('models/stacking_classifier.pkl'):
    st.info("Pre-trained model found. Ready for prediction updates.")

st.header("4. Predict New Instance")

if os.path.exists('models/stacking_classifier.pkl') and os.path.exists('models/scaler_clf.pkl'):
    input_data = {}
    cols = st.columns(5)
    
    # Dynamically generate input fields based on the dataset features
    for i, column in enumerate(X.columns):
        with cols[i % 5]:
            input_data[column] = st.number_input(f"{column}", value=float(df[column].mean()))
            
    if st.button("Run Classification"):
        pred, proba = predict_new_instance(input_data)
        
        st.subheader("Result")
        if pred == 0:
            st.error(f"Prediction: Malignant (0) | Confidence: {proba[0]*100:.2f}%")
        else:
            st.success(f"Prediction: Benign (1) | Confidence: {proba[1]*100:.2f}%")
else:
    st.warning("Please train the model first to unlock the prediction interface.")