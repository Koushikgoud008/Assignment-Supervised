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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.eda import show_eda

st.set_page_config(page_title="Cancer Classification", layout="wide")

st.title("Breast Cancer Classification using SVC")

if not os.path.exists('data/cancer.csv'):
    from src.data_preprocessing import load_and_save_data
    df = load_and_save_data()
else:
    df = pd.read_csv('data/cancer.csv')


st.subheader("Dataset Preview")
# Use st.table instead of st.dataframe to bypass PyArrow completely
st.table(df.head())

st.subheader("Dataset Information")
st.write(df.shape)
st.write(df.dtypes)

st.subheader("Missing Values")
st.write(df.isnull().sum())

show_eda(df)

st.header("Feature Engineering")

X = df.drop('target', axis=1)
y = df['target']

st.write("Features Shape:", X.shape)
st.write("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Training Data Shape:", X_train_scaled.shape)
st.write("Testing Data Shape:", X_test_scaled.shape)

model = SVC(kernel='rbf', C=1.0, probability=True)
model.fit(X_train_scaled, y_train)

st.success("Model Training Completed")

y_pred = model.predict(X_test_scaled)

st.header("Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy Score: {accuracy:.4f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/svc_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

st.success("Model Saved Successfully")

st.header("Predict New Data")

mean_radius = st.number_input('Mean Radius', value=14.0)
mean_texture = st.number_input('Mean Texture', value=19.0)
mean_perimeter = st.number_input('Mean Perimeter', value=90.0)
mean_area = st.number_input('Mean Area', value=650.0)
mean_smoothness = st.number_input('Mean Smoothness', value=0.1)

input_data = np.zeros((1, 30))
input_data[0, 0] = mean_radius
input_data[0, 1] = mean_texture
input_data[0, 2] = mean_perimeter
input_data[0, 3] = mean_area
input_data[0, 4] = mean_smoothness

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 0:
        st.error("Prediction: Malignant (0)")
    else:
        st.success("Prediction: Benign (1)")