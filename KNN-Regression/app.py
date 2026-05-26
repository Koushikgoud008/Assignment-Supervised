import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.eda import show_eda

st.set_page_config(page_title="Diabetes Progression", layout="wide")

st.title("Diabetes Progression using KNN Regression")

if not os.path.exists('data/diabetes.csv'):
    from src.data_preprocessing import load_and_save_data
    df = load_and_save_data()
else:
    df = pd.read_csv('data/diabetes.csv')

st.subheader("Dataset Preview")
st.markdown(df.head().to_html(classes='table table-striped'), unsafe_allow_html=True)

st.subheader("Dataset Information")
st.write(df.shape)
st.text(str(df.dtypes))

st.subheader("Missing Values")
st.text(str(df.isnull().sum()))

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

model = KNeighborsRegressor(n_neighbors=5, weights='distance')
model.fit(X_train_scaled, y_train)

st.success("Model Training Completed")

y_pred = model.predict(X_test_scaled)

st.header("Model Evaluation")

mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.4f}")

r2 = r2_score(y_test, y_pred)
st.write(f"R-squared Score: {r2:.4f}")

st.subheader("Actual vs Predicted")
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.5)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax2.set_xlabel('Actual Value')
ax2.set_ylabel('Predicted Value')
st.pyplot(fig2)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/knn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

st.success("Model Saved Successfully")

st.header("Predict New Data")

age = st.number_input('Age (Scaled)', value=0.038)
sex = st.number_input('Sex (Scaled)', value=0.05)
bmi = st.number_input('BMI (Scaled)', value=0.061)
bp = st.number_input('Blood Pressure (Scaled)', value=0.021)
s1 = st.number_input('S1 (Scaled)', value=-0.044)
s2 = st.number_input('S2 (Scaled)', value=-0.034)
s3 = st.number_input('S3 (Scaled)', value=-0.043)
s4 = st.number_input('S4 (Scaled)', value=-0.002)
s5 = st.number_input('S5 (Scaled)', value=0.019)
s6 = st.number_input('S6 (Scaled)', value=-0.017)

input_data = np.array([[
    age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
]])

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Disease Progression: {prediction[0]:.2f}")