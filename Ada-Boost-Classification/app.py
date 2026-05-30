import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.eda import show_eda

st.set_page_config(page_title="Banknote Authentication", layout="wide")

st.title("Banknote Forgery Detection using AdaBoost")

if not os.path.exists('data/banknote.csv'):
    from src.data_preprocessing import load_and_save_data
    df = load_and_save_data()
else:
    df = pd.read_csv('data/banknote.csv')

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

with st.spinner('Running GridSearchCV for Hyperparameter Tuning...'):
    base_model = AdaBoostClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

st.success("Model Training Completed")
st.write(f"Best Hyperparameters Found: {best_params}")

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
joblib.dump(model, 'models/adaboost_clf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

st.success("Model Saved Successfully")

st.header("Predict New Data")

variance = st.number_input('Variance of Wavelet Transformed Image', value=3.6)
skewness = st.number_input('Skewness of Wavelet Transformed Image', value=8.6)
curtosis = st.number_input('Curtosis of Wavelet Transformed Image', value=-2.8)
entropy = st.number_input('Entropy of Image', value=-1.1)

input_data = np.array([[variance, skewness, curtosis, entropy]])

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 0:
        st.success("Prediction: Genuine Banknote (0)")
    else:
        st.error("Prediction: Forged Banknote (1)")