import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from src.eda import show_eda

st.set_page_config(page_title="Auto MPG Prediction", layout="wide")

st.title("Vehicle MPG Prediction using AdaBoost & GridSearchCV")

if not os.path.exists('data/mpg.csv'):
    from src.data_preprocessing import load_and_save_data
    df = load_and_save_data()
else:
    df = pd.read_csv('data/mpg.csv')

st.subheader("Dataset Preview")
st.markdown(df.head().to_html(classes='table table-striped'), unsafe_allow_html=True)

st.subheader("Dataset Information")
st.write(df.shape)
st.text(str(df.dtypes))

st.subheader("Missing Values")
st.text(str(df.isnull().sum()))

show_eda(df)

st.header("Feature Engineering")

X = df.drop('mpg', axis=1)
y = df['mpg']

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
    base_model = AdaBoostRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

st.success("Model Training Completed")
st.write(f"Best Hyperparameters Found: {best_params}")

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
ax2.set_xlabel('Actual MPG')
ax2.set_ylabel('Predicted MPG')
st.pyplot(fig2)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/adaboost_reg_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

st.success("Model Saved Successfully")

st.header("Predict New Data")

cylinders = st.selectbox('Cylinders', [3, 4, 5, 6, 8], index=1)
displacement = st.number_input('Displacement (cu. inches)', value=100.0)
horsepower = st.number_input('Horsepower', value=90.0)
weight = st.number_input('Weight (lbs)', value=2500.0)
acceleration = st.number_input('Acceleration (sec 0-60)', value=15.0)
model_year = st.slider('Model Year (19XX)', 70, 82, 75)

input_data = np.array([[
    cylinders, displacement, horsepower, weight, acceleration, model_year
]])

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Fuel Efficiency: {prediction[0]:.2f} MPG")