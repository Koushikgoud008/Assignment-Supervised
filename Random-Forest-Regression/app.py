import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.eda import show_eda

st.set_page_config(page_title="House Value Prediction", layout="wide")

st.title("House Value Prediction using Random Forest")

if not os.path.exists('data/housing_rf.csv'):
    from src.data_preprocessing import load_and_save_data
    df = load_and_save_data()
else:
    df = pd.read_csv('data/housing_rf.csv')

st.subheader("Dataset Preview")
st.markdown(df.head().to_html(classes='table table-striped'), unsafe_allow_html=True)

st.subheader("Dataset Information")
st.write(df.shape)
st.text(str(df.dtypes))

st.subheader("Missing Values")
st.text(str(df.isnull().sum()))

show_eda(df)

st.header("Feature Engineering")

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

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

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
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
joblib.dump(model, 'models/rf_reg_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

st.success("Model Saved Successfully")

st.header("Predict New Data")

med_inc = st.number_input('Median Income (in tens of thousands)', value=3.5)
house_age = st.slider('House Age', 1.0, 100.0, 20.0)
ave_rooms = st.number_input('Average Rooms', value=5.0)
ave_bedrms = st.number_input('Average Bedrooms', value=1.0)
population = st.number_input('Population', value=1000.0)
ave_occup = st.number_input('Average Occupation', value=3.0)
latitude = st.number_input('Latitude', value=35.0)
longitude = st.number_input('Longitude', value=-120.0)

input_data = np.array([[
    med_inc, house_age, ave_rooms, ave_bedrms, 
    population, ave_occup, latitude, longitude
]])

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    predicted_value = prediction[0] * 100000
    st.success(f"Predicted Median House Value: ${predicted_value:,.2f}")