import streamlit as st
import pandas as pd
import os
import joblib

from src.data_preprocessing import load_data
from src.eda import show_eda
from src.feature_engineering import process_features
from src.train_model import train_stacking_model
from src.evaluate_model import evaluate_and_display
from src.predict import predict_new_instance

st.set_page_config(page_title="Stacking Regression Hub", layout="wide")

st.title("Stacking Regression: House Price Predictor")

df = load_data()

st.header("1. Dataset Overview")
st.table(df.head())

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.subheader("Data Information")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(df.dtypes.astype(str))
with col_info2:
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

st.header("2. Exploratory Data Analysis")
with st.expander("View EDA Plots"):
    show_eda(df)

st.header("3. Model Training & Evaluation")

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

if st.button("Train Stacking Model"):
    with st.spinner("Processing features and training models... This may take a minute."):
        X_train_s, X_test_s, y_train, y_test = process_features(X, y)
        model = train_stacking_model(X_train_s, y_train)
        
        st.success("Model trained and saved successfully!")
        st.subheader("Evaluation Metrics")
        evaluate_and_display(model, X_test_s, y_test)
elif os.path.exists('models/stacking_model.pkl'):
    st.info("Pre-trained model found. Ready for predictions.")
    
st.header("4. Predict New Data")

if os.path.exists('models/stacking_model.pkl') and os.path.exists('models/scaler.pkl'):
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        med_inc = st.number_input("Median Income", value=4.5)
        house_age = st.number_input("House Age", value=30.0)
        ave_rooms = st.number_input("Average Rooms", value=5.0)
        ave_bedrms = st.number_input("Average Bedrooms", value=1.0)
        
    with col_input2:
        population = st.number_input("Population", value=1400.0)
        ave_occup = st.number_input("Average Occupancy", value=3.0)
        latitude = st.number_input("Latitude", value=34.0)
        longitude = st.number_input("Longitude", value=-118.0)

    if st.button("Predict Price"):
        input_data = {
            'MedInc': med_inc,
            'HouseAge': house_age,
            'AveRooms': ave_rooms,
            'AveBedrms': ave_bedrms,
            'Population': population,
            'AveOccup': ave_occup,
            'Latitude': latitude,
            'Longitude': longitude
        }
        
        pred = predict_new_instance(input_data)
        st.success(f"Estimated Median House Value: ${pred * 100000:,.2f}")
else:
    st.warning("Please train the model first to make predictions.")