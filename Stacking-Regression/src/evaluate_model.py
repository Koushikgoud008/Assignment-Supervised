from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import streamlit as st

def evaluate_and_display(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")