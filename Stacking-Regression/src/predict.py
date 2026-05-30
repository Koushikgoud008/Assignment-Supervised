import joblib
import pandas as pd

def predict_new_instance(input_dict):
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/stacking_model.pkl')
    
    df = pd.DataFrame([input_dict])
    scaled_data = scaler.transform(df)
    
    prediction = model.predict(scaled_data)
    return prediction[0]