import joblib
import pandas as pd

def predict_new_instance(input_dict):
    scaler = joblib.load('models/scaler_clf.pkl')
    model = joblib.load('models/stacking_classifier.pkl')
    
    df = pd.DataFrame([input_dict])
    scaled_data = scaler.transform(df)
    
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    return prediction, probabilities