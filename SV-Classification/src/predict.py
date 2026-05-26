import joblib

def predict_new_data(X_new, scaler):
    model = joblib.load('models/svc_model.pkl')
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    return predictions