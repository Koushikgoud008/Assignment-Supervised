from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_clf_model.pkl')
    return model