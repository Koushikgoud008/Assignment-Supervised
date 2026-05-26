from sklearn.tree import DecisionTreeRegressor
import joblib
import os

def train(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/dt_reg_model.pkl')
    return model