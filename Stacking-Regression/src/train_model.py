from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
import joblib
import os

def train_stacking_model(X_train, y_train):
    base_models = [
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('svr', SVR(kernel='rbf', C=1.0))
    ]
    
    meta_model = LinearRegression()
    
    model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stacking_model.pkl')
    
    return model