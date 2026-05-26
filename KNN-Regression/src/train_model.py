from sklearn.neighbors import KNeighborsRegressor
import joblib
import os

def train(X_train, y_train):
    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/knn_model.pkl')
    return model