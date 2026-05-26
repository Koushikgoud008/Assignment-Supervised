from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def train(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/knnc_model.pkl')
    return model