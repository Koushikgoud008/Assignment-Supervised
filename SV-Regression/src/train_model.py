from sklearn.svm import SVR
import joblib
import os

def train(X_train, y_train):
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/svr_model.pkl')
    return model