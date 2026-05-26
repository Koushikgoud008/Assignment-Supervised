from sklearn.svm import SVC
import joblib
import os

def train(X_train, y_train):
    model = SVC(kernel='rbf', C=1.0, probability=True)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/svc_model.pkl')
    return model