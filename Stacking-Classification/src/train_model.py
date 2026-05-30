from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_stacking_classifier(X_train, y_train):
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ]
    
    meta_model = LogisticRegression(random_state=42)
    
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stacking_classifier.pkl')
    
    return model