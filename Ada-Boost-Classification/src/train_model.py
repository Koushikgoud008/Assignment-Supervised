from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_with_gridsearch(X_train, y_train):
    base_model = AdaBoostClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/adaboost_clf_model.pkl')
    
    return best_model, best_params