import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def load_data():
    if os.path.exists('data/cancer.csv'):
        return pd.read_csv('data/cancer.csv')
    
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/cancer.csv', index=False)
    
    return df