import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def load_data():
    if os.path.exists('data/housing.csv'):
        return pd.read_csv('data/housing.csv')
    
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/housing.csv', index=False)
    
    return df