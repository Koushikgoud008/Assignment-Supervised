import pandas as pd
from sklearn.datasets import load_diabetes
import os

def load_and_save_data():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/diabetes.csv', index=False)
    return df