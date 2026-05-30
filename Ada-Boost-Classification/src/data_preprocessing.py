import pandas as pd
import os

def load_and_save_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    df = pd.read_csv(url, header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/banknote.csv', index=False)
    return df