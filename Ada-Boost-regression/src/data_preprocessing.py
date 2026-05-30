import pandas as pd
import seaborn as sns
import os

def load_and_save_data():
    df = sns.load_dataset('mpg')
    df = df.dropna()
    df = df.drop(['name', 'origin'], axis=1)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/mpg.csv', index=False)
    return df