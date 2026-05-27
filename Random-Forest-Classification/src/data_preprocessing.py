import pandas as pd
import seaborn as sns
import os

def load_and_save_data():
    df = sns.load_dataset('penguins')
    df = df.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species'])
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/penguins.csv', index=False)
    return df