import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):
    st.subheader("Target Class Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='target', data=df, ax=ax)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + 0.35, p.get_height() + 5))
    st.pyplot(fig)
    
    st.subheader("Mean Radius vs Mean Texture by Target")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='mean radius', y='mean texture', hue='target', data=df, alpha=0.7, ax=ax2)
    st.pyplot(fig2)