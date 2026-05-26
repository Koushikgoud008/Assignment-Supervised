import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):
    st.header("Exploratory Data Analysis")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['target'], bins=30, kde=True, ax=ax)
    ax.set_title("Target Distribution (Disease Progression)")
    st.pyplot(fig)