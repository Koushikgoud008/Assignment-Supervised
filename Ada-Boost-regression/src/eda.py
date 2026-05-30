import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):
    st.header("Exploratory Data Analysis")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['mpg'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)