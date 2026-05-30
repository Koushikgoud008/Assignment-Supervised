import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.subheader("Target Distribution (House Prices)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(df['MedHouseVal'], bins=50, kde=True, color='blue', ax=ax2)
    st.pyplot(fig2)