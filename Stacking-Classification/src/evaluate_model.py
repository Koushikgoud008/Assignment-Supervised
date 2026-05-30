from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_and_display(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    st.metric("Accuracy Score", f"{accuracy:.4f}")
    
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, predictions))