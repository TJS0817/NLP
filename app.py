import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize NLTK
nltk.download('punkt')

st.title("NLP-Based Product Recommendation Model")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Overview")
    st.write(df.head())
    
    # Data exploration
    if st.checkbox("Show Data Description"):
        st.write(df.describe())
    if st.checkbox("Show Data Info"):
        st.write(df.dtypes)
    
    # Preprocessing
    st.subheader("Data Preprocessing")
    text_column = st.selectbox("Select text column for NLP processing", df.columns)
    
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in tokens])
    
    df[text_column] = df[text_column].astype(str).apply(preprocess_text)
    st.write("Processed Sample:", df[text_column].head())

    # Feature Engineering
    st.subheader("TF-IDF Feature Extraction")
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df[text_column])
    
    label_column = st.selectbox("Select target column for classification", df.columns)
    le = LabelEncoder()
    y = le.fit_transform(df[label_column])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection
    model_choice = st.selectbox("Select Classification Model", ["Logistic Regression", "Decision Tree"])
    
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = DecisionTreeClassifier()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model Evaluation
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
