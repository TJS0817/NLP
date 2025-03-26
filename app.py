import streamlit as st
import pandas as pd
import pickle
import os

# Load models and preprocessing tools
@st.cache_data  # Cache loaded models and tools to improve performance
def load_models():
    # Define model paths
    model_paths = {
        "Decision Tree": "Decision Tree_model.pkl",
        "Linear SVC": "Linear SVC_model.pkl",
        "Random Forest": "Random Forest_model.pkl"
    }
    
    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            with open(path, "rb") as file:
                models[name] = pickle.load(file)
        else:
            st.error(f"Model file {path} not found!")
            return None, None, None  # Stop loading if any model is missing

    # Load vectorizer and scaler
    vectorizer_path = "tfidf_vectorizer.pkl"
    scaler_path = "scaler.pkl"
    if os.path.exists(vectorizer_path) and os.path.exists(scaler_path):
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)
        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)
    else:
        st.error("Vectorizer or Scaler file not found!")
        return None, None, None  # Stop loading if preprocessing tools are missing

    return models, vectorizer, scaler

models, vectorizer, scaler = load_models()

# Check if models and preprocessing tools are loaded successfully
if not models or vectorizer is None or scaler is None:
    st.error("Failed to load models or preprocessing tools. Please check the file paths.")
    st.stop()  # Stop execution if models or tools are missing

# Title and Description
st.title("🧠 Mental Health Prediction Tool")
st.write("""
This tool predicts mental health categories based on text input.
Enter a statement describing your symptoms or feelings, and the app will predict the corresponding mental health category.
""")

# User selects the model
selected_model_name = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[selected_model_name]

# User Input Form
user_input = st.text_area("Enter Your Statement (e.g., symptoms or feelings):", height=150)

# Predict Mental Health State
if st.button("Predict Mental Health Category"):
    if user_input.strip() == "":
        st.error("Please provide valid input.")
    else:
        try:
            # Preprocess the input text
            user_input_vectorized = vectorizer.transform([user_input])  # Vectorize the text
            user_input_scaled = scaler.transform(user_input_vectorized)  # Scale the vectorized input
            
            # Make prediction
            prediction = selected_model.predict(user_input_scaled)[0]
            
            # Display the result
            st.subheader(f"Predicted Mental Health Category: **{prediction}** (Using {selected_model_name})")
            
            # Display confidence scores if supported
            if hasattr(selected_model, "predict_proba"):
                probabilities = selected_model.predict_proba(user_input_scaled)[0]
                categories = selected_model.classes_
                confidence_scores = {cat: prob for cat, prob in zip(categories, probabilities)}
                
                st.write("Confidence Scores:")
                for category, score in confidence_scores.items():
                    st.write(f"{category}: {score:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
