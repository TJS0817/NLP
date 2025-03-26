import streamlit as st
import pandas as pd
import pickle
import os

# Load models and encoder
@st.cache_data
def load_models():
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

    # Load encoder
    encoder_path = "encoder.pkl"
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as file:
            encoder = pickle.load(file)
    else:
        st.error("Encoder file not found!")
        encoder = None

    return models, encoder

models, encoder = load_models()

if not models or encoder is None:
    st.stop()  # Stop execution if models or encoder is missing

st.title("Mental Health Prediction Tool")
st.write("This tool predicts mental health based on social media usage patterns.")

# User selects the model
selected_model_name = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[selected_model_name]

# User Input Form
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
platform = st.selectbox("Social Media Platform", ["Instagram", "Twitter", "Facebook", "LinkedIn", "Snapchat", "WhatsApp", "Telegram"])
daily_usage = st.slider("Daily Usage Time (minutes)", 0, 600, 120)
posts_per_day = st.slider("Posts Per Day", 0, 50, 5)
likes_per_day = st.slider("Likes Received Per Day", 0, 500, 50)
comments_per_day = st.slider("Comments Received Per Day", 0, 200, 10)
messages_per_day = st.slider("Messages Sent Per Day", 0, 500, 50)

# Encode categorical data
input_data = pd.DataFrame([[age, gender, platform, daily_usage, posts_per_day, likes_per_day, comments_per_day, messages_per_day]],
                          columns=["Age", "Gender", "Platform", "Daily_Usage_Time", "Posts_Per_Day", "Likes_Received_Per_Day", "Comments_Received_Per_Day", "Messages_Sent_Per_Day"])
input_data[["Gender", "Platform"]] = encoder.transform(input_data[["Gender", "Platform"]])

# Predict Mental Health State
if st.button("Predict Mental Health State"):
    prediction = selected_model.predict(input_data)
    st.subheader(f"Predicted Mental Health State: {prediction[0]} (Using {selected_model_name})")
