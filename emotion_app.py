# emotion_app.py

import streamlit as st
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model components
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Emojis dictionary
emoji_dict = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😱",
    "love": "❤️",
    "surprise": "😲"
}

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Emotion Detector", page_icon="🔍")
st.title("😄 Emotion Detection from Text")

user_input = st.text_area("Enter a sentence to detect its emotion:")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess(user_input)
        vectorized_text = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_text)

        predicted_label = label_encoder.inverse_transform([int(prediction[0])])[0]
        emoji = emoji_dict.get(predicted_label.lower(), "🙂")

        st.success(f"🎉 The predicted emotion is: **{predicted_label.upper()} {emoji}**")




