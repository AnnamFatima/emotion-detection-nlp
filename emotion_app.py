import streamlit as st
import joblib
import re
import base64
from preprocessing import preprocess_text  # Import your preprocessing function

# Set Streamlit page config
st.set_page_config(page_title="Emotion Detection App", layout="centered")

# Load and apply background
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Path to your background image
set_background('c:\\Users\\Annam Fatima Shaikh\\Downloads\\ChatGPT Image May 10, 2025, 05_45_26 PM.png')

# Load the trained model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Mapping from model's numeric predictions to labels
emotion_map = {
    0: 'joy',
    1: 'sadness',
    2: 'anger',
    3: 'fear',
    4: 'disgust',
    5: 'neutral'
}

# Emojis for each emotion
emojis = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
}

# Utility to check if input contains valid characters
def is_meaningful(text):
    return bool(re.search(r'[a-zA-Z]', text))

# UI Layout
st.title("🧠 Emotion Detection from Text")
st.markdown("Enter a sentence to find out the emotion behind it!")

text_input = st.text_area("Enter your message here:")

# Handle Prediction
if st.button("Detect Emotion"):
    cleaned = preprocess_text(text_input.strip())

    if cleaned == "":
        st.warning("⚠️ Please enter some text.")
    elif not is_meaningful(cleaned):
        st.warning("⚠️ Input must contain alphabetic characters.")
    else:
        try:
            vectorized_input = vectorizer.transform([cleaned])

            if vectorized_input.nnz == 0:
                st.warning("⚠️ Input does not contain meaningful words.")
            else:
                prediction = model.predict(vectorized_input)
                pred_int = int(prediction[0])
                emotion = emotion_map.get(pred_int, str(pred_int))
                emoji = emojis.get(emotion.lower(), '')

                st.markdown(f"### 🎯 Detected Emotion: **{emotion.capitalize()}** {emoji}")

                # Optional: Save logs
                with open("prediction_log.txt", "a") as f:
                    f.write(f"{text_input} -> {emotion} ({pred_int})\n")

                st.markdown("Was this prediction helpful?")
                st.button("👍 Yes")
                st.button("👎 No")

        except Exception as e:
            st.error(f"🚫 Unexpected error: {e}")






