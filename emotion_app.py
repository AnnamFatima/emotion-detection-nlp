import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Mapping: model's numeric output → emotion label
emotion_map = {
    0: 'joy',
    1: 'sadness',
    2: 'anger',
    3: 'fear',
    4: 'disgust',
    5: 'neutral'
}

# Emoji for each emotion
emojis = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
}

# Text cleaner (optional, helps detect junk inputs)
def is_meaningful(text):
    return bool(re.search(r'[a-zA-Z]', text))

# Streamlit UI
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("🧠 Emotion Detection from Text")
st.markdown("Enter a sentence to find out the emotion behind it!")

# Input
text_input = st.text_area("Enter your message here:")

if st.button("Detect Emotion"):
    cleaned = text_input.strip()
    if cleaned == "":
        st.warning("⚠️ Please enter some text.")
    elif not is_meaningful(cleaned):
        st.warning("⚠️ Input must contain alphabetic characters.")
    else:
        try:
            vectorized_input = vectorizer.transform([cleaned])
            if vectorized_input.nnz == 0:  # No non-zero features
                st.warning("⚠️ Input does not contain meaningful words.")
            else:
                prediction = model.predict(vectorized_input)
                pred_int = int(prediction[0])

                emotion = emotion_map.get(pred_int, str(pred_int))
                emoji = emojis.get(emotion.lower(), '')

                st.markdown(f"### 🎯 Detected Emotion: **{emotion.capitalize()}** {emoji}")
                st.write("Raw Model Output:", pred_int)
        except Exception as e:
            st.error(f"🚫 Unexpected error: {e}")


