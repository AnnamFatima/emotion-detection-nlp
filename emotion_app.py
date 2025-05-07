import streamlit as st
import joblib

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

# Streamlit UI
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("🧠 Emotion Detection from Text")
st.markdown("Enter a sentence to find out the emotion behind it!")

# Input
text_input = st.text_area("Enter your message here:")

if st.button("Detect Emotion"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Vectorize and predict
        vectorized_input = vectorizer.transform([text_input])
        prediction = model.predict(vectorized_input)
        pred_int = int(prediction[0])

        # Map prediction to emotion
        emotion = emotion_map.get(pred_int, str(pred_int))
        emoji = emojis.get(emotion.lower(), '')

        # Show result
        st.markdown(f"### 🎯 Detected Emotion: **{emotion.capitalize()}** {emoji}")
        st.write("Raw Model Output:", pred_int)




