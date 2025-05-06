# train_and_save_models.py

import pandas as pd
import re
import string
import nltk
import pickle
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')

# Load your dataset
# Replace this with your actual dataset path and structure
df = pd.read_csv("emotion_dataset.csv")  # Assume columns: 'text', 'emotion'

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return " ".join(tokens)

# Preprocess text data
df['clean_text'] = df['text'].apply(preprocess)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['emotion'])

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Save the vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "emotion_model.pkl")

print("✅ Training complete. Model, vectorizer, and label encoder saved.")



