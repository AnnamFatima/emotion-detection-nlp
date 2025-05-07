import pandas as pd
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load dataset
df = pd.read_csv("emotion_dataset.csv")

# If labels are numeric, map them to string
if df['label'].dtype != 'object':
    label_map = {
        0: 'joy', 1: 'sadness', 2: 'anger',
        3: 'fear', 4: 'disgust', 5: 'neutral'
    }
    df['label'] = df['label'].map(label_map)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Save label encoder
joblib.dump(le, "label_encoder.pkl")

# Preprocess text
df['clean_text'] = df['text'].apply(preprocess_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Save vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "emotion_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

