import pandas as pd
import nltk
import string
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("emotion_dataset.csv")
print("Columns:", df.columns)

# Check if 'text' and 'label' columns exist
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Required columns 'text' and 'label' not found in dataset.")

# Text preprocessing function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['text'].astype(str).apply(preprocess_text)

# Label Encoding (handle both string and numeric labels)
if df['label'].dtype == 'object':
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    target_names = label_encoder.classes_
else:
    df['label_encoded'] = df['label']
    label_encoder = None  # Not needed
    target_names = [str(i) for i in sorted(df['label'].unique())]

# Features and Labels
X = df['clean_text']
y = df['label_encoded']

# Vectorize
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# Save model
with open("emotion_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

if label_encoder:
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

print("✅ Model, vectorizer, and encoder saved successfully.")

