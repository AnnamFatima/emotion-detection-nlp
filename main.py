import pandas as pd

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("emotion_dataset.csv")  # Ensure this has 'text' and 'label' columns

print("Available columns:", df.columns)
print(df.head())

# Encode emotion labels to string labels if not already
if df['label'].dtype != 'object':
    label_map = {
        0: 'joy', 1: 'sadness', 2: 'anger',
        3: 'fear', 4: 'disgust', 5: 'neutral'
    }
    df['label'] = df['label'].map(label_map)

# Encode labels using LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Save the label encoder
joblib.dump(le, "label_encoder.pkl")

# Text preprocessing
X = df['text']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Save vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "emotion_model.pkl")

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')

# Now correctly use stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers and extra spaces
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Show the results
# Add a readable label column
df['emotion_name'] = le.inverse_transform(df['label'])
print(df[['text', 'emotion_name']].head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize the clean_text column
vectorizer = TfidfVectorizer(max_features=5000)  # You can change max_features if needed
X = vectorizer.fit_transform(df['clean_text'])

# Store the labels
y = df['label']

print("TF-IDF matrix shape:", X.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save model and vectorizer
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

from sklearn.preprocessing import LabelEncoder

# After loading or preparing your labels
le = LabelEncoder()
le.fit(y)  # 'y' is your label column

import joblib
joblib.dump(le, "label_encoder.pkl")

le = joblib.load("label_encoder.pkl")


# Load model later (in a different file or session)
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Predict emotion for custom input
text_input = ["I am really scared of what might happen"]
text_vector = vectorizer.transform(text_input)
prediction = model.predict(text_vector)
print("Predicted Emotion:", le.inverse_transform(prediction)[0])

text_input = ["I’m feeling extremely nervous and scared."]
text_vector = vectorizer.transform(text_input)
prediction = model.predict(text_vector)
emotion = le.inverse_transform(prediction)[0]
print("Predicted Emotion:", emotion)

