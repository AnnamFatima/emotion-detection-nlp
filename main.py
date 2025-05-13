# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('emotion_dataset.csv')

# Drop rows with nulls
df.dropna(inplace=True)

# Ensure labels are strings
df['label'] = df['label'].astype(str)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return " ".join(tokens)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression with Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}
log_reg = LogisticRegression()
grid = GridSearchCV(log_reg, param_grid, cv=3)
grid.fit(X_train_tfidf, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross-Validated Accuracy:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_tfidf)
print("Test Accuracy After Tuning:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Logistic Regression comparison
log_reg2 = LogisticRegression(C=1, penalty='l2', solver='liblinear')
log_reg2.fit(X_train_tfidf, y_train)
y_pred_log = log_reg2.predict(X_test_tfidf)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)
print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# SVM
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Check overfitting or underfitting
training_accuracy = accuracy_score(y_train, best_model.predict(X_train_tfidf))
test_accuracy = accuracy_score(y_test, best_model.predict(X_test_tfidf))
print("\nTraining Accuracy:", training_accuracy)
print("Test Accuracy:", test_accuracy)

if abs(training_accuracy - test_accuracy) < 0.02:
    print("✅ The model generalizes well (no significant overfitting).")
elif training_accuracy > test_accuracy:
    print("⚠️ The model may be overfitting.")
else:
    print("🔍 The model may be underfitting.")

