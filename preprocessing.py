import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download if not already present
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Custom sentiment/negation patterns
negation_phrases = {
    "not happy": "not_happy",
    "not good": "not_good",
    "no support": "no_support",
    "not working": "not_working",
    "not satisfied": "not_satisfied",
    "not like": "not_like",
    "did not enjoy": "did_not_enjoy",
    "wasn't good": "wasnt_good",
    "couldn't stand": "couldnt_stand",
    "don't like": "dont_like",
    "not worth": "not_worth",
    "not at all happy": "not_happy",
    "not at all satisfied": "not_satisfied"
}

negation_words = {"not", "no", "never", "n't"}

def handle_negations(text):
    words = word_tokenize(text)
    new_words = []
    skip_next = False

    for i in range(len(words)):
        if skip_next:
            skip_next = False
            continue

        word = words[i]
        if word in negation_words and i + 1 < len(words):
            new_words.append(f"{word}_{words[i+1]}")
            skip_next = True
        else:
            new_words.append(word)

    return " ".join(new_words)

def apply_custom_rules(text):
    text = text.lower()  # <--- add this
    for phrase, replacement in negation_phrases.items():
        text = re.sub(rf'\b{phrase}\b', replacement, text)
    return text

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Replace custom phrases
    text = apply_custom_rules(text)

    # Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    # Handle negations
    text = handle_negations(text)

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
