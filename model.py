from flask import Flask, render_template, request
import pickle
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the sentiment analysis model, TfidfVectorizer, and stopwords
with open('sentiment_model.pkl', 'rb') as model_file:
    sentiment_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open('stopwords.pkl', 'rb') as stopwords_file:
    stop_words = pickle.load(stopwords_file)

nlp = spacy.load('en_core_web_sm')
nlp.vocab['not'].is_stop = False
lemmatizer = WordNetLemmatizer()

# Define the clean_words function
def clean_words(text):
    text = text.lower()
    replace_list = {
        r"'m": ' am',
        r"'re": ' are',
        r"let’s": 'let us',
        r"'s": ' is',
        r"'ve": ' have',
        r"can't": 'can not',
        r"cannot": 'can not',
        r"shan’t": 'shall not',
        r"n't": ' not',
        r"'d": ' would',
        r"'ll": ' will',
        r"'scuse": 'excuse',
        ',': ' ,',
        '.': ' .',
        '!': ' !',
        '?': ' ?',
        '\s+': ' '
    }
    for s in replace_list:
        text = re.sub(s, replace_list[s], text)
    text = ' '.join(text.split())
    return text

def remove_stop_words(document):
    document = re.sub(r'\W', ' ', str(document))
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\s*\d\s+', '', document)
    document = word_tokenize(document)
    document = [lemmatizer.lemmatize(word, pos='v') for word in document if word not in stop_words and word not in punctuation]
    document = ' '.join(document)
    return document

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form["user_text"]

        # Preprocess user input
        cleaned_input_text = remove_stop_words(clean_words(user_text))

        # Transform user input using TfidfVectorizer
        input_text_vector = tfidf_vectorizer.transform([cleaned_input_text])

        # Predict sentiment using the loaded model
        predicted_rating = sentiment_model.predict(input_text_vector)[0]

        # Determine sentiment label
        sentiment_label = ""
        if predicted_rating == 1:
            sentiment_label = "bad"
        elif predicted_rating == 2:
            sentiment_label = "bad"
        elif predicted_rating == 3:
            sentiment_label = "neutral"
        elif predicted_rating == 4:
            sentiment_label = "good"
        elif predicted_rating == 5:
            sentiment_label = "good"

        return render_template("result.html", user_input=user_text, sentiment=sentiment_label)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
