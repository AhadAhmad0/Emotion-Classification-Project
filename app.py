from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))

# -------------------------------
# LOAD DATA & TRAIN MODEL
# -------------------------------
def load_data():
    data = []

    for file in ["train.txt", "test.txt", "val.txt"]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                text, label = line.strip().split(";")
                data.append([text, label])

    df = pd.DataFrame(data, columns=["text", "label"])
    return df


def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return " ".join(text)


# Train once
df = load_data()
df["cleaned"] = df["text"].apply(clean_text)

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df["cleaned"])

lb = LabelEncoder()
y = lb.fit_transform(df["label"])

model = LogisticRegression()
model.fit(X, y)


# -------------------------------
# PREDICTION
# -------------------------------
def predict_emotion(text):
    cleaned = clean_text(text)
    vector = tfidf_vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    return lb.inverse_transform([pred])[0]


# -------------------------------
# ROUTES
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def analyze_emotion():
    if request.method == 'POST':
        comment = request.form.get('comment', '').strip()

        if not comment:
            return render_template('index.html', sentiment="Please enter text")

        try:
            result = predict_emotion(comment)
        except Exception as e:
            result = f"Error: {str(e)}"

        return render_template('index.html', sentiment=result)

    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)