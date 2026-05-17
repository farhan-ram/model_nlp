import pandas as pd
import re
import joblib

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from transformers import pipeline
from youtube_comment_downloader import YoutubeCommentDownloader
from urllib.parse import urlparse, parse_qs
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

# LOAD CSV

dfn = pd.read_csv("normalisasi.csv")
normalisasi = dict(zip(dfn['slang'], dfn['formal']))

dfs = pd.read_csv("stopwords.csv")
stopwords = set(dfs['stopword'])


# STEMMER

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# LOAD MODEL

model_lr = joblib.load("model_lr.pkl")
tfidf = joblib.load("tfidf.pkl")


# INDOBERT

classifier = pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier"
)

from urllib.parse import urlparse, parse_qs

def normalize_youtube_url(url):

    # URL pendek
    if "youtu.be" in url:

        video_id = url.split("/")[-1].split("?")[0]

        return f"https://www.youtube.com/watch?v={video_id}"

    # URL normal
    if "youtube.com" in url:

        parsed = urlparse(url)

        query = parse_qs(parsed.query)

        if "v" in query:
            return f"https://www.youtube.com/watch?v={query['v'][0]}"

    return url

# CLEANING

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# PREPROCESS

def preprocess_nb(text):

    tokens = word_tokenize(text)

    tokens = [normalisasi.get(word, word) for word in tokens]

    tokens = [word for word in tokens if word not in stopwords]

    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

def hybrid_predict(text):

    clean = clean_text(text)

    processed = preprocess_nb(clean)

    X_input = tfidf.transform([processed])

   
    # LOGISTIC REGRESSION

    lr_pred = model_lr.predict(X_input)[0]

    probs = model_lr.predict_proba(X_input)[0]

    class_idx = list(model_lr.classes_).index(lr_pred)

    lr_prob = probs[class_idx]

    print("LR PRED:", lr_pred)
    print("LR CONF:", lr_prob)

    
    # RULE HYBRID
    
    use_bert = False

    # confidence tidak cukup kuat
    if lr_prob < 0.90:
        use_bert = True

    # INDOBERT

    if use_bert:

        print("PAKAI INDOBERT")

        bert = classifier(
            clean,
            truncation=True,
            max_length=512
        )[0]

        print("HASIL BERT:", bert)

        label = bert['label'].lower()

        label_map = {
            'positive': 'positif',
            'negative': 'negatif',
            'neutral': 'netral',

            'label_0': 'negatif',
            'label_1': 'netral',
            'label_2': 'positif'
        }

        return label_map.get(label, label)

    # PAKAI LR

    print("PAKAI LOGISTIC REGRESSION")

    return lr_pred

# FASTAPI

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class YoutubeRequest(BaseModel):
    url: str

@app.post("/predict")
def predict(data: TextRequest):

    result = hybrid_predict(data.text)

    return {
        "text": data.text,
        "sentiment": result
    }

@app.post("/analyze-youtube")
def analyze_youtube(data: YoutubeRequest):

    try:

        downloader = YoutubeCommentDownloader()

        # NORMALISASI URL
        video_url = normalize_youtube_url(data.url)

        # PAKAI URL YANG SUDAH DINORMALISASI
        comments_generator = downloader.get_comments_from_url(video_url)

        comments = list(comments_generator)

        if not comments:
            return {
                "message": "Komentar tidak ditemukan",
                "total_comments": 0,
                "results": []
            }

        results = []

        positif = 0
        negatif = 0
        netral = 0

        for item in comments:

            text = item.get("text", "")

            if text.strip() == "":
                continue

            sentiment = hybrid_predict(text)

            if sentiment == "positif":
                positif += 1

            elif sentiment == "negatif":
                negatif += 1

            else:
                netral += 1

            results.append({
                "comment": text,
                "sentiment": sentiment
            })

        return {

            "total_comments": len(results),

            "summary": {
                "positif": positif,
                "negatif": negatif,
                "netral": netral
            },

            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )