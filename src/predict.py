import pickle
from src.data_preprocessing import clean_text
import os

MODEL_PATH = 'models/fake_news_model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model/vectorizer not found. Run main.py to train first.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "REAL" if prediction == 1 else "FAKE"
