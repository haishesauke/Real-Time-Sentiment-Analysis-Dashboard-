import os, joblib, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import normalize_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.h5')
MAX_LEN = 40

class SentimentModel:
    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
        self.model = load_model(model_path) if os.path.exists(model_path) else None
        self.tokenizer = joblib.load(tokenizer_path) if os.path.exists(tokenizer_path) else None

    def predict(self, texts):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not found. Train first.")
        if isinstance(texts, str):
            texts = [texts]
        clean = [normalize_text(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(clean)
        X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
        probs = self.model.predict(X).ravel()
        labels = (probs > 0.5).astype(int).tolist()
        return [{'text': t, 'prob_pos': float(p), 'label': int(l)} for t, p, l in zip(texts, probs, labels)]
