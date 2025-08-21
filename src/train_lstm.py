import os, json, math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from src.preprocessing import normalize_text

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_tweets.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.h5')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')

MAX_WORDS = 20000
MAX_LEN = 40

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df['text'] = df['text'].astype(str).apply(normalize_text)
    X = df['text'].tolist()
    y = df['label'].astype(int).values
    return X, y

def vectorize(texts, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    return X, tokenizer

def build_model(vocab_size=MAX_WORDS, max_len=MAX_LEN):
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_len))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_texts, y = load_data()
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(X_texts, y, test_size=0.2, random_state=42, stratify=y)
    X_train, tokenizer = vectorize(X_train_texts, tokenizer=None)
    X_test, _ = vectorize(X_test_texts, tokenizer=tokenizer)

    model = build_model()
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1, epochs=6, batch_size=64, callbacks=[es], verbose=2)

    y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
    f1 = f1_score(y_test, y_pred)

    model.save(MODEL_PATH)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    with open(METRICS_PATH, 'w') as f:
        json.dump({'f1': float(f1), 'timestamp': pd.Timestamp.utcnow().isoformat()}, f, indent=2)
    print(f"Saved model to {MODEL_PATH}, tokenizer to {TOKENIZER_PATH}, F1={f1:.4f}")

if __name__ == '__main__':
    train()
