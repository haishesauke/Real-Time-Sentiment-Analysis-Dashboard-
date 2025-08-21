# Real-Time Sentiment Analysis Dashboard

A production-style demo that streams tweets, scores sentiment with an LSTM, and visualizes trends live with Dash/Plotly.

**Highlights**
- Tweepy (Twitter API v2) streaming (with mock mode for offline demo)
- Text cleaning & tokenization pipeline
- LSTM (TensorFlow/Keras) model training, ~88% F1 on sample (target; depends on data)
- Dash + Plotly live dashboard
- Heroku-ready (Procfile + runtime.txt) and Dockerfile
- Nightly retraining & drift monitoring with GitHub Actions

## Quickstart

### 0) Environment
- Python 3.10 recommended.
- Create a virtualenv and install requirements:
  ```bash
  pip install -r requirements.txt
  ```

### 1) (Optional) Create .env for Twitter API
Copy `.env.example` to `.env` and fill your Twitter credentials:
```env
TWITTER_BEARER_TOKEN=...
TWITTER_SEARCH_QUERY="product launch (from:YourBrand) OR YourProductName"
```

### 2) Generate sample data
```bash
python src/data/make_sample_data.py
```

### 3) Train model (saves to `models/model.h5` + `models/tokenizer.pkl` + `models/metrics.json`)
```bash
python src/train_lstm.py
```

### 4) Run streamer (mock mode by default; uses sample tweets)
```bash
python src/streamer.py --mode mock
```

### 5) Run dashboard
```bash
python -m src.dashboard.app
# Then open http://127.0.0.1:8050
```

## Deploy

### Heroku
1. Ensure `runtime.txt` and `Procfile` are present (they are).
2. Login and create app:
   ```bash
   heroku create your-app-name
   heroku buildpacks:add heroku/python
   git push heroku main
   ```
3. Set config vars for Twitter if using live streaming.

### Docker
```bash
docker build -t rt-sentiment .
docker run -p 8050:8050 --env-file .env rt-sentiment
```

## Nightly Retraining & Drift Monitoring
- `.github/workflows/retrain.yml` runs nightly on a cron.
- It re-trains the model, computes F1, compares to `models/baseline_metrics.json`.
- If F1 drops more than 5% relative, it fails the job and (optionally) opens an issue.

## Project Structure
```
rt-sentiment-dashboard/
├─ README.md
├─ requirements.txt
├─ Procfile
├─ runtime.txt
├─ docker-compose.yml
├─ Dockerfile
├─ .env.example
├─ .github/
│  └─ workflows/
│     └─ retrain.yml
├─ models/
│  └─ baseline_metrics.json
├─ src/
│  ├─ preprocessing.py
│  ├─ train_lstm.py
│  ├─ inference.py
│  ├─ streamer.py
│  └─ dashboard/
│     └─ app.py
├─ src/data/
│  ├─ make_sample_data.py
│  ├─ sample_tweets.csv
│  └─ stream.jsonl           # created at runtime by streamer
└─ tests/
   ├─ test_preprocessing.py
   └─ test_inference.py
```
