import os, time, json, argparse, random
from pathlib import Path
from dotenv import load_dotenv
from src.inference import SentimentModel
from src.preprocessing import normalize_text

load_dotenv()

DATA_DIR = Path(__file__).parent / 'data'
STREAM_PATH = DATA_DIR / 'stream.jsonl'
SAMPLE_PATH = DATA_DIR / 'sample_tweets.csv'

try:
    import tweepy
except Exception:
    tweepy = None

def write_event(obj):
    STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STREAM_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def mock_stream(interval=1.0):
    import csv
    with open(SAMPLE_PATH, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    random.shuffle(reader)
    model = SentimentModel()
    for row in reader:
        txt = row['text']
        pred = model.predict([txt])[0]
        event = {
            'text': txt,
            'ts': time.time(),
            'prob_pos': pred['prob_pos'],
            'label': pred['label'],
            'source': 'mock'
        }
        write_event(event)
        time.sleep(interval)

def twitter_stream(query, interval=0.0):
    if tweepy is None:
        raise RuntimeError("tweepy not installed")
    bearer = os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer:
        raise RuntimeError("TWITTER_BEARER_TOKEN missing in env")
    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
    model = SentimentModel()

    next_token = None
    while True:
        resp = client.search_recent_tweets(
            query=query,
            tweet_fields=['created_at','lang'],
            max_results=10,
            next_token=next_token
        )
        if resp is None or resp.data is None:
            time.sleep(5); continue
        for tw in resp.data:
            if tw.lang and tw.lang != 'en':
                continue
            txt = tw.text
            pred = model.predict([txt])[0]
            event = {
                'text': txt,
                'ts': time.time(),
                'prob_pos': pred['prob_pos'],
                'label': pred['label'],
                'source': 'twitter'
            }
            write_event(event)
            if interval > 0:
                time.sleep(interval)
        next_token = resp.meta.get('next_token')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['mock','twitter'], default='mock')
    ap.add_argument('--query', default=os.getenv('TWITTER_SEARCH_QUERY', 'product launch OR announcement'))
    ap.add_argument('--interval', type=float, default=1.0)
    args = ap.parse_args()
    if args.mode == 'mock':
        mock_stream(args.interval)
    else:
        twitter_stream(args.query, args.interval)
