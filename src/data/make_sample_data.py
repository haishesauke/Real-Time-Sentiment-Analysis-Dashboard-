import csv, random, os
from pathlib import Path

OUT = Path(__file__).parent / 'sample_tweets.csv'

POS = [
    "Absolutely love this product! Game changer.",
    "The launch went flawlessly, super excited!",
    "Great announcement today, very impressed.",
    "This new feature is fantastic and easy to use.",
    "Incredible performance improvements, well done team!"
]
NEG = [
    "Pretty disappointed with the launch, lots of bugs.",
    "Hate the new update, ruined my workflow.",
    "Terrible experience, would not recommend.",
    "The announcement fell flat and lacked details.",
    "Performance is worse now, very unhappy."
]

NEUTRAL_NOISE = [
    "Watching the keynote now.",
    "Trying the product later today.",
    "Anyone else following the launch?"
]

def synthesize(n=1000, pos_ratio=0.5):
    rows = []
    for i in range(n):
        if random.random() < pos_ratio:
            text = random.choice(POS + NEUTRAL_NOISE)
            label = 1 if text in POS else 1  # bias neutral to positive in launch context
        else:
            text = random.choice(NEG + NEUTRAL_NOISE)
            label = 0 if text in NEG else 0  # bias neutral to negative for balance
        rows.append((text, label))
    return rows

if __name__ == '__main__':
    rows = synthesize(n=1200, pos_ratio=0.5)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['text','label'])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {OUT}")
