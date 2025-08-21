import re
import unicodedata

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')
NON_ALPHANUM = re.compile(r'[^a-z0-9 ]+')

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = NON_ALPHANUM.sub(" ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
