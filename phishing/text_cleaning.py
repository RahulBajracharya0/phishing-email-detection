import re

# Emails longer than this are truncated before cleaning. A few rows in the
# dataset are millions of characters long, which slows down training and
# can hang the web app. The same limit is used for training and prediction
# so the model always sees text prepared in the same way.
MAX_CHARS = 10_000


def clean_text(text):
    """Clean raw email text. Used by BOTH preprocess_data.py and predictor.py."""
    if not isinstance(text, str):
        return ""

    text = text[:MAX_CHARS]
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)   # links -> URL token
    text = re.sub(r"\S+@\S+", " EMAIL ", text)           # addresses -> EMAIL token
    text = re.sub(r"<[^>]+>", " ", text)                 # strip HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)             # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text
