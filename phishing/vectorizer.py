from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 5000


def build_vectorizer():
    """TF-IDF settings in one place. The vectorizer is placed INSIDE the model
    pipeline, so during cross-validation it is re-fitted on each training fold
    only (no information leaks from the validation fold)."""
    return TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
