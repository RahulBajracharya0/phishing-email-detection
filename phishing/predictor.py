import json
import os

import joblib

from phishing.text_cleaning import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.json")

# Loaded once, the first time it is needed, then reused for every request.
_model = None
_info = None


def load_model():
    global _model, _info
    if _model is None or _info is None:
        _model = joblib.load(MODEL_PATH)
        with open(INFO_PATH) as f:
            _info = json.load(f)
    return _model, _info


def predict_email(email_text):
    """Return {'label', 'confidence' (percent), 'phishing_probability' (percent)}."""
    model, info = load_model()
    positive = info["positive_label"]
    classes = list(model.classes_)
    negative = next(c for c in classes if c != positive)

    cleaned = clean_text(email_text)  # same cleaning as the training data
    p_phish = float(model.predict_proba([cleaned])[0][classes.index(positive)])

    is_phishing = p_phish >= info["threshold"]
    confidence = p_phish if is_phishing else 1 - p_phish

    return {
        "label": positive if is_phishing else negative,
        "is_phishing": bool(is_phishing),
        "confidence": round(confidence * 100, 1),
        "phishing_probability": round(p_phish * 100, 1),
    }


def main():
    email_text = input("Enter email text: ")
    result = predict_email(email_text)
    print(f"\nPrediction: {result['label']} ({result['confidence']}% confidence)")


if __name__ == "__main__":
    main()
