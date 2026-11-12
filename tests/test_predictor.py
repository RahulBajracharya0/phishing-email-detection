import os

import pytest

from phishing.predictor import MODEL_PATH, predict_email

SAFE = (
    "Hi team, attached is the agenda for tomorrow's project meeting at 10am. "
    "Please review it beforehand. Thanks, Sarah"
)
PHISHING = (
    "URGENT: Your account has been suspended. Click http://bit.ly/x9 now to "
    "verify your password and bank details immediately or lose access."
)


def _need_model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("run phishing_detector.py first")


def test_result_has_expected_fields():
    _need_model()
    result = predict_email(SAFE)
    assert set(result) == {"label", "is_phishing", "confidence", "phishing_probability"}


def test_obvious_phishing_email_is_flagged():
    _need_model()
    result = predict_email(PHISHING)
    assert result["is_phishing"] is True
    assert result["label"] == "Phishing Email"


def test_normal_work_email_is_safe():
    _need_model()
    result = predict_email(SAFE)
    assert result["is_phishing"] is False
    assert result["label"] == "Safe Email"


def test_percentages_are_between_0_and_100():
    _need_model()
    for text in (SAFE, PHISHING):
        result = predict_email(text)
        assert 0 <= result["confidence"] <= 100
        assert 0 <= result["phishing_probability"] <= 100


def test_confidence_is_at_least_50_percent():
    _need_model()
    assert predict_email(PHISHING)["confidence"] >= 50
    assert predict_email(SAFE)["confidence"] >= 50


def test_same_input_gives_same_output():
    _need_model()
    assert predict_email(PHISHING) == predict_email(PHISHING)


def test_empty_text_does_not_crash():
    _need_model()
    assert "label" in predict_email("")


def test_symbols_only_does_not_crash():
    _need_model()
    assert "label" in predict_email("!!! ??? 12345 @@@")


def test_very_long_input_does_not_crash():
    _need_model()
    assert "label" in predict_email("free money click here " * 200_000)
