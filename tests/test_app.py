import os

import pytest

from app import app
from phishing.predictor import MODEL_PATH

PHISHING = "URGENT: verify your password at http://bit.ly/x9 or your account will be closed"


def _client():
    app.config["TESTING"] = True
    return app.test_client()


def _need_model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("run phishing_detector.py first")


def test_home_page_loads():
    response = _client().get("/")
    assert response.status_code == 200
    assert b"Phishing Email Detector" in response.data


def test_home_page_has_subject_and_body_fields():
    html = _client().get("/").get_data(as_text=True)
    assert 'name="subject"' in html
    assert 'name="email_text"' in html


def test_empty_body_shows_error():
    html = _client().post("/", data={"subject": "Hello", "email_text": "   "}).get_data(as_text=True)
    assert "Please enter the email body" in html
    assert "Prediction:" not in html


def test_missing_body_field_shows_error_not_crash():
    response = _client().post("/", data={})
    assert response.status_code == 200
    assert b"Please enter the email body" in response.data


def test_phishing_email_shows_phishing_result():
    _need_model()
    html = _client().post("/", data={"email_text": PHISHING}).get_data(as_text=True)
    assert "Prediction: Phishing Email" in html
    assert "Confidence:" in html


def test_safe_email_shows_safe_result():
    _need_model()
    html = _client().post(
        "/", data={"email_text": "Hi team, the meeting is at 10am tomorrow. Agenda attached."}
    ).get_data(as_text=True)
    assert "Prediction: Safe Email" in html


def test_entered_text_is_kept_after_submit():
    _need_model()
    html = _client().post(
        "/", data={"subject": "Account notice", "email_text": PHISHING}
    ).get_data(as_text=True)
    assert "Account notice" in html
    assert "verify your password" in html


def test_html_in_input_is_escaped():
    _need_model()
    html = _client().post(
        "/", data={"email_text": "<script>alert(1)</script> verify your password"}
    ).get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html


def test_dashboard_page_loads():
    response = _client().get("/dashboard")
    assert response.status_code == 200
    assert b"Model results" in response.data


def test_unknown_page_returns_404():
    assert _client().get("/does-not-exist").status_code == 404
