import json
import os

from flask import Flask, render_template, request

from phishing.predictor import predict_email

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")

PLOTS = [
    ("model_comparison.png", "Model comparison on the test set"),
    ("cv_vs_test.png", "Cross-validation vs test F1"),
    ("confusion_matrices.png", "Confusion matrix for every model"),
    ("roc_curve.png", "ROC curve of the chosen model"),
    ("precision_recall_curve.png", "Precision-recall curve of the chosen model"),
]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    subject = ""
    email_text = ""

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        email_text = request.form.get("email_text", "").strip()
        if not email_text:
            error = "Please enter the email body."
        else:
            # The model reads subject and body together as one piece of text.
            prediction = predict_email(f"{subject} {email_text}".strip())

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        subject=subject,
        email_text=email_text,
    )


@app.route("/dashboard")
def dashboard():
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    plots = [
        {"file": f, "title": t}
        for f, t in PLOTS
        if os.path.exists(os.path.join(PLOT_DIR, f))
    ]
    return render_template("dashboard.html", metrics=metrics, plots=plots)


if __name__ == "__main__":
    app.run(debug=True)
