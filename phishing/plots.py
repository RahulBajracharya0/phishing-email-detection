"""Creates the performance graphs used in the dashboard and the final report.

Inputs (all produced by phishing_detector.py):
    models/metrics.json   scores and confusion matrices for every model
    models/best_model.pkl the chosen model (used for the ROC / PR curves)
    data/test.csv         the held-out test set

Outputs are saved to static/plots/.
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")

COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_model_comparison(metrics):
    names = [m["name"] for m in metrics["models"]]
    scores = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(scores):
        vals = [m["test"][s] for m in metrics["models"]]
        ax.bar(x + (i - 1.5) * width, vals, width, label=s.capitalize(), color=COLOURS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    low = min(m["test"][s] for m in metrics["models"] for s in scores)
    ax.set_ylim(max(0.0, np.floor((low - 0.01) * 50) / 50), 1.0)
    ax.set_ylabel(f"Score (y-axis starts at {ax.get_ylim()[0]:.2f})")
    ax.set_title("Model comparison on the test set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4)
    _save(fig, "model_comparison.png")


def plot_cv_vs_test(metrics):
    names = [m["name"] for m in metrics["models"]]
    cv = [m["cv_f1"] for m in metrics["models"]]
    err = [m["cv_f1_std"] for m in metrics["models"]]
    test = [m["test"]["f1"] for m in metrics["models"]]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, cv, width, yerr=err, capsize=4, label="Cross-validation F1 (train)", color="#4C72B0")
    ax.bar(x + width / 2, test, width, label="Test F1", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    low = min(min(cv), min(test))
    ax.set_ylim(max(0.0, np.floor((low - 0.01) * 50) / 50), 1.0)
    ax.set_ylabel(f"F1-score (y-axis starts at {ax.get_ylim()[0]:.2f})")
    ax.set_title("Cross-validation vs test F1 (similar values = little overfitting)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    _save(fig, "cv_vs_test.png")


def _draw_cm(ax, cm, title):
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Phishing", "Safe"])
    ax.set_yticklabels(["Phishing", "Safe"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)


def plot_confusion_matrices(metrics):
    models = metrics["models"]
    cols = 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, m in zip(axes, models):
        _draw_cm(ax, m["test"]["confusion_matrix"], m["name"])
    for ax in axes[len(models):]:
        ax.axis("off")
    _save(fig, "confusion_matrices.png")

    chosen = metrics["chosen_test_tuned_threshold"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    _draw_cm(ax, chosen["confusion_matrix"],
             f"{metrics['chosen_model']}\n(threshold {chosen['threshold']:.2f})")
    _save(fig, "confusion_matrix_chosen.png")


def plot_roc_and_pr(metrics):
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TEST_PATH)):
        print("Skipping ROC / PR curves: model or test data not found.")
        return

    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])
    positive = metrics["positive_label"]
    p = model.predict_proba(test["text"])[:, list(model.classes_).index(positive)]
    y = (test["label"] == positive).astype(int).to_numpy()

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#4C72B0", label=f"AUC = {auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(f"ROC curve: {metrics['chosen_model']}")
    ax.legend(loc="lower right")
    _save(fig, "roc_curve.png")

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, color="#DD8452")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.8, 1.01)
    ax.set_title(f"Precision-recall curve: {metrics['chosen_model']}")
    _save(fig, "precision_recall_curve.png")


def generate_all():
    if not os.path.exists(METRICS_PATH):
        print("No models/metrics.json found. Run phishing_detector.py first.")
        return
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    plot_model_comparison(metrics)
    plot_cv_vs_test(metrics)
    plot_confusion_matrices(metrics)
    plot_roc_and_pr(metrics)


if __name__ == "__main__":
    generate_all()
