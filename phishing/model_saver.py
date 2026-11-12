import json
import os

import joblib

MODEL_DIR = "models"


def save_best_model(summary):
    os.makedirs(MODEL_DIR, exist_ok=True)

    best = summary["best"]
    print(f"\nSaving best model: {best['name']}")

    # One file containing TF-IDF + classifier, so they can never get out of sync.
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best["model"], model_path)

    info = {
        "model_name": best["name"],
        "positive_label": summary["positive_label"],
        "threshold": summary["threshold"],
        "best_params": best["best_params"],
    }
    with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Everything the report / dashboard needs, in one place.
    metrics = {
        "positive_label": summary["positive_label"],
        "n_train": summary["n_train"],
        "n_test": summary["n_test"],
        "cv_folds": summary["cv_folds"],
        "chosen_model": best["name"],
        "threshold": summary["threshold"],
        "models": [
            {
                "name": m["name"],
                "best_params": m["best_params"],
                "cv_f1": m["cv_f1"],
                "cv_f1_std": m["cv_f1_std"],
                "test": m["test"],
            }
            for m in summary["models"]
        ],
        "chosen_test_default_threshold": summary["best_test_default_threshold"],
        "chosen_test_tuned_threshold": summary["best_test_tuned_threshold"],
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to:   {model_path}")
    print(f"Saved settings:   {os.path.join(MODEL_DIR, 'model_info.json')}")
    print(f"Saved metrics:    {os.path.join(MODEL_DIR, 'metrics.json')}")
