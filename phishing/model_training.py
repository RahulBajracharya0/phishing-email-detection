import shutil
import tempfile

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_recall_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from phishing.evaluator import evaluate_model
from phishing.vectorizer import build_vectorizer

RANDOM_STATE = 42
N_SPLITS = 5
N_JOBS = 2  # parallel workers; raise if you have plenty of RAM, lower to 1 if memory is tight


def find_positive_label(y):
    """Which label means 'phishing'? (used for precision / recall / F1)"""
    labels = y.unique().tolist()
    return next(
        (l for l in labels if "phishing" in str(l).lower() or "spam" in str(l).lower()),
        labels[0],
    )


def get_candidates():
    """(name, classifier, hyperparameter grid).

    Class imbalance: the data is roughly 62% safe / 38% phishing. We handle it with
    class_weight="balanced" (errors on the smaller class cost more) instead of
    SMOTE. SMOTE would invent synthetic points in a 5000-dimensional TF-IDF space
    that do not correspond to real emails. Naive Bayes has no class_weight, so
    fit_prior=False makes it use equal class priors instead.
    Regularisation strength (C / alpha) is tuned in the grids below.
    """
    return [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"clf__C": [0.1, 1, 10, 30]},
        ),
        (
            "Naive Bayes",
            MultinomialNB(fit_prior=False),
            {"clf__alpha": [0.001, 0.01, 0.1, 0.5, 1.0]},
        ),
        (
            "Support Vector Machine",
            # LinearSVC has no probabilities, so it is wrapped in a calibrator.
            CalibratedClassifierCV(
                LinearSVC(class_weight="balanced"), cv=3
            ),
            {"clf__estimator__C": [0.1, 0.5, 1, 2]},
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                class_weight="balanced",
                min_samples_leaf=2,  # smaller trees: less memory, less overfitting
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {"clf__n_estimators": [100, 200]},
        ),
    ]


def choose_threshold(pipeline, X_train, y_train, positive_label, cv):
    """Pick the probability cut-off that maximises F1, using out-of-fold
    predictions on the TRAINING data only (the test set is not touched)."""
    proba = cross_val_predict(
        clone(pipeline), X_train, y_train, cv=cv, method="predict_proba", n_jobs=N_JOBS
    )
    classes = sorted(y_train.unique().tolist())  # same order sklearn uses
    p_phish = proba[:, classes.index(positive_label)]
    y_true = (y_train.to_numpy() == positive_label).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_true, p_phish)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def train_and_compare(X_train, y_train, X_test, y_test):
    positive_label = find_positive_label(y_train)
    print(f"\nUsing '{positive_label}' as the positive (phishing) class.")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(f1_score, pos_label=positive_label)

    cache_dir = tempfile.mkdtemp()  # caches TF-IDF per fold so tuning is faster
    try:
        # ---------- 1. Hyperparameter tuning with stratified k-fold CV ----------
        tuned = []
        for name, clf, grid in get_candidates():
            print(f"\nTuning {name} ({N_SPLITS}-fold stratified CV)...")
            pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", clf)], memory=cache_dir)
            search = GridSearchCV(pipe, grid, scoring=scorer, cv=cv, n_jobs=N_JOBS, refit=True)
            search.fit(X_train, y_train)

            std = search.cv_results_["std_test_score"][search.best_index_]
            params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            print(f"  best params: {params}")
            print(f"  CV F1: {search.best_score_:.4f} (+/- {std:.4f})")
            tuned.append(
                {
                    "name": name,
                    "model": search.best_estimator_,
                    "best_params": params,
                    "cv_f1": float(search.best_score_),
                    "cv_f1_std": float(std),
                }
            )

        # ---------- 2. Ensemble (soft voting over the tuned models) ----------
        print("\nEvaluating soft-voting ensemble (CV)...")
        voting = VotingClassifier(
            [(t["name"], clone(t["model"].named_steps["clf"])) for t in tuned],
            voting="soft",
        )
        ensemble = Pipeline([("tfidf", build_vectorizer()), ("clf", voting)], memory=cache_dir)
        scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring=scorer, n_jobs=N_JOBS)
        print(f"  CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
        ensemble.fit(X_train, y_train)
        tuned.append(
            {
                "name": "Ensemble (soft voting)",
                "model": ensemble,
                "best_params": {"members": [t["name"] for t in tuned]},
                "cv_f1": float(scores.mean()),
                "cv_f1_std": float(scores.std()),
            }
        )

        # ---------- 3. Choose the winner using CV only (never the test set) ----------
        best = max(tuned, key=lambda t: t["cv_f1"])
        print(f"\nBest model by cross-validation: {best['name']} (CV F1 = {best['cv_f1']:.4f})")

        # ---------- 4. Threshold tuning (training data only) ----------
        threshold = choose_threshold(best["model"], X_train, y_train, positive_label, cv)
        print(f"Tuned decision threshold: {threshold:.3f}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        for t in tuned:
            t["model"].set_params(memory=None)

    # ---------- 5. Final evaluation on the held-out test set ----------
    print("\n================ TEST SET RESULTS ================")
    for t in tuned:
        t["test"] = evaluate_model(t["name"], t["model"], X_test, y_test, positive_label)

    best_default = best["test"]
    best_tuned = evaluate_model(
        best["name"] + " [tuned threshold]", best["model"], X_test, y_test,
        positive_label, threshold=threshold,
    )

    return {
        "positive_label": positive_label,
        "models": tuned,
        "best": best,
        "threshold": threshold,
        "best_test_default_threshold": best_default,
        "best_test_tuned_threshold": best_tuned,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_folds": N_SPLITS,
    }
