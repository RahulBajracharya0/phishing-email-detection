import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(name, model, X_test, y_test, positive_label, threshold=0.5):
    """Evaluate a fitted pipeline on held-out data.

    A message is called phishing when P(phishing) >= threshold.
    Confusion matrix layout (rows = actual, columns = predicted):
        [[TP, FN],
         [FP, TN]]   with 'phishing' as the positive class.
    """
    classes = list(model.classes_)
    negative_label = next(c for c in classes if c != positive_label)

    p_phish = model.predict_proba(X_test)[:, classes.index(positive_label)]
    y_pred = np.where(p_phish >= threshold, positive_label, negative_label)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    auc = roc_auc_score((np.asarray(y_test) == positive_label).astype(int), p_phish)
    cm = confusion_matrix(y_test, y_pred, labels=[positive_label, negative_label])

    print(f"\n--- {name} (threshold = {threshold:.2f}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}   (of emails flagged as phishing, how many really were)")
    print(f"Recall:    {rec:.4f}   (of real phishing emails, how many were caught)")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Confusion matrix [[TP, FN], [FP, TN]]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "name": name,
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }
