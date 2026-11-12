import os

import pandas as pd
import pytest

import preprocess_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frame(rows):
    return pd.DataFrame(rows, columns=["Email Text", "Email Type"])


def test_missing_values_are_dropped():
    df = _frame([("hello there", "Safe Email"), (None, "Safe Email"), ("hi", None)])
    out = preprocess_data.clean_data(df, "Email Text", "Email Type")
    assert len(out) == 1


def test_duplicates_are_removed_after_cleaning():
    # Different raw text, identical once cleaned.
    df = _frame([("Free money!!!", "Phishing Email"), ("free money", "Phishing Email")])
    out = preprocess_data.clean_data(df, "Email Text", "Email Type")
    assert len(out) == 1


def test_texts_with_conflicting_labels_are_removed():
    df = _frame([
        ("same text", "Safe Email"),
        ("same text", "Phishing Email"),
        ("another one", "Safe Email"),
    ])
    out = preprocess_data.clean_data(df, "Email Text", "Email Type")
    assert list(out["clean_text"]) == ["another one"]


def test_rows_that_are_empty_after_cleaning_are_removed():
    df = _frame([("12345 !!!", "Safe Email"), ("real words", "Safe Email")])
    out = preprocess_data.clean_data(df, "Email Text", "Email Type")
    assert list(out["clean_text"]) == ["real words"]


def test_no_email_appears_in_both_train_and_test():
    train_path = os.path.join(ROOT, "data", "train.csv")
    test_path = os.path.join(ROOT, "data", "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        pytest.skip("run preprocess_data.py first")
    train = set(pd.read_csv(train_path)["text"].dropna())
    test = set(pd.read_csv(test_path)["text"].dropna())
    assert train.isdisjoint(test)


def test_train_and_test_have_both_classes():
    train_path = os.path.join(ROOT, "data", "train.csv")
    test_path = os.path.join(ROOT, "data", "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        pytest.skip("run preprocess_data.py first")
    assert pd.read_csv(train_path)["label"].nunique() == 2
    assert pd.read_csv(test_path)["label"].nunique() == 2
