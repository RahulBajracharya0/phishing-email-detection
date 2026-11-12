import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from phishing.text_cleaning import clean_text

DATA_PATH = os.path.join("data", "emails.csv")
OUTPUT_DIR = "data"
TEXT_COLUMN = "Email Text"
LABEL_COLUMN = "Email Type"
TEST_SIZE = 0.2
RANDOM_STATE = 42

TEXT_CANDIDATES = ["text", "body", "email", "email_text", "message", "content"]
LABEL_CANDIDATES = ["label", "type", "class", "category", "is_phishing", "target"]


def detect_column(df, candidates, override):
    if override:
        return override
    for c in candidates:
        for col in df.columns:
            if col.strip().lower() == c:
                return col
    return None


def load_data(path):
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    return df


def clean_data(df, text_col, label_col):
    df = df.copy()

    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    print(f"\nDropped {before - len(df)} rows with missing text/label")

    df["clean_text"] = df[text_col].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.len() > 0]
    print(f"Dropped {before - len(df)} rows with empty cleaned text")

    # Remove duplicates AFTER cleaning. Two different raw emails can become
    # identical once cleaned, and if one lands in train and the other in test
    # the test score is inflated (data leakage).
    before = len(df)
    labels_per_text = df.groupby("clean_text")[label_col].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    df = df[~df["clean_text"].isin(conflicting)]
    print(f"Dropped {before - len(df)} rows whose text appears with conflicting labels")

    before = len(df)
    df = df.drop_duplicates(subset="clean_text")
    print(f"Dropped {before - len(df)} duplicate rows (after cleaning)")

    print(f"Rows remaining: {len(df)}")
    return df


def analyse_distribution(df, label_col, output_dir):
    counts = df[label_col].value_counts()
    print("\nClass distribution:")
    print(counts)
    print("\nClass distribution (%):")
    print((counts / counts.sum() * 100).round(2))

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title("Phishing vs Legitimate Email Distribution")
    plt.xlabel(label_col)
    plt.ylabel("Number of emails")
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(chart_path)
    print(f"\nSaved distribution chart to: {chart_path}")
    plt.close()


def split_and_save(df, label_col, output_dir):
    X = df["clean_text"]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    overlap = len(set(train_df["text"]) & set(test_df["text"]))
    print(f"\nTrain set: {len(train_df)} rows -> {train_path}")
    print(f"Test set:  {len(test_df)} rows -> {test_path}")
    print(f"Texts appearing in both train and test (should be 0): {overlap}")


def main():
    df = load_data(DATA_PATH)

    text_col = detect_column(df, TEXT_CANDIDATES, TEXT_COLUMN)
    label_col = detect_column(df, LABEL_CANDIDATES, LABEL_COLUMN)

    if text_col is None or label_col is None:
        print("\nCould not auto-detect text/label columns.")
        print("Please set TEXT_COLUMN and LABEL_COLUMN manually at the top of this script.")
        print(f"Available columns: {list(df.columns)}")
        return

    print(f"\nUsing text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")

    df = clean_data(df, text_col, label_col)
    analyse_distribution(df, label_col, OUTPUT_DIR)
    split_and_save(df, label_col, OUTPUT_DIR)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
