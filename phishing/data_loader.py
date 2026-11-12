import os
import pandas as pd

TRAIN_PATH = os.path.join("data", "train.csv")
TEST_PATH = os.path.join("data", "test.csv")


def load_split_data():
    print("Loading train/test data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = train_df.dropna(subset=["text", "label"])
    test_df = test_df.dropna(subset=["text", "label"])

    print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    return train_df, test_df