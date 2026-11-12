from phishing.data_loader import load_split_data
from phishing.model_training import train_and_compare
from phishing.model_saver import save_best_model
from phishing.plots import generate_all


def main():
    train_df, test_df = load_split_data()
    summary = train_and_compare(
        train_df["text"], train_df["label"], test_df["text"], test_df["label"]
    )
    save_best_model(summary)
    generate_all()  # graphs for the dashboard and the report
    print("\nModel training and evaluation complete.")


if __name__ == "__main__":
    main()
