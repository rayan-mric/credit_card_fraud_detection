import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(path):
    """Load the credit card transaction dataset."""
    df = pd.read_csv(path)
    return df


def preprocess(df):
    """Clean the dataset and remove duplicate transactions."""
    df = df.copy()

    # Remove duplicate transactions
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    print(f"Removed {removed} duplicate transactions.")
    print(f"Remaining transactions: {len(df)}")

    return df


def split_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data("data/raw/creditcard.csv")
    df = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("\nData preprocessing complete.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training fraud cases: {y_train.sum()}")
    print(f"Testing fraud cases: {y_test.sum()}")