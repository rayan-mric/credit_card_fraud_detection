import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_data(path=RAW_DATA_PATH):
    """Load the credit card fraud dataset."""

    df = pd.read_csv(path)

    return df


def preprocess(df):
    """
    Basic data cleaning.

    Important:
    Scaling and SMOTE are NOT performed here.
    They are handled inside the ML pipeline after splitting
    the dataset to prevent data leakage.
    """

    df = df.copy()

    print(f"Original transactions: {len(df):,}")

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    print(f"Duplicate transactions removed: {removed:,}")
    print(f"Remaining transactions: {len(df):,}")

    return df


def split_data(df):
    """
    Split the dataset into training and testing sets.

    Stratification preserves the fraud/non-fraud ratio.
    """

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def save_cleaned_data(df, path):
    """Save cleaned dataset."""

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Cleaned dataset saved to: {path}")


if __name__ == "__main__":

    df = load_data()

    df = preprocess(df)

    save_cleaned_data(
        df,
        "data/processed/cleaned.csv"
    )

    X_train, X_test, y_train, y_test = split_data(df)

    print("\nData split complete.")
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    print(f"Training fraud cases: {y_train.sum():,}")
    print(f"Testing fraud cases: {y_test.sum():,}")