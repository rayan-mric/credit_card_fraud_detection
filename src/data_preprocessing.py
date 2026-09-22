import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load the credit card fraud dataset.
    """
    df = pd.read_csv(file_path)

    return df


def prepare_data(df):
    """
    Prepare the dataset for machine learning.

    Returns:
        X_train_scaled
        X_test_scaled
        y_train
        y_test
        scaler
    """

    # Remove rows with missing values
    df = df.dropna().copy()

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale the Amount column
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    if "Amount" in X_train.columns:
        X_train["Amount"] = scaler.fit_transform(
            X_train[["Amount"]]
        )

        X_test["Amount"] = scaler.transform(
            X_test[["Amount"]]
        )

    return X_train, X_test, y_train, y_test, scaler