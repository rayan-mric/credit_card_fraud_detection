import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(path):
    """Load the cleaned dataset."""
    return pd.read_csv(path)


def prepare_data(df):
    """Separate features and target and split the dataset."""
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


def create_models():
    """Create the supervised fraud detection models."""

    models = {
        "Logistic Regression": ImbPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),

        "Random Forest": ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        ),

        "Gradient Boosting": ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=RANDOM_STATE
                    )
                )
            ]
        )
    }

    return models


def train_models(X_train, y_train):
    """Train all supervised models."""
    models = create_models()

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} training completed.")

    return trained_models


def save_models(models):
    """Save trained models to the models directory."""
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join("models", filename)

        joblib.dump(model, path)

        print(f"Saved: {path}")


def main():
    print("\n==============================")
    print("MODEL TRAINING")
    print("==============================")

    df = load_data("data/processed/cleaned.csv")

    X_train, X_test, y_train, y_test = prepare_data(df)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training fraud cases: {y_train.sum()}")
    print(f"Testing fraud cases: {y_test.sum()}")

    models = train_models(X_train, y_train)

    save_models(models)

    return models, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()