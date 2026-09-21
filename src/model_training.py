import joblib

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from src.config import (
    RANDOM_STATE,
    MODELS_DIR,
)


def create_models():
    """
    Create supervised fraud detection models.

    SMOTE is inside the pipeline so it is applied only
    to the training data.
    """

    models = {

        "Logistic Regression": ImbPipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    )
                ),

                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                ),
            ]
        ),

        "Random Forest": ImbPipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    )
                ),

                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        class_weight=None,
                    )
                ),
            ]
        ),

        "Gradient Boosting": ImbPipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    )
                ),

                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    )
                ),
            ]
        ),
    }

    return models


def train_models(X_train, y_train):
    """Train all models."""

    models = create_models()

    trained_models = {}

    for name, model in models.items():

        print("\n" + "=" * 50)
        print(f"Training {name}")
        print("=" * 50)

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        print(f"{name} training completed.")

    return trained_models


def save_models(models):
    """Save trained models."""

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_paths = {}

    for name, model in models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        path = MODELS_DIR / filename

        joblib.dump(
            model,
            path
        )

        saved_paths[name] = path

        print(f"Saved: {path}")

    return saved_paths