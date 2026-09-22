import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def train_models(X_train, y_train, models_dir):
    """
    Train three classification models and save them.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    # Save models
    joblib.dump(
        trained_models["Logistic Regression"],
        models_dir / "logistic_regression.pkl"
    )

    joblib.dump(
        trained_models["Random Forest"],
        models_dir / "random_forest.pkl"
    )

    joblib.dump(
        trained_models["Gradient Boosting"],
        models_dir / "gradient_boosting.pkl"
    )

    print("All models saved successfully.")

    return trained_models