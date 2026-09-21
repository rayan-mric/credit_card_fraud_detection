import joblib
import pandas as pd


def load_model(path):
    """Load a saved model."""

    return joblib.load(path)


def predict_transaction(
    model,
    transaction
):
    """
    Predict fraud probability for one transaction.

    transaction should contain the same feature
    columns used during model training.
    """

    if isinstance(transaction, dict):

        transaction = pd.DataFrame(
            [transaction]
        )

    probability = model.predict_proba(
        transaction
    )[:, 1][0]

    prediction = int(
        probability >= 0.50
    )

    if probability >= 0.75:
        risk_level = "High"

    elif probability >= 0.40:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return {
        "fraud_probability": float(
            probability
        ),
        "prediction": prediction,
        "risk_level": risk_level,
    }