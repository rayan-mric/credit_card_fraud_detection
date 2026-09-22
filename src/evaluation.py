import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models.

    Returns a DataFrame containing:
    Precision, Recall, F1 and ROC-AUC.
    """

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc
        })

    results_df = pd.DataFrame(results)

    return results_df