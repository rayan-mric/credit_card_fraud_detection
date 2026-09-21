import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=0.50
):
    """
    Evaluate a trained model.

    The default classification threshold is 0.50.
    """

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion Matrix": matrix,
        "Predictions": predictions,
        "Probabilities": probabilities,
    }


def create_comparison_table(results):
    """Create a model comparison DataFrame."""

    rows = []

    for model_name, metrics in results.items():

        rows.append(
            {
                "Model": model_name,
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
                "ROC-AUC": metrics["ROC-AUC"],
                "PR-AUC": metrics["PR-AUC"],
            }
        )

    return pd.DataFrame(rows)


def save_results(results, path):
    """Save model comparison results."""

    comparison = create_comparison_table(
        results
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        path,
        index=False
    )

    print(
        f"\nModel comparison saved to: {path}"
    )

    return comparison


def print_results(results):
    """Print evaluation results."""

    for model_name, metrics in results.items():

        print("\n" + "=" * 50)
        print(model_name)
        print("=" * 50)

        print(
            f"Precision : {metrics['Precision']:.4f}"
        )

        print(
            f"Recall    : {metrics['Recall']:.4f}"
        )

        print(
            f"F1 Score  : {metrics['F1 Score']:.4f}"
        )

        print(
            f"ROC-AUC   : {metrics['ROC-AUC']:.4f}"
        )

        print(
            f"PR-AUC    : {metrics['PR-AUC']:.4f}"
        )

        print("\nConfusion Matrix:")

        print(
            metrics["Confusion Matrix"]
        )