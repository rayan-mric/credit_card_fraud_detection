import os
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.model_training import (
    load_data,
    prepare_data,
    train_models
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a single fraud detection model."""

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

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion Matrix": matrix,
        "Predictions": predictions,
        "Probabilities": probabilities
    }


def save_results(results):
    """Save model comparison results to CSV."""

    os.makedirs("outputs/metrics", exist_ok=True)

    rows = []

    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "ROC-AUC": metrics["ROC-AUC"],
            "PR-AUC": metrics["PR-AUC"]
        })

    results_df = pd.DataFrame(rows)

    output_path = "outputs/metrics/model_comparison.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nModel comparison saved to: {output_path}")

    return results_df


def print_results(results):
    """Print detailed evaluation results."""

    for model_name, metrics in results.items():

        print("\n========================================")
        print(model_name)
        print("========================================")

        print(f"Precision : {metrics['Precision']:.4f}")
        print(f"Recall    : {metrics['Recall']:.4f}")
        print(f"F1 Score  : {metrics['F1 Score']:.4f}")
        print(f"ROC-AUC   : {metrics['ROC-AUC']:.4f}")
        print(f"PR-AUC    : {metrics['PR-AUC']:.4f}")

        print("\nConfusion Matrix:")
        print(metrics["Confusion Matrix"])


def main():
    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    df = load_data("data/processed/cleaned.csv")

    X_train, X_test, y_train, y_test = prepare_data(df)

    models = train_models(
        X_train,
        y_train
    )

    results = {}

    for model_name, model in models.items():

        print(f"\nEvaluating {model_name}...")

        results[model_name] = evaluate_model(
            model,
            X_test,
            y_test
        )

    print_results(results)

    results_df = save_results(results)

    return results, results_df


if __name__ == "__main__":
    main()