import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def analyze_thresholds(
    probabilities,
    y_test
):
    """
    Evaluate model performance across
    different classification thresholds.
    """

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    results = []

    for threshold in thresholds:

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

        results.append(
            {
                "Threshold": round(
                    threshold,
                    2
                ),
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }
        )

    return pd.DataFrame(results)