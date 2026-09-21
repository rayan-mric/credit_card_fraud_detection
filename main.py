from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    METRICS_DIR,
    RANDOM_STATE,
    create_directories,
)

from src.data_preprocessing import (
    load_data,
    preprocess,
    split_data,
    save_cleaned_data,
)

from src.model_training import (
    train_models,
    save_models,
)

from src.evaluation import (
    evaluate_model,
    print_results,
    save_results,
)


def main():

    print("\n")
    print("=" * 60)
    print("        CREDIT CARD FRAUD DETECTION")
    print("=" * 60)

    # --------------------------------------------------
    # STEP 1
    # --------------------------------------------------

    print("\nSTEP 1: Creating project directories...")

    create_directories()

    # --------------------------------------------------
    # STEP 2
    # --------------------------------------------------

    print("\nSTEP 2: Loading dataset...")

    df = load_data(
        RAW_DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Fraud transactions: "
        f"{df['Class'].sum():,}"
    )

    print(
        f"Normal transactions: "
        f"{(df['Class'] == 0).sum():,}"
    )

    # --------------------------------------------------
    # STEP 3
    # --------------------------------------------------

    print("\nSTEP 3: Cleaning dataset...")

    df = preprocess(df)

    save_cleaned_data(
        df,
        PROCESSED_DATA_PATH
    )

    # --------------------------------------------------
    # STEP 4
    # --------------------------------------------------

    print("\nSTEP 4: Splitting dataset...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(df)

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test):,}"
    )

    print(
        f"Training fraud cases: "
        f"{y_train.sum():,}"
    )

    print(
        f"Testing fraud cases: "
        f"{y_test.sum():,}"
    )

    # --------------------------------------------------
    # STEP 5
    # --------------------------------------------------

    print("\nSTEP 5: Training models...")

    models = train_models(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # STEP 6
    # --------------------------------------------------

    print("\nSTEP 6: Saving trained models...")

    save_models(
        models
    )

    # --------------------------------------------------
    # STEP 7
    # --------------------------------------------------

    print("\nSTEP 7: Evaluating models...")

    results = {}

    for model_name, model in models.items():

        print(
            f"\nEvaluating {model_name}..."
        )

        results[model_name] = evaluate_model(
            model,
            X_test,
            y_test,
        )

    # --------------------------------------------------
    # STEP 8
    # --------------------------------------------------

    print("\nSTEP 8: Model results...")

    print_results(
        results
    )

    # --------------------------------------------------
    # STEP 9
    # --------------------------------------------------

    print("\nSTEP 9: Saving model comparison...")

    comparison = save_results(
        results,
        METRICS_DIR / "model_comparison.csv"
    )

    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nModel comparison:")
    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()