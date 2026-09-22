from src.config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    METRICS_DIR,
    MODEL_COMPARISON_PATH
)

from src.data_preprocessing import (
    load_data,
    prepare_data
)

from src.model_training import train_models

from src.evaluation import evaluate_models


def main():

    print("=" * 60)
    print("Credit Card Fraud Detection")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    df = load_data(RAW_DATA_PATH)

    print(f"Dataset loaded: {df.shape[0]:,} transactions")
    print(f"Features: {df.shape[1] - 1}")

    # --------------------------------------------------
    # 2. Prepare data
    # --------------------------------------------------

    print("\nPreparing data...")

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")

    # --------------------------------------------------
    # 3. Train models
    # --------------------------------------------------

    print("\nTraining models...")

    models = train_models(
        X_train,
        y_train,
        MODELS_DIR
    )

    # --------------------------------------------------
    # 4. Evaluate models
    # --------------------------------------------------

    print("\nEvaluating models...")

    results = evaluate_models(
        models,
        X_test,
        y_test
    )

    # --------------------------------------------------
    # 5. Save results
    # --------------------------------------------------

    results.to_csv(
        MODEL_COMPARISON_PATH,
        index=False
    )

    print("\nModel comparison:")
    print(results.to_string(index=False))

    print("\nResults saved to:")
    print(MODEL_COMPARISON_PATH)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()