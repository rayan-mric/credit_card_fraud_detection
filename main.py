import os

from src.data_preprocessing import (
    load_data,
    preprocess,
    split_data
)

from src.model_training import (
    create_models,
    train_models,
    save_models
)

from src.evaluation import (
    evaluate_model,
    save_results,
    print_results
)


DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_PATH = "data/processed/cleaned.csv"


def ensure_directories():
    """Create required project directories."""

    directories = [
        "data/processed",
        "outputs/charts",
        "outputs/metrics",
        "outputs/reports",
        "models"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def main():

    print("\n==========================================")
    print("     CREDIT CARD FRAUD DETECTION")
    print("==========================================\n")

    ensure_directories()

    # ======================================
    # STEP 1: LOAD DATA
    # ======================================

    print("STEP 1: Loading dataset...")

    df = load_data(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # ======================================
    # STEP 2: DATA PREPROCESSING
    # ======================================

    print("\nSTEP 2: Preprocessing data...")

    df = preprocess(df)

    df.to_csv(
        PROCESSED_PATH,
        index=False
    )

    print(f"Cleaned dataset saved to: {PROCESSED_PATH}")

    # ======================================
    # STEP 3: TRAIN / TEST SPLIT
    # ======================================

    print("\nSTEP 3: Splitting dataset...")

    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training fraud cases: {y_train.sum()}")
    print(f"Testing fraud cases: {y_test.sum()}")

    # ======================================
    # STEP 4: TRAIN MODELS
    # ======================================

    print("\nSTEP 4: Training models...")

    models = train_models(
        X_train,
        y_train
    )

    # ======================================
    # STEP 5: SAVE MODELS
    # ======================================

    print("\nSTEP 5: Saving models...")

    save_models(models)

    # ======================================
    # STEP 6: EVALUATE MODELS
    # ======================================

    print("\nSTEP 6: Evaluating models...")

    results = {}

    for model_name, model in models.items():

        results[model_name] = evaluate_model(
            model,
            X_test,
            y_test
        )

    # ======================================
    # STEP 7: DISPLAY RESULTS
    # ======================================

    print("\nSTEP 7: Model results")

    print_results(results)

    # ======================================
    # STEP 8: SAVE RESULTS
    # ======================================

    print("\nSTEP 8: Saving evaluation results...")

    results_df = save_results(results)

    print("\n==========================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("==========================================")

    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()