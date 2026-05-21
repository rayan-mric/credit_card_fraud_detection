import os
import subprocess


def ensure_directories():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)


def run_script(script_path):
    """Run a python file as subprocess"""
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


def main():
    print("\n==============================")
    print("💳 FRAUD DETECTION FULL PIPELINE")
    print("==============================\n")

    ensure_directories()

    # ---------------------------
    # STEP 1: PREPROCESS DATA
    # ---------------------------
    print("🧹 Running preprocessing...")
    run_script("src/data_preprocessing.py")

    # ---------------------------
    # STEP 2: MODEL TRAINING (SMOTE + ML models)
    # ---------------------------
    print("🤖 Running model training...")
    run_script("src/model_training.py")

    # ---------------------------
    # STEP 3: ANOMALY DETECTION
    # ---------------------------
    print("⚠️ Running anomaly detection...")
    run_script("src/anomaly_detection.py")

    # ---------------------------
    # STEP 4: EVALUATION
    # ---------------------------
    print("📊 Running evaluation...")
    run_script("src/evaluation.py")

    # ---------------------------
    # STEP 5: VISUALIZATION
    # ---------------------------
    print("📈 Creating visualizations...")
    run_script("src/visualization.py")

    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()