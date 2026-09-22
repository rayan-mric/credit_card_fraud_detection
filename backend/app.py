from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "creditcard.csv"
)


# ============================================================
# MODELS
# ============================================================

MODEL_PATHS = {

    "Logistic Regression":
        MODELS_DIR / "logistic_regression.pkl",

    "Random Forest":
        MODELS_DIR / "random_forest.pkl",

    "Gradient Boosting":
        MODELS_DIR / "gradient_boosting.pkl"
}


models = {}


for name, path in MODEL_PATHS.items():

    if path.exists():

        try:

            models[name] = joblib.load(path)

        except Exception as error:

            print(
                f"Could not load {name}: {error}"
            )


# ============================================================
# DATASET
# ============================================================

dataset = None


if DATA_PATH.exists():

    dataset = pd.read_csv(DATA_PATH)


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "ok",

        "models": list(models.keys()),

        "dataset_loaded": dataset is not None

    })


# ============================================================
# PROJECT SUMMARY
# ============================================================

@app.route("/api/summary", methods=["GET"])
def summary():

    if dataset is None:

        return jsonify({
            "error": "Dataset not found."
        }), 500


    total = len(dataset)

    fraud = int(dataset["Class"].sum())

    normal = total - fraud

    fraud_rate = (
        fraud / total * 100
    )


    return jsonify({

        "total_transactions": total,

        "fraud_cases": fraud,

        "normal_transactions": normal,

        "fraud_rate": round(
            fraud_rate,
            2
        )

    })


# ============================================================
# TRANSACTION
# ============================================================

@app.route(
    "/api/transaction/<int:index>",
    methods=["GET"]
)
def transaction(index):

    if dataset is None:

        return jsonify({
            "error": "Dataset not found."
        }), 500


    if index < 0 or index >= len(dataset):

        return jsonify({
            "error": "Transaction does not exist."
        }), 404


    row = dataset.iloc[index]


    amount = (
        float(row["Amount"])
        if "Amount" in dataset.columns
        else 0
    )


    time_value = (
        float(row["Time"])
        if "Time" in dataset.columns
        else 0
    )


    return jsonify({

        "index": index,

        "amount": round(
            amount,
            2
        ),

        "time": round(
            time_value,
            2
        ),

        "actual_class": int(
            row["Class"]
        )

    })


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        model_name = data.get("model")

        index = int(
            data.get("index")
        )


        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        if model_name not in models:

            return jsonify({
                "error":
                "Selected model is not available."
            }), 400


        # ----------------------------------------------------
        # Validate dataset
        # ----------------------------------------------------

        if dataset is None:

            return jsonify({
                "error":
                "Dataset not found."
            }), 500


        # ----------------------------------------------------
        # Validate transaction
        # ----------------------------------------------------

        if index < 0 or index >= len(dataset):

            return jsonify({
                "error":
                "Transaction does not exist."
            }), 400


        model = models[model_name]


        # ----------------------------------------------------
        # Get transaction
        # ----------------------------------------------------

        transaction = dataset.iloc[
            [index]
        ].copy()


        # ----------------------------------------------------
        # Get basic transaction information
        # ----------------------------------------------------

        amount = (
            float(transaction["Amount"].iloc[0])
            if "Amount" in transaction.columns
            else 0
        )

        time_value = (
            float(transaction["Time"].iloc[0])
            if "Time" in transaction.columns
            else 0
        )

        actual_class = int(
            transaction["Class"].iloc[0]
        )


        # ----------------------------------------------------
        # Separate target
        # ----------------------------------------------------

        X = transaction.drop(
            columns=["Class"]
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(X)[0]


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probability = (
                model.predict_proba(X)[0][1]
            )


        # ----------------------------------------------------
        # Result text
        # ----------------------------------------------------

        if prediction == 1:

            result = "Potential Fraud"

        else:

            result = "Normal Transaction"


        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return jsonify({

            "model": model_name,

            "index": index,

            "amount": round(
                amount,
                2
            ),

            "time": round(
                time_value,
                2
            ),

            "prediction": int(
                prediction
            ),

            "result": result,

            "probability": (
                round(
                    float(probability),
                    4
                )
                if probability is not None
                else None
            ),

            "actual_class": actual_class

        })


    except Exception as error:

        return jsonify({

            "error": str(error)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "Credit Card Fraud Detection API"
    )

    print(
        f"Models loaded: {list(models.keys())}"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )