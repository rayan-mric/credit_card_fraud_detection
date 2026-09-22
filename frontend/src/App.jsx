import React, { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:5000/api";


function App() {

  const [summary, setSummary] = useState(null);

  const [model, setModel] = useState(
    "Random Forest"
  );

  const [transactionIndex, setTransactionIndex] =
    useState(0);

  const [transaction, setTransaction] =
    useState(null);

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");


  // ----------------------------------------------------------
  // LOAD SUMMARY
  // ----------------------------------------------------------

  useEffect(() => {

    fetch(`${API_URL}/summary`)

      .then((response) => {

        if (!response.ok) {
          throw new Error(
            "Could not connect to the API."
          );
        }

        return response.json();

      })

      .then((data) => {

        setSummary(data);

      })

      .catch((error) => {

        setError(error.message);

      });

  }, []);


  // ----------------------------------------------------------
  // LOAD TRANSACTION
  // ----------------------------------------------------------

  const loadTransaction = async () => {

    setError("");

    setResult(null);

    try {

      const response = await fetch(
        `${API_URL}/transaction/${transactionIndex}`
      );


      const data = await response.json();


      if (!response.ok) {

        throw new Error(
          data.error || "Transaction not found."
        );

      }


      setTransaction(data);

    } catch (error) {

      setError(error.message);

    }

  };


  // ----------------------------------------------------------
  // PREDICT
  // ----------------------------------------------------------

  const analyzeTransaction = async () => {

    setLoading(true);

    setError("");

    setResult(null);


    try {

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            model: model,
            index: Number(
              transactionIndex
            )
          })
        }
      );


      const data = await response.json();


      if (!response.ok) {

        throw new Error(
          data.error ||
          "Prediction failed."
        );

      }


      setResult(data);


    setTransaction({
      index: transactionIndex,
      amount: Number(data.amount || 0),
      time: data.time || "N/A",
      actual_class: data.actual_class
    });

    } catch (error) {

      setError(error.message);

    } finally {

      setLoading(false);

    }

  };


  return (

    <div className="app">

      {/* HEADER */}

      <header className="header">

        <div>

          <p className="eyebrow">
            MACHINE LEARNING PROJECT
          </p>

          <h1>
            Credit Card Fraud Detection
          </h1>

          <p className="intro">

            A simple machine learning application
            for identifying potentially fraudulent
            credit card transactions.

          </p>

        </div>

      </header>


      <main>


        {/* ERROR */}

        {error && (

          <div className="error">
            {error}
          </div>

        )}


        {/* DATASET */}

        <section>

          <div className="section-heading">

            <h2>
              Dataset Overview
            </h2>

            <p>
              Basic information about the
              transactions used for training.
            </p>

          </div>


          <div className="metrics">

            <div className="metric">

              <span>
                Total Transactions
              </span>

              <strong>
                {summary
                  ? summary.total_transactions.toLocaleString()
                  : "..."}
              </strong>

            </div>


            <div className="metric">

              <span>
                Fraud Cases
              </span>

              <strong>
                {summary
                  ? summary.fraud_cases.toLocaleString()
                  : "..."}
              </strong>

            </div>


            <div className="metric">

              <span>
                Normal Transactions
              </span>

              <strong>
                {summary
                  ? summary.normal_transactions.toLocaleString()
                  : "..."}
              </strong>

            </div>


            <div className="metric">

              <span>
                Fraud Rate
              </span>

              <strong>
                {summary
                  ? `${summary.fraud_rate}%`
                  : "..."}
              </strong>

            </div>

          </div>

        </section>


        {/* MODEL */}

        <section>

          <div className="section-heading">

            <h2>
              Test a Transaction
            </h2>

            <p>
              Select a machine learning model
              and a transaction from the dataset.
            </p>

          </div>


          <div className="test-card">


            <div className="form-group">

              <label>
                Machine Learning Model
              </label>

              <select
                value={model}
                onChange={(e) =>
                  setModel(e.target.value)
                }
              >

                <option>
                  Logistic Regression
                </option>

                <option>
                  Random Forest
                </option>

                <option>
                  Gradient Boosting
                </option>

              </select>

            </div>


            <div className="form-group">

              <label>
                Transaction Number
              </label>

              <input
                type="number"
                min="0"
                max={
                  summary
                    ? summary.total_transactions - 1
                    : undefined
                }
                value={transactionIndex}
                onChange={(e) =>
                  setTransactionIndex(
                    e.target.value
                  )
                }
              />

            </div>


            <div className="button-row">

              <button
                className="secondary-button"
                onClick={loadTransaction}
              >
                View Transaction
              </button>


              <button
                className="primary-button"
                onClick={analyzeTransaction}
                disabled={loading}
              >

                {loading
                  ? "Analyzing..."
                  : "Analyze Transaction"}

              </button>

            </div>


          </div>


          {/* TRANSACTION */}

          {transaction && (

  <div className="transaction-card">

    <div>
      <span>
        Transaction
      </span>

      <strong>
        #{transaction.index}
      </strong>
    </div>


    <div>
      <span>
        Amount
      </span>

      <strong>
        {transaction.amount !== undefined &&
        transaction.amount !== null
          ? `$${Number(transaction.amount).toFixed(2)}`
          : "N/A"}
      </strong>
    </div>


    <div>
      <span>
        Time
      </span>

      <strong>
        {transaction.time || "N/A"}
      </strong>
    </div>

  </div>

)}


        </section>


        {/* RESULT */}

        {result && (

          <section>

            <div className="section-heading">

              <h2>
                Prediction Result
              </h2>

            </div>


            <div className="result-card">

              <p className="result-label">
                {result.model}
              </p>


              <h3>
                {result.result}
              </h3>


              {result.probability !== null && (

                <p className="probability">

                  Fraud probability:

                  <strong>
                    {" "}
                    {(result.probability * 100).toFixed(2)}%
                  </strong>

                </p>

              )}


              <div className="result-details">

                <div>

                  <span>
                    Model prediction
                  </span>

                  <strong>
                    {result.prediction === 1
                      ? "Fraud"
                      : "Normal"}
                  </strong>

                </div>


                <div>

                  <span>
                    Dataset label
                  </span>

                  <strong>
                    {result.actual_class === 1
                      ? "Fraud"
                      : "Normal"}
                  </strong>

                </div>

              </div>

            </div>

          </section>

        )}


      </main>


      <footer>

        Credit Card Fraud Detection ·
        Machine Learning Portfolio Project

      </footer>

    </div>

  );

}


export default App;