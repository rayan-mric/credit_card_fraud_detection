import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
)

from src.config import (
    RAW_DATA_PATH,
    METRICS_DIR,
    MODELS_DIR,
)

from src.data_preprocessing import (
    load_data,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Analytics",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    'Credit Card Fraud Detection'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning analytics dashboard for '
    'transaction fraud detection'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def get_data():

    if not RAW_DATA_PATH.exists():
        return None

    return load_data(
        RAW_DATA_PATH
    )


df = get_data()


if df is None:

    st.error(
        "Dataset not found. "
        "Place creditcard.csv inside "
        "data/raw/"
    )

    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title(
    "Fraud Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Comparison",
        "Transaction Explorer",
    ],
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header(
        "Dataset Overview"
    )

    total_transactions = len(df)

    fraud_count = int(
        df["Class"].sum()
    )

    normal_count = (
        total_transactions
        - fraud_count
    )

    fraud_rate = (
        fraud_count
        / total_transactions
        * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}",
    )

    col2.metric(
        "Fraud Transactions",
        f"{fraud_count:,}",
    )

    col3.metric(
        "Normal Transactions",
        f"{normal_count:,}",
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_rate:.3f}%",
    )

    st.divider()

    st.subheader(
        "Transaction Class Distribution"
    )

    class_counts = (
        df["Class"]
        .value_counts()
        .rename(
            {
                0: "Normal",
                1: "Fraud",
            }
        )
    )

    st.bar_chart(
        class_counts
    )

    st.subheader(
        "Transaction Amount Distribution"
    )

    st.line_chart(
        df["Amount"]
        .head(1000)
    )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.header(
        "Model Performance"
    )

    results_path = (
        METRICS_DIR
        / "model_comparison.csv"
    )

    if not results_path.exists():

        st.warning(
            "Model results have not been generated yet."
        )

        st.info(
            "Run `python main.py` first."
        )

        st.stop()

    results = pd.read_csv(
        results_path
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "F1 Score Comparison"
    )

    chart_data = results.set_index(
        "Model"
    )["F1 Score"]

    st.bar_chart(
        chart_data
    )

    st.subheader(
        "Precision vs Recall"
    )

    precision_recall = results[
        [
            "Model",
            "Precision",
            "Recall",
        ]
    ].set_index("Model")

    st.bar_chart(
        precision_recall
    )


# --------------------------------------------------
# TRANSACTION EXPLORER
# --------------------------------------------------

elif page == "Transaction Explorer":

    st.header(
        "Transaction Explorer"
    )

    st.write(
        "Inspect individual transactions "
        "from the dataset."
    )

    sample_size = st.slider(
        "Number of transactions",
        min_value=10,
        max_value=500,
        value=50,
    )

    fraud_only = st.checkbox(
        "Show fraud transactions only"
    )

    if fraud_only:

        filtered = df[
            df["Class"] == 1
        ].head(sample_size)

    else:

        filtered = df.head(
            sample_size
        )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Transaction Statistics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Average Amount",
            f"${filtered['Amount'].mean():.2f}",
        )

    with col2:

        st.metric(
            "Maximum Amount",
            f"${filtered['Amount'].max():.2f}",
        )