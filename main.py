import os

import streamlit as st

from config import APP_TITLE, UPLOAD_DIR, get_llm
from data_loader import (
    calculate_business_metrics,
    detect_dataset_type,
    get_preview,
    get_statistics,
    load_dataset,
    validate_dataset,
)
from analyzer import detect_intent, execute_analysis, explain_result, generate_initial_insights

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title(f"📊 {APP_TITLE}")
st.caption("Upload a dataset, get instant metrics, and ask business questions in plain English.")

if "GROQ_API_KEY" not in st.secrets and not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it in Streamlit Cloud's Secrets settings.")
    st.stop()

if "df" not in st.session_state:
    st.session_state.df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, intent, result, response)

with st.sidebar:
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    st.caption("No file handy? Try the demo dataset:")
    load_sample = st.button("Load Sample Dataset")

    trigger_load = load_sample or (uploaded_file is not None and st.button("Load Dataset", type="primary"))

    if trigger_load:
        with st.spinner("Reading and analyzing dataset..."):
            try:
                if load_sample:
                    file_name = "filtered_sales.csv"
                    with open(file_name, "rb") as f:
                        file_bytes = f.read()
                else:
                    file_name = uploaded_file.name
                    file_bytes = uploaded_file.getvalue()
                    save_path = os.path.join(UPLOAD_DIR, file_name)
                    with open(save_path, "wb") as f:
                        f.write(file_bytes)

                df = load_dataset(file_bytes, file_name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read this file: {exc}")
                st.stop()

            warnings = validate_dataset(df)
            if warnings:
                for w in warnings:
                    st.warning(w)
                st.stop()

            llm = get_llm()
            metrics = calculate_business_metrics(df)
            dataset_summary = generate_initial_insights(df, metrics, llm=llm)

            st.session_state.df = df
            st.session_state.metrics = metrics
            st.session_state.dataset_summary = dataset_summary
            st.session_state.chat_history = []
        st.success("Dataset loaded successfully!")

if st.session_state.df is not None:
    df = st.session_state.df
    metrics = st.session_state.metrics
    dataset_summary = st.session_state.dataset_summary

    st.header("Dataset Summary")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Type: {dataset_summary.dataset_type}")
        st.write(dataset_summary.summary)

        if dataset_summary.data_quality_notes:
            st.subheader("Data Quality Notes")
            for note in dataset_summary.data_quality_notes:
                st.write(f"- {note}")

        if dataset_summary.suggested_questions:
            st.subheader("Try Asking")
            for suggestion in dataset_summary.suggested_questions:
                st.write(f"- {suggestion}")

    with col2:
        st.subheader("Business Metrics")
        metric_labels = {
            "total_sales": "Total Sales",
            "total_revenue": "Total Revenue",
            "total_profit": "Total Profit",
            "total_expenses": "Total Expenses",
            "average_order_value": "Average Order Value",
            "total_customers": "Total Customers",
            "total_products": "Total Products",
        }
        any_metric = False
        for field, label in metric_labels.items():
            value = getattr(metrics, field)
            if value is not None:
                any_metric = True
                st.metric(label, f"{value:,.2f}")
        if not any_metric:
            st.info("No standard business metric columns were detected.")

    with st.expander("Dataset Preview"):
        st.dataframe(get_preview(df))

    with st.expander("Missing Values & Duplicates"):
        stats = get_statistics(df)
        st.write("**Missing values by column:**")
        st.write(stats["missing_values"] or "None found.")
        st.write(f"**Duplicate rows:** {stats['duplicate_rows']}")

    st.divider()

    st.header("2. Ask a Business Question")
    st.caption(
        "Try: total sales, total profit, which region had the highest sales, "
        "top 5 products, how many customers, or ask for recommendations."
    )
    question = st.text_input("Your question")

    if st.button("Ask", disabled=not question.strip()):
        with st.spinner("Thinking..."):
            llm = get_llm()
            intent = detect_intent(question, df)
            result = execute_analysis(df, intent, metrics)
            response = explain_result(question, intent, result, dataset_summary, metrics, llm=llm)
            st.session_state.chat_history.insert(0, (question, intent, result, response))

    for question_asked, intent, result, response in st.session_state.chat_history:
        st.subheader(f"Q: {question_asked}")
        st.write(f"**Answer:** {response.answer}")
        st.write(f"**Explanation:** {response.explanation}")
        st.write(f"**Recommendation:** {response.recommendation}")
        with st.expander("Details"):
            st.write(f"Detected intent: `{intent['type']}`")
            st.json(result)
        st.divider()

else:
    st.info("Upload a dataset (CSV or Excel) from the sidebar and click **Load Dataset** to get started.")
