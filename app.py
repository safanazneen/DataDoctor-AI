
import streamlit as st
import pandas as pd
import plotly.express as px

from data_quality import analyze_data
from cleaning import clean_data


st.set_page_config(
    page_title="DataDoctor AI",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 DataDoctor AI")
st.subheader("Intelligent Data Quality Checker")

st.write(
    "Upload a dataset, discover data-quality issues, "
    "review cleaning recommendations, and download your results."
)

st.info(
    "Your data is processed in this app session. "
    "Use non-sensitive sample data while testing."
)


def read_dataset(uploaded_file):
    """Read CSV or Excel uploads safely."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV or Excel file.")


uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:
    try:
        original_df = read_dataset(uploaded_file)

        if original_df.empty:
            st.warning("This dataset has no rows to analyze.")
            st.stop()

        if len(original_df.columns) == 0:
            st.warning("This dataset has no columns.")
            st.stop()

        # Reset cleaned data when a different file is uploaded.
        file_signature = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if st.session_state.get("file_signature") != file_signature:
            st.session_state["file_signature"] = file_signature
            st.session_state["cleaned_df"] = original_df.copy()

        st.success("Dataset uploaded successfully!")

        st.header("1. Dataset Preview")

        st.dataframe(original_df.head(10), use_container_width=True)

        st.caption(
            f"{original_df.shape[0]} rows · "
            f"{original_df.shape[1]} columns"
        )

        before = analyze_data(original_df)

        st.header("2. Data Quality Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Quality Score", f"{before['quality_score']}/100")
        col2.metric("Missing Values", before["missing_total"])
        col3.metric("Duplicate Rows", before["duplicate_total"])
        col4.metric("Detected Outliers", before["outlier_total"])

        score = before["quality_score"]

        if score >= 85:
            st.success("Good initial data-quality score.")
        elif score >= 60:
            st.warning("Some data-quality issues need attention.")
        else:
            st.error("Several data-quality issues need review.")

        st.caption(
            "The score is a heuristic based on missing cells, "
            "duplicate rows and detected numeric outliers. "
            "It does not measure every aspect of data quality."
        )

        st.header("3. Missing Values")

        missing_df = pd.DataFrame({
            "Column": before["missing_by_column"].index,
            "Missing Values": before["missing_by_column"].values,
        })

        missing_df = missing_df[
            missing_df["Missing Values"] > 0
        ]

        if missing_df.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(missing_df, use_container_width=True)

            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing Values",
                title="Missing Values by Column",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.header("4. Outlier Analysis")

        outlier_df = pd.DataFrame(
            list(before["outlier_by_column"].items()),
            columns=["Numeric Column", "Outlier Count"],
        )

        outlier_df = outlier_df[
            outlier_df["Outlier Count"] > 0
        ]

        if outlier_df.empty:
            st.write("No numeric outliers detected by the IQR rule.")
        else:
            st.dataframe(outlier_df, use_container_width=True)

            fig = px.bar(
                outlier_df,
                x="Numeric Column",
                y="Outlier Count",
                title="Potential Outliers by Numeric Column",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.header("5. Suspicious Columns")

        if not before["suspicious_columns"]:
            st.success("No suspicious columns detected by these rules.")
        else:
            suspicious_df = pd.DataFrame(
                before["suspicious_columns"],
                columns=["Column", "Issue"],
            )
            st.dataframe(suspicious_df, use_container_width=True)

        st.header("6. Cleaning Recommendations")

        st.write(
            "Select the cleaning operations you want to apply. "
            "Review the output before using it for further analysis."
        )

        remove_duplicates = st.checkbox(
            "Remove exact duplicate rows",
            value=True,
        )

        fill_missing = st.checkbox(
            "Fill missing values (numeric median, "
            "categorical mode)",
            value=True,
        )

        handle_outliers = st.checkbox(
            "Cap numeric outliers using IQR bounds",
            value=False,
        )

        if st.button("Apply Selected Cleaning", type="primary"):
            st.session_state["cleaned_df"] = clean_data(
                original_df,
                remove_duplicates=remove_duplicates,
                fill_missing=fill_missing,
                handle_outliers=handle_outliers,
            )

            st.success("Cleaning completed. Review the results below.")

        cleaned_df = st.session_state["cleaned_df"]
        after = analyze_data(cleaned_df)

        st.header("7. Before vs After Cleaning")

        comparison = pd.DataFrame({
            "Metric": [
                "Rows",
                "Missing Values",
                "Duplicate Rows",
                "Detected Outliers",
                "Quality Score",
            ],
            "Before": [
                before["rows"],
                before["missing_total"],
                before["duplicate_total"],
                before["outlier_total"],
                before["quality_score"],
            ],
            "After": [
                after["rows"],
                after["missing_total"],
                after["duplicate_total"],
                after["outlier_total"],
                after["quality_score"],
            ],
        })

        st.dataframe(comparison, use_container_width=True)

        score_change = round(
            after["quality_score"] - before["quality_score"], 1
        )

        st.metric(
            "Change in Quality Score",
            f"{score_change:+.1f} points",
        )

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head(10), use_container_width=True)

        st.download_button(
            "Download Cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="datadoctor_cleaned.csv",
            mime="text/csv",
        )

        report = f"""DATADOCTOR AI - DATA QUALITY REPORT

Original rows: {before['rows']}
Original columns: {before['columns']}
Original missing values: {before['missing_total']}
Original duplicate rows: {before['duplicate_total']}
Original detected outliers: {before['outlier_total']}
Original quality score: {before['quality_score']}/100

Cleaned rows: {after['rows']}
Cleaned missing values: {after['missing_total']}
Cleaned duplicate rows: {after['duplicate_total']}
Cleaned detected outliers: {after['outlier_total']}
Cleaned quality score: {after['quality_score']}/100

Score change: {score_change:+.1f} points

Note: The score is a heuristic, not a universal measure.
Review all cleaning decisions before using the dataset.
"""

        st.download_button(
            "Download Quality Report",
            data=report,
            file_name="datadoctor_report.txt",
            mime="text/plain",
        )

    except Exception as error:
        st.error(
            "The file could not be processed. "
            "Check that it is a valid CSV or Excel file."
        )
        st.caption(f"Technical details: {error}")

else:
    st.info(
        "Upload a CSV or Excel file to begin your data-quality analysis."
    )

    st.subheader("What DataDoctor AI checks")
    st.markdown(
        "- Missing values\n"
        "- Exact duplicate rows\n"
        "- Potential numeric outliers\n"
        "- Empty, constant and high-missingness columns\n"
        "- Data quality before and after cleaning"
    )
