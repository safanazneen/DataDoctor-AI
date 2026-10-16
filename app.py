from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from cleaning import apply_cleaning_operations, get_cleaning_recommendations
from data_quality import analyze_data_quality, build_issue_explanations


st.set_page_config(page_title="DataDoctor AI", page_icon="🩺", layout="wide")


def _read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    if uploaded_file.size == 0:
        raise ValueError("Uploaded file is empty.")

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format. Upload CSV or Excel (.xlsx/.xls).")


def _quality_cards(analysis: dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Quality Score", f"{analysis['quality_score']} / 100")
    c2.metric("Missing Values", analysis["missing"]["total_missing"])
    c3.metric("Duplicate Rows", analysis["duplicates"]["count"])
    c4.metric("Outliers", analysis["outliers"]["total_count"])
    c5.metric("Constant Columns", len(analysis["constant_columns"]))


def _show_score_breakdown(analysis: dict[str, Any]) -> None:
    breakdown = analysis["score_breakdown"]
    st.write("### Score Explanation")
    st.caption(
        "Score starts at 100. Penalties are applied for missing values, duplicates, outliers, and validation issues."
    )
    st.json(
        {
            "Missing Penalty (max 40)": breakdown["missing_penalty"],
            "Duplicate Penalty (max 20)": breakdown["duplicate_penalty"],
            "Outlier Penalty (max 20)": breakdown["outlier_penalty"],
            "Validation Penalty (max 20)": breakdown["validation_penalty"],
            "Total Penalty": breakdown["total_penalty"],
            "Final Score": analysis["quality_score"],
        }
    )


def _download_cleaned_dataset(cleaned_df: pd.DataFrame) -> None:
    st.write("### Download Cleaned Dataset")

    csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        cleaned_df.to_excel(writer, index=False, sheet_name="cleaned_data")
    st.download_button(
        "Download as Excel",
        data=excel_buffer.getvalue(),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _download_report(
    file_name: str,
    before_analysis: dict[str, Any],
    after_analysis: dict[str, Any],
    recommendations: list[str],
) -> None:
    timestamp = datetime.utcnow().isoformat()
    report_lines = [
        "DataDoctor AI - Data Quality Report",
        f"Generated at (UTC): {timestamp}",
        f"Source file: {file_name}",
        "",
        "Before Cleaning:",
        f"- Quality score: {before_analysis['quality_score']}",
        f"- Missing values: {before_analysis['missing']['total_missing']}",
        f"- Duplicate rows: {before_analysis['duplicates']['count']}",
        f"- Outliers: {before_analysis['outliers']['total_count']}",
        "",
        "After Cleaning:",
        f"- Quality score: {after_analysis['quality_score']}",
        f"- Missing values: {after_analysis['missing']['total_missing']}",
        f"- Duplicate rows: {after_analysis['duplicates']['count']}",
        f"- Outliers: {after_analysis['outliers']['total_count']}",
        "",
        "Recommendations:",
    ]
    report_lines.extend([f"- {item}" for item in recommendations])

    report_content = "\n".join(report_lines)
    st.download_button(
        "Download Quality Report (TXT)",
        data=report_content.encode("utf-8"),
        file_name="data_quality_report.txt",
        mime="text/plain",
    )


def main() -> None:
    st.title("🩺 DataDoctor AI — An Intelligent Data Quality Checker")
    st.caption("Upload a dataset, inspect quality issues, apply cleaning actions, and export results.")

    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Go to",
        ["Overview", "Data Explorer", "Quality Analysis", "Data Cleaning", "Reports"],
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"]
    )

    use_sample = st.sidebar.checkbox("Use sample_data.csv")

    df: pd.DataFrame | None = None
    source_name = ""

    try:
        if use_sample:
            df = pd.read_csv("sample_data.csv")
            source_name = "sample_data.csv"
        elif uploaded_file is not None:
            df = _read_uploaded_file(uploaded_file)
            source_name = uploaded_file.name
    except Exception as exc:
        st.error(f"Unable to load dataset: {exc}")

    if df is None:
        st.info("Upload a CSV/Excel file or enable sample dataset to begin.")
        return

    if df.empty:
        st.warning("The uploaded dataset has no rows. Please upload a non-empty dataset.")
        return

    original_df = df.copy(deep=True)

    try:
        analysis_before = analyze_data_quality(original_df)
    except Exception as exc:
        st.error(f"Failed to analyze dataset: {exc}")
        return

    recommendations = get_cleaning_recommendations(analysis_before)

    if section == "Overview":
        _quality_cards(analysis_before)
        st.write("### Dataset Snapshot")
        st.dataframe(original_df.head(20), use_container_width=True)
        _show_score_breakdown(analysis_before)

    elif section == "Data Explorer":
        st.write("### Dataset Shape")
        st.write(analysis_before["shape"])
        st.write("### Data Types")
        st.dataframe(original_df.dtypes.astype(str).reset_index().rename(columns={"index": "column", 0: "dtype"}))

        numeric_cols = original_df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            selected = st.selectbox("Select numeric column", numeric_cols)
            fig = px.histogram(original_df, x=selected, title=f"Distribution of {selected}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for distribution charts.")

    elif section == "Quality Analysis":
        _quality_cards(analysis_before)

        st.write("### Missing Values by Column")
        missing_items = analysis_before["missing"]["by_column"]
        if missing_items:
            missing_df = pd.DataFrame(
                {"column": list(missing_items.keys()), "missing_count": list(missing_items.values())}
            )
            fig = px.bar(missing_df, x="column", y="missing_count", title="Missing Values")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values detected.")

        st.write("### Detected Issues")
        issue_explanations = build_issue_explanations(analysis_before)
        if issue_explanations:
            for issue in issue_explanations:
                st.write(f"- {issue}")
        else:
            st.success("No major data quality issues detected.")

        st.write("### Detailed Findings")
        st.json(
            {
                "type_issues": analysis_before["type_issues"],
                "constant_columns": analysis_before["constant_columns"],
                "suspicious_columns": analysis_before["suspicious_columns"],
                "inconsistent_categories": analysis_before["inconsistent_categories"],
                "outlier_details": analysis_before["outliers"]["by_column"],
            }
        )

    elif section == "Data Cleaning":
        st.write("### Smart Cleaning Recommendations")
        for recommendation in recommendations:
            st.write(f"- {recommendation}")

        c1, c2 = st.columns(2)
        with c1:
            fill_missing = st.checkbox("Impute missing values (median/mode)", value=True)
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            drop_constant = st.checkbox("Drop constant columns", value=False)

        with c2:
            outlier_strategy = st.selectbox(
                "Outlier handling",
                ["None", "Cap (IQR)", "Remove rows (IQR)"],
                index=1,
            )
            convert_types = st.checkbox("Convert data type issues", value=True)
            standardize_categories = st.checkbox("Standardize categorical text", value=True)

        if st.button("Apply Cleaning"):
            cleaned_df = apply_cleaning_operations(
                source_df=original_df,
                fill_missing=fill_missing,
                remove_duplicates=remove_duplicates,
                outlier_strategy=outlier_strategy,
                convert_types=convert_types,
                standardize_categories=standardize_categories,
                drop_constant_columns=drop_constant,
            )
            st.session_state["cleaned_df"] = cleaned_df
            st.success("Cleaning operations applied successfully.")

        if "cleaned_df" in st.session_state:
            cleaned_df = st.session_state["cleaned_df"]
            st.write("### Cleaned Data Preview")
            st.dataframe(cleaned_df.head(20), use_container_width=True)

    elif section == "Reports":
        cleaned_df = st.session_state.get("cleaned_df", original_df.copy(deep=True))
        analysis_after = analyze_data_quality(cleaned_df)

        st.write("### Before vs After Comparison")
        comparison = pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Columns",
                    "Missing Values",
                    "Duplicate Rows",
                    "Outliers",
                    "Quality Score",
                ],
                "Before": [
                    analysis_before["shape"]["rows"],
                    analysis_before["shape"]["columns"],
                    analysis_before["missing"]["total_missing"],
                    analysis_before["duplicates"]["count"],
                    analysis_before["outliers"]["total_count"],
                    analysis_before["quality_score"],
                ],
                "After": [
                    analysis_after["shape"]["rows"],
                    analysis_after["shape"]["columns"],
                    analysis_after["missing"]["total_missing"],
                    analysis_after["duplicates"]["count"],
                    analysis_after["outliers"]["total_count"],
                    analysis_after["quality_score"],
                ],
            }
        )
        st.dataframe(comparison, use_container_width=True)

        fig = px.bar(comparison, x="Metric", y=["Before", "After"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        _download_cleaned_dataset(cleaned_df)
        _download_report(source_name, analysis_before, analysis_after, recommendations)


if __name__ == "__main__":
    main()
