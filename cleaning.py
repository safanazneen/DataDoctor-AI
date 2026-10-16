from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def get_cleaning_recommendations(analysis: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []

    if analysis["missing"]["total_missing"] > 0:
        recommendations.append(
            "Handle missing values using median/mode imputation to preserve rows when possible."
        )
    if analysis["duplicates"]["count"] > 0:
        recommendations.append("Remove duplicate rows to avoid duplicated evidence in analysis.")
    if analysis["outliers"]["total_count"] > 0:
        recommendations.append(
            "Cap outliers using IQR bounds before modeling to reduce extreme-value distortion."
        )
    if analysis["type_issues"]:
        recommendations.append(
            "Convert numeric/date-like text columns to the correct data type for accurate operations."
        )
    if analysis["inconsistent_categories"]:
        recommendations.append(
            "Standardize categorical text (trim spaces, consistent capitalization)."
        )
    if analysis["constant_columns"]:
        recommendations.append("Drop constant columns that do not carry information.")

    if not recommendations:
        recommendations.append("No critical issues detected. Dataset is already in good shape.")

    return recommendations


def _cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy(deep=True)
    for column in cleaned.select_dtypes(include=[np.number]).columns:
        series = cleaned[column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned[column] = series.clip(lower=lower, upper=upper)
    return cleaned


def _remove_outlier_rows_iqr(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy(deep=True)

    mask = pd.Series(False, index=df.index)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        series = df[column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = mask | (series < lower) | (series > upper)

    return df.loc[~mask].copy(deep=True)


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy(deep=True)

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = cleaned.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="median")
        cleaned[numeric_cols] = numeric_imputer.fit_transform(cleaned[numeric_cols])

    if object_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cleaned[object_cols] = cat_imputer.fit_transform(cleaned[object_cols])

    return cleaned


def _convert_type_issues(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy(deep=True)

    for column in cleaned.select_dtypes(include=["object"]).columns:
        non_null = cleaned[column].dropna().astype(str).str.strip()
        if non_null.empty:
            continue

        numeric_parsed = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = numeric_parsed.notna().mean()
        if numeric_ratio > 0.9:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            continue

        datetime_parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        datetime_ratio = datetime_parsed.notna().mean()
        if datetime_ratio > 0.9:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce", format="mixed")

    return cleaned


def _standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy(deep=True)
    for column in cleaned.select_dtypes(include=["object", "category"]).columns:
        cleaned[column] = (
            cleaned[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace("nan", np.nan)
            .str.title()
        )
    return cleaned


def apply_cleaning_operations(
    source_df: pd.DataFrame,
    fill_missing: bool,
    remove_duplicates: bool,
    outlier_strategy: str,
    convert_types: bool,
    standardize_categories: bool,
    drop_constant_columns: bool,
) -> pd.DataFrame:
    cleaned = source_df.copy(deep=True)

    if fill_missing:
        cleaned = _impute_missing(cleaned)

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates().copy(deep=True)

    if outlier_strategy == "Cap (IQR)":
        cleaned = _cap_outliers_iqr(cleaned)
    elif outlier_strategy == "Remove rows (IQR)":
        cleaned = _remove_outlier_rows_iqr(cleaned)

    if convert_types:
        cleaned = _convert_type_issues(cleaned)

    if standardize_categories:
        cleaned = _standardize_categories(cleaned)

    if drop_constant_columns:
        constant_cols = [
            col for col in cleaned.columns if cleaned[col].nunique(dropna=False) <= 1
        ]
        if constant_cols:
            cleaned = cleaned.drop(columns=constant_cols)

    return cleaned
