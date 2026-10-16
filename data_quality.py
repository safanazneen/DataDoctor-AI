from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ScoreBreakdown:
    missing_penalty: float
    duplicate_penalty: float
    outlier_penalty: float
    validation_penalty: float

    @property
    def total_penalty(self) -> float:
        return (
            self.missing_penalty
            + self.duplicate_penalty
            + self.outlier_penalty
            + self.validation_penalty
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def detect_outliers_iqr(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    outlier_summary: dict[str, dict[str, Any]] = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = int(mask.sum())

        if outlier_count > 0:
            outlier_summary[column] = {
                "count": outlier_count,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }

    return outlier_summary


def detect_inconsistent_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    inconsistencies: dict[str, list[str]] = {}

    for column in df.select_dtypes(include=["object", "category"]).columns:
        values = df[column].dropna().astype(str)
        if values.empty:
            continue

        normalized_map: dict[str, set[str]] = {}
        for value in values:
            cleaned = " ".join(value.strip().split()).lower()
            normalized_map.setdefault(cleaned, set()).add(value)

        variants = [sorted(list(group)) for group in normalized_map.values() if len(group) > 1]
        if variants:
            inconsistencies[column] = [", ".join(group) for group in variants]

    return inconsistencies


def detect_type_issues(df: pd.DataFrame) -> dict[str, str]:
    issues: dict[str, str] = {}

    for column in df.select_dtypes(include=["object"]).columns:
        non_null = df[column].dropna().astype(str).str.strip()
        if non_null.empty:
            continue

        numeric_parsed = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = _safe_ratio(numeric_parsed.notna().sum(), len(non_null))

        if numeric_ratio > 0.9:
            issues[column] = "Mostly numeric values stored as text."
            continue

        datetime_parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        datetime_ratio = _safe_ratio(datetime_parsed.notna().sum(), len(non_null))
        if datetime_ratio > 0.9:
            issues[column] = "Mostly date-like values stored as text."

    return issues


def detect_suspicious_columns(df: pd.DataFrame) -> list[str]:
    suspicious: list[str] = []

    for column in df.columns:
        lowered = column.lower()
        if any(token in lowered for token in ["unnamed", "temp", "misc"]):
            suspicious.append(column)
            continue

        unique_ratio = _safe_ratio(df[column].nunique(dropna=False), len(df))
        if (
            unique_ratio > 0.98
            and df[column].dtype == "object"
            and not lowered.endswith("id")
            and "id" not in lowered
        ):
            suspicious.append(column)

    return sorted(set(suspicious))


def calculate_quality_score(df: pd.DataFrame, analysis: dict[str, Any]) -> tuple[float, ScoreBreakdown]:
    rows, cols = df.shape
    total_cells = rows * cols if rows and cols else 0

    missing_cells = analysis["missing"]["total_missing"]
    duplicate_rows = analysis["duplicates"]["count"]
    total_outliers = analysis["outliers"]["total_count"]

    missing_ratio = _safe_ratio(missing_cells, total_cells)
    duplicate_ratio = _safe_ratio(duplicate_rows, rows)

    numeric_columns_count = max(1, len(df.select_dtypes(include=[np.number]).columns))
    outlier_denominator = rows * numeric_columns_count if rows else 0
    outlier_ratio = _safe_ratio(total_outliers, outlier_denominator)

    validation_issue_count = (
        len(analysis["type_issues"])
        + len(analysis["constant_columns"])
        + len(analysis["suspicious_columns"])
        + sum(len(v) for v in analysis["inconsistent_categories"].values())
    )
    validation_denominator = max(1, cols * 2)
    validation_ratio = min(1.0, _safe_ratio(validation_issue_count, validation_denominator))

    breakdown = ScoreBreakdown(
        missing_penalty=min(40.0, missing_ratio * 100.0 * 0.40),
        duplicate_penalty=min(20.0, duplicate_ratio * 100.0 * 0.20),
        outlier_penalty=min(20.0, outlier_ratio * 100.0 * 0.20),
        validation_penalty=min(20.0, validation_ratio * 20.0),
    )

    score = max(0.0, 100.0 - breakdown.total_penalty)
    return round(score, 2), breakdown


def analyze_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("Dataset is empty. Please upload a file with rows and columns.")

    missing_per_column = df.isna().sum()
    total_missing = int(missing_per_column.sum())

    duplicates_count = int(df.duplicated().sum())
    outliers = detect_outliers_iqr(df)
    outlier_total = int(sum(item["count"] for item in outliers.values()))

    constant_columns = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 1
    ]
    inconsistent_categories = detect_inconsistent_categories(df)
    type_issues = detect_type_issues(df)
    suspicious_columns = detect_suspicious_columns(df)

    analysis: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing": {
            "total_missing": total_missing,
            "missing_percentage": round(_safe_ratio(total_missing, max(1, df.size)) * 100, 2),
            "by_column": missing_per_column[missing_per_column > 0].sort_values(ascending=False).to_dict(),
        },
        "duplicates": {
            "count": duplicates_count,
            "percentage": round(_safe_ratio(duplicates_count, max(1, len(df))) * 100, 2),
        },
        "outliers": {
            "total_count": outlier_total,
            "by_column": outliers,
        },
        "type_issues": type_issues,
        "constant_columns": constant_columns,
        "suspicious_columns": suspicious_columns,
        "inconsistent_categories": inconsistent_categories,
    }

    score, breakdown = calculate_quality_score(df, analysis)
    analysis["quality_score"] = score
    analysis["score_breakdown"] = {
        "missing_penalty": round(breakdown.missing_penalty, 2),
        "duplicate_penalty": round(breakdown.duplicate_penalty, 2),
        "outlier_penalty": round(breakdown.outlier_penalty, 2),
        "validation_penalty": round(breakdown.validation_penalty, 2),
        "total_penalty": round(breakdown.total_penalty, 2),
    }

    return analysis


def build_issue_explanations(analysis: dict[str, Any]) -> list[str]:
    explanations: list[str] = []

    if analysis["missing"]["total_missing"] > 0:
        explanations.append(
            "Missing values can bias analysis and break machine learning pipelines if not handled."
        )
    if analysis["duplicates"]["count"] > 0:
        explanations.append(
            "Duplicate rows can inflate trends and distort model training distributions."
        )
    if analysis["outliers"]["total_count"] > 0:
        explanations.append(
            "Outliers may indicate data entry errors or rare events that can skew statistics."
        )
    if analysis["type_issues"]:
        explanations.append(
            "Incorrect data types can cause failed calculations and wrong model features."
        )
    if analysis["constant_columns"]:
        explanations.append(
            "Constant columns add no predictive value and can be removed safely."
        )
    if analysis["inconsistent_categories"]:
        explanations.append(
            "Inconsistent category spelling/casing splits the same value into multiple groups."
        )
    if analysis["suspicious_columns"]:
        explanations.append(
            "Suspicious columns may be placeholders or noisy fields that require review."
        )

    return explanations
