
import pandas as pd
import numpy as np


def analyze_data(df):
    """Analyze the quality of a pandas DataFrame."""

    rows, columns = df.shape

    missing_by_column = df.isnull().sum()
    missing_total = int(missing_by_column.sum())

    duplicate_total = int(df.duplicated().sum())

    # Detect outliers using the IQR method.
    outlier_by_column = {}

    for column in df.select_dtypes(include=np.number).columns:
        values = df[column].dropna()

        if len(values) < 4 or values.nunique() < 2:
            outlier_by_column[column] = 0
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_by_column[column] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_by_column[column] = int(
            ((values < lower) | (values > upper)).sum()
        )

    outlier_total = sum(outlier_by_column.values())

    # Flag columns that may need attention.
    suspicious_columns = []

    for column in df.columns:
        if df[column].isnull().all():
            suspicious_columns.append(
                (column, "Entire column is missing")
            )
        elif df[column].nunique(dropna=True) <= 1:
            suspicious_columns.append(
                (column, "Constant or nearly empty column")
            )
        elif df[column].isnull().mean() > 0.40:
            suspicious_columns.append(
                (column, "More than 40% of values are missing")
            )

    total_cells = max(rows * columns, 1)

    # A transparent heuristic, not a statistical guarantee.
    missing_rate = missing_total / total_cells
    duplicate_rate = duplicate_total / max(rows, 1)
    outlier_rate = outlier_total / max(rows * max(len(outlier_by_column), 1), 1)

    score = round(
        max(0, min(100,
            100
            - 50 * missing_rate * 100
            - 30 * duplicate_rate * 100
            - 20 * outlier_rate * 100
        )),
        1
    )

    return {
        "rows": rows,
        "columns": columns,
        "missing_total": missing_total,
        "missing_by_column": missing_by_column,
        "duplicate_total": duplicate_total,
        "outlier_total": outlier_total,
        "outlier_by_column": outlier_by_column,
        "suspicious_columns": suspicious_columns,
        "quality_score": score,
    }
