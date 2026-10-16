
import pandas as pd
import numpy as np


def clean_data(
    df,
    remove_duplicates=True,
    fill_missing=True,
    handle_outliers=False,
):
    """Return a cleaned copy without changing the original."""

    cleaned = df.copy()

    # Remove exact duplicate rows.
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    # Fill completely empty columns with a visible placeholder.
    for column in cleaned.columns:
        if cleaned[column].isnull().all():
            cleaned[column] = cleaned[column].astype("object")
            cleaned[column] = cleaned[column].fillna(
                "Unknown"
            )

    # Fill missing values in remaining columns.
    if fill_missing:
        for column in cleaned.columns:
            if cleaned[column].isnull().any():

                if pd.api.types.is_numeric_dtype(cleaned[column]):
                    median_value = cleaned[column].median()

                    if pd.notna(median_value):
                        cleaned[column] = cleaned[column].fillna(
                            median_value
                        )
                else:
                    mode_values = cleaned[column].mode(dropna=True)

                    if not mode_values.empty:
                        cleaned[column] = cleaned[column].fillna(
                            mode_values.iloc[0]
                        )
                    else:
                        cleaned[column] = cleaned[column].fillna(
                            "Unknown"
                        )

    # Optionally cap numeric outliers using IQR bounds.
    if handle_outliers:
        for column in cleaned.select_dtypes(
            include=np.number
        ).columns:

            values = cleaned[column].dropna()

            if len(values) < 4 or values.nunique() < 2:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            cleaned[column] = cleaned[column].clip(
                lower=lower,
                upper=upper,
            )

    return cleaned
