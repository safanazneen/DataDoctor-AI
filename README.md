
# 🩺 DataDoctor AI
### An Intelligent Data Quality Checker

DataDoctor AI is a Python-based data quality analysis application
that helps users identify common data issues, review cleaning
recommendations, and compare datasets before and after cleaning.

## Features

- Upload CSV and Excel datasets
- Detect missing values and duplicate rows
- Identify potential numeric outliers using the IQR method
- Flag suspicious columns
- Calculate a transparent heuristic data quality score
- Apply optional data cleaning
- Compare before-and-after quality metrics
- Download the cleaned dataset and quality report
- Interactive visualizations using Plotly

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Openpyxl

## Project Structure

    DataDoctor-AI/
    ├── app.py
    ├── data_quality.py
    ├── cleaning.py
    ├── requirements.txt
    ├── sample_data.csv
    └── README.md

## Installation

1. Clone this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install the dependencies:

       python -m pip install -r requirements.txt

5. Run the application:

       python -m streamlit run app.py

6. Upload sample_data.csv or your own test dataset.

## How It Works

1. Upload a dataset.
2. Review the initial data quality analysis.
3. Inspect missing values, duplicates, outliers and suspicious columns.
4. Select the cleaning operations you want.
5. Apply cleaning and compare the results.
6. Download the cleaned dataset and report.

## Limitations

- The quality score is a heuristic, not a universal standard.
- IQR-based outlier detection may flag valid observations.
- Filling missing values can affect statistical distributions.
- Review cleaning decisions before using data for analysis or ML.

## Future Improvements

- PDF report generation
- More configurable cleaning recommendations
- Additional data validation rules
- Automated data type and date-format checks
- Model-based anomaly detection
- Deployment as a public web application

## Author

Safa Nazneen
B.Tech in Data Science
