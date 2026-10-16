# DataDoctor AI — An Intelligent Data Quality Checker

DataDoctor AI is a portfolio-ready Data Science project that helps you inspect and improve dataset quality before analysis or machine learning.

## Features

- Upload CSV and Excel files with validation and graceful error handling
- Preview datasets and inspect schema, data types, and distributions
- Detect:
  - Missing values
  - Duplicate rows
  - Numeric outliers (IQR)
  - Data type issues
  - Suspicious columns
  - Constant columns
  - Inconsistent categorical values
- Transparent 0–100 data quality score with penalty breakdown
- Smart cleaning recommendations with user-controlled operations
- Before-and-after quality comparison dashboard
- Download cleaned data (CSV/Excel) and a detailed quality report
- Built-in `sample_data.csv` for quick testing

## Project Structure

Create these files in the repository root:

- `app.py` – Streamlit dashboard and workflow orchestration
- `data_quality.py` – quality checks, issue detection, and scoring
- `cleaning.py` – data cleaning operations and recommendations
- `sample_data.csv` – test dataset with intentional quality issues
- `requirements.txt` – Python dependencies
- `.gitignore` – common Python and Streamlit ignores

## Installation (VS Code on Windows)

```bash
# 1) Open terminal in project folder
cd DataDoctor-AI

# 2) Create virtual environment
python -m venv .venv

# 3) Activate virtual environment
.venv\Scripts\activate

# 4) Install dependencies
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How to Test the Workflow

1. Launch the app.
2. Upload your own CSV/Excel file or enable **Use sample_data.csv** from the sidebar.
3. Review **Overview**, **Data Explorer**, and **Quality Analysis** sections.
4. In **Data Cleaning**, select operations and click **Apply Cleaning**.
5. In **Reports**, compare before vs after metrics and download cleaned files/report.

## How Quality Score Is Calculated

Score starts at **100** and subtracts penalties:

- Missing values penalty (max 40)
- Duplicate rows penalty (max 20)
- Outlier penalty (max 20)
- Validation issues penalty (max 20)

Final score is clamped to `0–100` and shown with a full breakdown in the app.

## Upload to GitHub

```bash
git init
git add .
git commit -m "Build DataDoctor AI project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Notes

- The original uploaded dataset is never modified directly; cleaning runs on deep copies.
- The app avoids hardcoded metrics and computes all results from the active dataset.
