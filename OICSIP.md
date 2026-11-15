# Product Requirements Document (PRD)
## Oasis Infobyte — Data Analytics Internship Projects

**Document filename:** `OICSIP.md`  
**Purpose:** Implementation brief for Claude Code  
**Selected projects:** Three projects chosen to satisfy the minimum Data Analytics task requirement  
**Required GitHub repository name:** `OIBSIP` (exactly, all caps)

---

## 1. Project Overview

Build three complete, well-documented Python data analytics projects for the Oasis Infobyte Data Analytics internship. Each project must be understandable to a beginner, reproducible, and suitable for a public GitHub portfolio and LinkedIn demo.

### Selected projects

1. **Level 1 — EDA on Retail Sales Data**
2. **Level 1 — Customer Segmentation Analysis**
3. **Level 2 — Predicting House Prices with Linear Regression**

The internship task list says Data Analytics interns must complete at least three tasks across Level 1 and Level 2 combined. Implement the full checklist for each selected task; do not substitute a different project.

## 2. Objectives

- Complete the required analytics tasks in a structured, reproducible way.
- Demonstrate data loading, cleaning, exploratory analysis, visualization, clustering, regression, and interpretation.
- Produce notebook outputs, charts, a concise README for each project, and clear run instructions.
- Make the repository suitable for review by internship evaluators and recruiters.
- Keep claims grounded in actual dataset results; never fabricate metrics, insights, charts, or findings.

## 3. Required Repository Structure

Create or work inside a repository named **`OIBSIP`**. Do not create separate repositories for each task.

```text
OIBSIP/
├── README.md
├── requirements.txt
├── .gitignore
├── DataAnalytics-L1-EDARetailSales/
│   ├── README.md
│   ├── notebooks/
│   │   └── retail_sales_eda.ipynb
│   ├── data/
│   │   ├── raw/                 # user-provided/downloaded dataset; do not commit large/private data
│   │   └── processed/
│   ├── outputs/
│   │   └── figures/
│   └── reports/
├── DataAnalytics-L1-CustomerSegmentation/
│   ├── README.md
│   ├── notebooks/
│   │   └── customer_segmentation.ipynb
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   └── outputs/
│       └── figures/
└── DataAnalytics-L2-HousePricePrediction/
    ├── README.md
    ├── notebooks/
    │   └── house_price_prediction.ipynb
    ├── data/
    │   ├── raw/
    │   └── processed/
    └── outputs/
        └── figures/
```

If the repository already exists, inspect it first and preserve unrelated files. Do not overwrite user work. Keep the required project folder names exactly as shown above. Add generated files only where appropriate.

## 4. Technical Requirements

- Python 3.10 or later (prefer a version compatible with installed packages).
- Jupyter notebooks (`.ipynb`) as the primary deliverables.
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`.
- Use clear, descriptive variable names and modular helper functions where they improve readability.
- Include markdown explanations before major steps and observations after each important chart or model result.
- Set random seeds where relevant for reproducibility.
- Use relative paths, `pathlib`, and a clearly documented expected dataset location. Avoid machine-specific absolute paths.
- Never hard-code fabricated results. All tables, charts, metrics, and written conclusions must be calculated from the loaded data.
- Handle missing files with a helpful error explaining the expected path and how to obtain/place the dataset.
- Do not silently download large datasets or require credentials/API keys. If a dataset cannot be obtained automatically, document the official source and manual placement instructions.
- Do not commit secrets, personal data, huge raw files, or generated environment folders.

## 5. Project 1 — EDA on Retail Sales Data

**Folder:** `DataAnalytics-L1-EDARetailSales/`  
**Notebook:** `notebooks/retail_sales_eda.ipynb`  
**Objective:** Explore retail sales data to identify sales patterns, customer behavior, product performance, and actionable business insights.

### Required implementation checklist

1. Load the dataset and inspect shape, columns, data types, and null values.
2. Clean or document missing/invalid values where necessary; explain decisions.
3. Calculate mean, median, mode, and standard deviation for every appropriate numerical column.
4. Parse dates correctly and plot **monthly and quarterly sales trends** using line charts, if the dataset has a usable date and sales field.
5. Analyze customer demographics: age-group distribution and gender breakdown when those fields exist. If unavailable, explain the limitation and do not invent them.
6. Identify the top 10 best-selling products. Clearly state whether “best-selling” is based on quantity, revenue, or another available measure.
7. Plot revenue by product category using a bar chart when category and revenue fields exist.
8. Create a correlation heatmap for suitable numerical variables.
9. Include at least one additional visualization that reveals a potentially non-obvious insight.
10. Add a markdown observation directly after each important chart. Observations must reflect the actual plotted data.
11. End with at least **three specific, actionable business recommendations** supported by the analysis.
12. Save useful figures under `outputs/figures/` and any cleaned dataset under `data/processed/`, when appropriate.

### Dataset handling

Use a suitable retail sales dataset, such as a retail/superstore sales dataset from Kaggle or another reputable source. Document the exact dataset name, source URL, license/usage notes if available, expected filename, and required columns. Do not claim that optional columns exist unless confirmed. If a chosen dataset lacks age, gender, or product names, either find a dataset that supports the required checklist or explicitly document the unavailable fields and request a suitable dataset rather than inventing data.

## 6. Project 2 — Customer Segmentation Analysis

**Folder:** `DataAnalytics-L1-CustomerSegmentation/`  
**Notebook:** `notebooks/customer_segmentation.ipynb`  
**Objective:** Group customers by purchasing behavior and translate cluster profiles into marketing actions.

### Required implementation checklist

1. Load and inspect the dataset; report dimensions, types, missing values, and inconsistent data.
2. Clean invalid records and missing values with documented reasoning.
3. Calculate or derive average purchase value, purchase frequency, and customer lifetime value when the dataset supports them. Define each metric clearly. If lifetime value cannot be reliably calculated, explain the limitation and use a clearly labelled proxy only if justified.
4. Select 2–3 meaningful behavioral features, preferably Recency, Frequency, and Monetary (RFM), when the transaction data supports them.
5. Standardize selected numeric features with `StandardScaler` before clustering.
6. Use the Elbow Method to inspect within-cluster sum of squares across a reasonable range of K values.
7. Fit K-Means with a justified K and a fixed random seed.
8. Visualize clusters with at least two scatter plots using different feature combinations. Label axes and cluster legends.
9. Profile each cluster using mean feature values and describe the segment in plain language, supported by calculated values.
10. Create a bar chart showing customer count per cluster.
11. Recommend at least one targeted marketing action for each segment and explain the evidence behind it.
12. Save relevant charts and a processed customer-level segmentation output, if appropriate.

### Dataset handling

Prefer a transaction-level e-commerce/online retail dataset from a reputable source, such as the UCI Online Retail dataset or an equivalent dataset with customer IDs, transaction dates, quantities, and prices. Document source, expected file name, required columns, and any preprocessing assumptions. Never treat an invoice row as a unique customer without aggregation.

## 7. Project 3 — Predicting House Prices with Linear Regression

**Folder:** `DataAnalytics-L2-HousePricePrediction/`  
**Notebook:** `notebooks/house_price_prediction.ipynb`  
**Objective:** Build and evaluate a regression model that predicts house prices from relevant property features.

### Required implementation checklist

1. Load the dataset and inspect shape, types, missing values, descriptive statistics, and the target price distribution.
2. Explain feature selection in a markdown cell, including why each selected feature may help predict price.
3. Handle missing values with an appropriate documented strategy.
4. Encode categorical predictors using One-Hot Encoding. Use a preprocessing pipeline to avoid data leakage where feasible.
5. Plot a correlation heatmap for appropriate numeric features and discuss relationships with the target.
6. Split data into training and test sets using an **80/20 split**. Set a reproducible random seed.
7. Train a `LinearRegression` model from scikit-learn.
8. Evaluate on the held-out test set using **MSE, RMSE, and R²**. Explain what each metric means and avoid implying that a good score proves causality.
9. Plot actual versus predicted prices with a reference diagonal.
10. Plot residuals and discuss whether patterns suggest model limitations.
11. Analyze coefficients carefully, accounting for feature scaling/encoding and the baseline category for one-hot features.
12. **Bonus:** compare Linear Regression with Ridge or Lasso using a leakage-safe preprocessing pipeline and comparable evaluation data.

### Dataset handling

Use a reputable house-price dataset, such as Ames Housing or another dataset suitable for regression. Record the exact source, target column, selected features, and preprocessing decisions. If a dataset is not present, provide instructions to download and place it in the project’s `data/raw/` folder. Do not generate fake house prices.

## 8. README Requirements

Each project README must include:

- Project title and internship task level.
- Problem statement and objective.
- Dataset name, source link, file placement instructions, and key columns.
- Tools and libraries used.
- Folder structure.
- Setup and run instructions.
- Analysis/modeling workflow.
- Main charts and findings, based only on executed results.
- Limitations and assumptions.
- How to reproduce results.
- Screenshots or figure links when available.

The root `README.md` must introduce the internship portfolio, list the three projects, link to their folders, provide environment setup instructions, and include a completion checklist.

## 9. Quality, Validation, and Acceptance Criteria

A project is complete only when its notebook is executable from top to bottom with the documented dataset and environment.

- [ ] Repository root name is exactly `OIBSIP`.
- [ ] All three project folders use the exact names specified in Section 3.
- [ ] Each project has a notebook and README.
- [ ] Every task checklist item is implemented or explicitly marked as unavailable because the dataset lacks a required field, with no fabricated substitute.
- [ ] Notebook outputs are produced by real computations; no placeholder metric values.
- [ ] Charts have titles, axis labels, legends where needed, and readable layouts.
- [ ] Markdown interpretations follow the charts and reflect actual outputs.
- [ ] Data leakage is avoided in the house-price model.
- [ ] Clustering is performed on standardized features and explained in business terms.
- [ ] Dependencies are listed in `requirements.txt`.
- [ ] Relative paths work on another computer.
- [ ] Errors from missing datasets are actionable and explain the expected path.
- [ ] No secrets or unnecessary large files are committed.
- [ ] Final summary reports files created, validation performed, and any remaining user actions.

## 10. GitHub and Internship Submission Requirements

Follow the internship task list:

1. Use a single public GitHub repository named **`OIBSIP`**.
2. Keep each task in its required folder, with source notebooks, README, and relevant screenshots/output files.
3. Push the completed work to GitHub.
4. Record a screen-demo video for each task. The first 2 seconds must show a static title card or overlay with the intern’s full name, assigned track, and task title. Then show the notebook/report functioning end-to-end.
5. Publish the demo on LinkedIn, tag Oasis Infobyte, and include `#oasisinfobyte` plus relevant domain hashtags.
6. Leave substantive comments on at least two other interns’ demo videos.
7. Submit the repository link through the official task submission form shared by the internship.
8. Do not assume submission is complete merely because the code is generated locally.

Do not invent the intern’s full name, LinkedIn URL, GitHub username, submission form URL, or internship-specific credentials. Use clearly labelled placeholders in templates where needed.

## 11. Instructions to Claude Code

Work incrementally and inspect the existing workspace before editing.

1. Confirm the current directory and inspect existing files before creating or changing anything.
2. Create the repository structure and three notebooks with meaningful markdown and code cells.
3. Prefer a dataset that satisfies each task’s requirements. Record the source and schema. If the required dataset is not available, create a robust loader and clear acquisition instructions; do not fabricate data or insights.
4. Implement each notebook in logical sections, with validation and visualizations.
5. Create the project READMEs, root README, `.gitignore`, and `requirements.txt`.
6. Run syntax/import checks and execute notebooks if the environment and datasets permit. If execution is not possible, state exactly what was not verified and why.
7. Review generated charts and notebook outputs for readability and correctness.
8. Finish with a concise report of files created, tests/execution performed, dataset setup still needed, and exact commands for running the projects.
9. Do not create or push commits, publish LinkedIn posts, or submit forms unless the user explicitly asks for those external actions.
10. Preserve existing user work and never overwrite unrelated files.

## 12. Definition of Done

The deliverable is a clean, reproducible GitHub portfolio containing three fully documented Data Analytics internship projects: Retail Sales EDA, Customer Segmentation, and House Price Prediction with Linear Regression. All findings must be traceable to actual dataset calculations, and any unavailable checklist item must be transparently documented.
