# OIBSIP — Oasis Infobyte Data Analytics Internship Projects

> **Intern:** _[Your Full Name]_ · **Track:** Data Analytics · **Organisation:** Oasis Infobyte
> _GitHub: [github.com/your-username/OIBSIP](https://github.com/your-username/OIBSIP) ·
> LinkedIn: [linkedin.com/in/your-linkedin-id](https://linkedin.com/in/your-linkedin-id)_
>
> _Replace the placeholders above with your own details before publishing._

Three complete data analytics projects built for the Oasis Infobyte Data Analytics internship: exploratory analysis, customer
segmentation, and predictive regression. Every chart, table and metric in this repository was produced by running the
committed notebooks against real datasets — nothing is hand-written or estimated.

---

## The projects

| # | Project | Task level | What it does |
|---|---|---|---|
| 1 | **[Retail Sales EDA](DataAnalytics-L1-EDARetailSales/)** | Level 1 | Loads 1,000 retail transactions, validates and cleans them, and works through descriptive statistics, monthly/quarterly trends, demographics, best sellers, revenue by category, a correlation heatmap and an extra basket-size visualisation — ending in three evidence-linked business recommendations. |
| 2 | **[Customer Segmentation](DataAnalytics-L1-CustomerSegmentation/)** | Level 1 | Aggregates 541,909 UCI Online Retail line items to 4,338 customers, derives APV / purchase frequency / a labelled CLV proxy, standardises RFM features, chooses K with the elbow method and silhouette scores, fits K-Means (`random_state=42`) and turns four segments into four targeted marketing actions. |
| 3 | **[House Price Prediction](DataAnalytics-L2-HousePricePrediction/)** | Level 2 | Fits a leakage-safe `LinearRegression` pipeline on the Ames Housing data (2,930 houses, 20 features → 68 encoded columns), evaluates on a held-out 20% with MSE / RMSE / R², analyses residuals, interprets coefficients against their baseline categories, and benchmarks Ridge and Lasso as a bonus. |

**Headline results**

- **Retail Sales EDA** — revenue is driven by *spend per transaction*, not transaction count: average basket value varies by
  **29%** across age × gender cells while volume varies by only **2–6%**.
- **Customer Segmentation** — **5.0% of customers hold 47.7% of revenue**; the k=2 silhouette artefact is shown rather than
  hidden, and K=4 is chosen with the alternatives printed.
- **House Price Prediction** — **R² 0.8684**, **RMSE 32,479** on 586 unseen houses, versus a mean-only baseline of R² −0.0153 /
  RMSE 90,222. **Location outweighs size**: `Neighborhood=GrnHill` is worth **+115,985.78** against the baseline neighbourhood,
  more than 3× a standard deviation of living space.

---

## Repository structure

```text
OIBSIP/
├── README.md
├── requirements.txt
├── .gitignore
├── OICSIP.md                        # the project brief this repo implements
├── DataAnalytics-L1-EDARetailSales/
│   ├── README.md
│   ├── notebooks/retail_sales_eda.ipynb
│   ├── data/raw/ · data/processed/
│   ├── outputs/figures/
│   └── reports/eda_summary.md
├── DataAnalytics-L1-CustomerSegmentation/
│   ├── README.md
│   ├── notebooks/customer_segmentation.ipynb
│   ├── data/raw/ · data/processed/
│   └── outputs/figures/
├── DataAnalytics-L2-HousePricePrediction/
│   ├── README.md
│   ├── notebooks/house_price_prediction.ipynb
│   ├── data/raw/ · data/processed/
│   └── outputs/figures/
└── scripts/                         # notebook generators (build_p1/p2/p3.py)
```

Each project folder keeps its raw data, processed data, figures and notebook exactly as the internship task list requires.

---

## Environment setup

**Requires Python 3.10 or later.**

```bash
git clone https://github.com/your-username/OIBSIP.git
cd OIBSIP

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Then open Jupyter:

```bash
jupyter lab
```

### Datasets

All three raw datasets are already committed in each project's `data/raw/` folder, so the notebooks run immediately after
cloning. If you need to re-download them:

| Project | File | Source |
|---|---|---|
| Retail Sales EDA | `data/raw/retail_sales.csv` | [raw.githubusercontent.com/arehmanaziz/100DaysOfBytewise](https://raw.githubusercontent.com/arehmanaziz/100DaysOfBytewise/main/task-10/datasets/retail_sales_dataset.csv) |
| Customer Segmentation | `data/raw/online_retail.csv` | [UCI ML Repository — Online Retail (352)](https://archive.ics.uci.edu/static/public/352/online+retail.zip) |
| House Price Prediction | `data/raw/AmesHousing.csv` | [raw.githubusercontent.com/STATCowboy/pbidataflowstalk](https://raw.githubusercontent.com/STATCowboy/pbidataflowstalk/master/AmesHousing.csv) |

If a file is missing, the notebook stops with a `FileNotFoundError` that names the exact expected path and the download URL.

---

## Running the projects

Each notebook is committed **already executed**, so you can read the charts and outputs without running anything. To reproduce
them from scratch, open the notebook and use **Run All**:

```bash
cd DataAnalytics-L1-EDARetailSales        && jupyter lab notebooks/retail_sales_eda.ipynb
cd DataAnalytics-L1-CustomerSegmentation   && jupyter lab notebooks/customer_segmentation.ipynb
cd DataAnalytics-L2-HousePricePrediction   && jupyter lab notebooks/house_price_prediction.ipynb
```

All three are deterministic — `random_state=42` for the train/test split and for K-Means — so a re-run reproduces every number
exactly.

---

## Figures

| | |
|---|---|
| **Retail Sales EDA** | [monthly & quarterly trends](DataAnalytics-L1-EDARetailSales/outputs/figures/monthly_quarterly_trends.png) · [demographics](DataAnalytics-L1-EDARetailSales/outputs/figures/demographics.png) · [revenue by category](DataAnalytics-L1-EDARetailSales/outputs/figures/revenue_by_category.png) · [correlation heatmap](DataAnalytics-L1-EDARetailSales/outputs/figures/correlation_heatmap.png) · [basket size by age × gender](DataAnalytics-L1-EDARetailSales/outputs/figures/basket_by_age_gender.png) |
| **Customer Segmentation** | [elbow method](DataAnalytics-L1-CustomerSegmentation/outputs/figures/elbow_method.png) · [cluster scatter plots](DataAnalytics-L1-CustomerSegmentation/outputs/figures/cluster_scatter.png) · [customers per cluster](DataAnalytics-L1-CustomerSegmentation/outputs/figures/customers_per_cluster.png) · [customers per segment](DataAnalytics-L1-CustomerSegmentation/outputs/figures/customers_per_segment.png) |
| **House Price Prediction** | [target distribution](DataAnalytics-L2-HousePricePrediction/outputs/figures/target_distribution.png) · [correlation heatmap](DataAnalytics-L2-HousePricePrediction/outputs/figures/correlation_heatmap.png) · [actual vs predicted](DataAnalytics-L2-HousePricePrediction/outputs/figures/actual_vs_predicted.png) · [residuals](DataAnalytics-L2-HousePricePrediction/outputs/figures/residuals.png) · [coefficients](DataAnalytics-L2-HousePricePrediction/outputs/figures/coefficients.png) · [model comparison](DataAnalytics-L2-HousePricePrediction/outputs/figures/model_comparison.png) |

---

## Completion checklist

### Project 1 — Retail Sales EDA (`DataAnalytics-L1-EDARetailSales/`)
- [x] Dataset loaded; shape, columns, dtypes and nulls inspected
- [x] Missing / invalid values checked and the cleaning decisions documented
- [x] Mean, median, mode and standard deviation computed for every numeric column
- [x] Dates parsed; **monthly and quarterly** sales trends plotted as line charts
- [x] Age-group distribution and gender breakdown analysed
- [x] Top best sellers identified, with the ranking basis stated explicitly
- [x] Revenue by product category plotted as a bar chart
- [x] Correlation heatmap created for the numeric variables
- [x] Extra visualisation added (average basket size by age × gender)
- [x] Markdown observation after every chart, reflecting the plotted data
- [x] Three specific, actionable business recommendations, each citing its evidence
- [x] Figures saved to `outputs/figures/`, cleaned data to `data/processed/`
- [x] README with dataset, setup, workflow, findings, limitations and figure links

### Project 2 — Customer Segmentation (`DataAnalytics-L1-CustomerSegmentation/`)
- [x] Dataset loaded and inspected (dimensions, types, nulls, inconsistencies)
- [x] Invalid records and missing values cleaned, with the row counts logged
- [x] APV, purchase frequency and a clearly-labelled CLV proxy derived and defined
- [x] RFM features selected with a stated rationale
- [x] `StandardScaler` applied before clustering
- [x] Elbow Method run across k = 2…10
- [x] K-Means fitted with a justified K and a fixed seed (`random_state=42`)
- [x] Two scatter plots with **different** feature pairs, labelled axes and legends
- [x] Every cluster profiled in plain language with calculated means and medians
- [x] Bar chart of customer count per cluster
- [x] At least one targeted marketing action per segment, with the evidence cited
- [x] Charts saved and a customer-level segmentation output written
- [x] Invoices are never treated as customers — aggregation happens before clustering
- [x] README with dataset, setup, workflow, findings, limitations and figure links

### Project 3 — House Price Prediction (`DataAnalytics-L2-HousePricePrediction/`)
- [x] Dataset loaded; shape, types, nulls, descriptive statistics and target distribution inspected
- [x] Feature selection explained in a markdown cell, with a reason per feature
- [x] Missing values handled with a documented, per-column strategy
- [x] One-hot encoding performed **inside a pipeline** to avoid leakage
- [x] Correlation heatmap plotted and discussed against the target
- [x] **80/20 split** with a reproducible seed (`random_state=42`)
- [x] `LinearRegression` trained from scikit-learn
- [x] Evaluated with **MSE, RMSE and R²**, each explained, with no causal claim
- [x] Actual-vs-predicted plotted with the y = x reference diagonal
- [x] Residuals plotted (3 panels) and the patterns discussed
- [x] Coefficients analysed accounting for scaling, encoding and baseline category
- [x] **Bonus:** Ridge and Lasso compared through the same leakage-safe pipeline on the same test set
- [x] README with dataset, setup, workflow, findings, limitations and figure links

### Repository
- [x] Repository named exactly `OIBSIP`, single repository for all three tasks
- [x] Exact project folder names as required by the task list
- [x] Root `README.md`, `requirements.txt` and `.gitignore` present
- [x] Relative paths and `pathlib` throughout — no machine-specific absolute paths
- [x] Missing-dataset errors are actionable and name the expected path
- [x] Random seeds set wherever results could vary
- [x] All three notebooks executed end-to-end with zero cell errors
- [x] No secrets, no credentials, no API keys, no large generated folders committed

---

## Repository conventions

- **Notebooks are the deliverable.** `scripts/build_p1.py`, `build_p2.py` and `build_p3.py` are the generators that assemble and
  execute each notebook; the `.ipynb` files are committed with their outputs. To regenerate a notebook, run
  `python scripts/build_p1.py` (or `p2` / `p3`) from the repository root.
- **No fabricated results.** Every table, chart, metric and written conclusion is calculated from the loaded data. Where a
  metric could mislead — the CLV proxy, the k=2 silhouette, the `GarageType` baseline — the notebook says so explicitly.
- **Absolute paths are avoided.** Notebooks locate their project root by walking up from the working directory until they find
  `data/raw/`, so they run from any checkout location.

---

## Acknowledgement

Built for the **Oasis Infobyte Data Analytics internship**. Datasets are public: *Retail Sales Dataset*, *UCI Online Retail*
(CC BY 4.0) and the *Ames Housing* data.
