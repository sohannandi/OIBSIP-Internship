# Level 1 — Exploratory Data Analysis on Retail Sales

**Internship task:** Oasis Infobyte — Data Analytics, Level 1 (Task 1 of 3)
**Notebook:** [`notebooks/retail_sales_eda.ipynb`](notebooks/retail_sales_eda.ipynb)

---

## Problem statement and objective

A retail business receives a year of transaction records but has no systematic view of *when* it sells, *what* it sells, or
*how much each customer spends per transaction*. Pricing, stock planning and campaign timing all depend on those three
questions, and each of them needs a number rather than an opinion.

The objective of this project is to run a full exploratory analysis over the transaction data and end with a small set of
**specific, evidence-linked business recommendations** that a merchandising or marketing team could act on this quarter.

---

## Dataset

| | |
|---|---|
| **Name** | Retail Sales Dataset (1,000 transactions) |
| **Source** | `https://raw.githubusercontent.com/arehmanaziz/100DaysOfBytewise/main/task-10/datasets/retail_sales_dataset.csv` |
| **Expected filename** | `retail_sales.csv` |
| **Placement** | `DataAnalytics-L1-EDARetailSales/data/raw/retail_sales.csv` |
| **Size / shape** | 1,000 rows × 9 columns |
| **License / usage** | Public GitHub raw file, no credentials required. Verify the upstream licence before any commercial redistribution. |

### Key columns

| Column | Type | Meaning |
|---|---|---|
| `Transaction ID` | int | Unique transaction identifier (1–1000) |
| `Date` | str | Transaction date, `YYYY-MM-DD` |
| `Customer ID` | str | `CUST001`…`CUST1000` — one transaction per customer in this file |
| `Gender` | str | `Male` / `Female` |
| `Age` | int | Customer age, 18–64 |
| `Product Category` | str | `Beauty` / `Clothing` / `Electronics` |
| `Quantity` | int | Units sold, 1–4 |
| `Price per Unit` | int | Unit price, 25–500 |
| `Total Amount` | int | `Quantity × Price per Unit` |

### If the file is missing

The notebook raises a `FileNotFoundError` that names the exact expected path and the download URL. Place the file at
`data/raw/retail_sales.csv` and re-run — nothing else needs to change.

---

## Tools and libraries

Python 3.10+ · `pandas` · `numpy` · `matplotlib` · `seaborn` · `jupyter`

---

## Folder structure

```text
DataAnalytics-L1-EDARetailSales/
├── README.md
├── notebooks/
│   └── retail_sales_eda.ipynb      # the deliverable
├── data/
│   ├── raw/
│   │   └── retail_sales.csv        # input dataset (not committed if large)
│   └── processed/
│       └── retail_sales_cleaned.csv
├── outputs/
│   └── figures/                    # 5 PNG charts
│       ├── monthly_quarterly_trends.png
│       ├── demographics.png
│       ├── revenue_by_category.png
│       ├── correlation_heatmap.png
│       └── basket_by_age_gender.png
└── reports/
    └── eda_summary.md              # written EDA report
```

---

## Setup and run

```bash
# from the repository root
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cd DataAnalytics-L1-EDARetailSales
jupyter lab notebooks/retail_sales_eda.ipynb
```

Run all cells top to bottom (`Run All`). The notebook is executed and committed with its outputs, so the charts and every
number below are reproducible from the committed file.

---

## Analysis workflow

1. **Load & inspect** — shape, columns, dtypes, null counts, first rows, categorical value counts.
2. **Validate** — duplicate rows, `Total Amount ≠ Quantity × Price per Unit`, non-positive quantities/prices/totals, ages
   outside 18–100.
3. **Clean** — parse `Date` to datetime, drop duplicates, drop non-positive/invalid rows, recompute `Total Amount` and add an
   independent `Revenue` column for cross-checking. Row counts before/after are printed at every step.
4. **Descriptive statistics** — mean, median, mode and standard deviation for **every** numeric column.
5. **Time series** — monthly and quarterly revenue totals, plotted as line charts, with the partial final period annotated.
6. **Demographics** — age-group distribution and gender breakdown.
7. **Best sellers** — ranked by quantity *and* by revenue, with the basis stated explicitly.
8. **Revenue by category** — bar chart.
9. **Correlation heatmap** — over the numeric variables, with the structural correlations called out.
10. **Extra visualisation** — average basket size by age group × gender (the non-obvious insight).
11. **Recommendations** — three, each citing the figure it rests on.
12. **Save** — cleaned CSV to `data/processed/`, all figures to `outputs/figures/`.

---

## Main charts and findings

All numbers below are the notebook's own executed output.

### 1. Monthly & quarterly trends — [`monthly_quarterly_trends.png`](outputs/figures/monthly_quarterly_trends.png)

Total revenue is **456,000** across 1,000 transactions. The best full month is **May 2023 at 53,150** (105 transactions) and the
weakest full month is **September 2023 at 23,620** (65 transactions) — a **2.25× spread** between the strongest and weakest
trading month. February (44,060) and December (44,690) form a second, smaller peak, so the pattern reads as a recurring seasonal
lift rather than a single spike.

Quarterly: **Q1 108,500 → Q2 123,735 → Q3 96,045 → Q4 126,190**. Q3 is the soft quarter and Q4 the strongest. The final data
point (2024-01, 1,530 from 2 transactions) is a **partial period** — the dataset ends on 2024-01-01 — and is annotated on the
chart so it is not misread as a collapse.

### 2. Demographics — [`demographics.png`](outputs/figures/demographics.png)

Age is spread fairly evenly: 46–55 is the largest band at **229 transactions**, 18–25 the smallest at **169**, with 26–35 (205),
36–45 (202) and 56–65 (195) in between. Gender is essentially a coin flip: **510 Female (51%) vs 490 Male (49%)**. Mean age is
41.4 (σ 13.7).

### 3. Best sellers — basis stated

This dataset exposes products at **category level only** (Beauty, Clothing, Electronics) with no product names, so a "top 10
products" list resolves to the three available categories. Ranking is shown on **both** bases, and the two disagree:

| Category | Units sold | Revenue |
|---|---:|---:|
| Electronics | 849 | **156,905** |
| Clothing | **894** | 155,580 |
| Beauty | 771 | 143,515 |

Clothing sells the most *units*; Electronics earns the most *revenue* on fewer units — i.e. Electronics carries a higher
average unit price. No finer-grained product ranking is possible from this file, and none is invented.

### 4. Revenue by category — [`revenue_by_category.png`](outputs/figures/revenue_by_category.png)

A remarkably flat split: **Electronics 34.4%**, **Clothing 34.1%**, **Beauty 31.5%** of 456,000. The top and bottom
categories are separated by only **13,390, or 2.9 percentage points**. Category mix is therefore not the lever on this dataset.

### 5. Correlation heatmap — [`correlation_heatmap.png`](outputs/figures/correlation_heatmap.png)

- `Price per Unit` ↔ `Total Amount` = **0.852** — structural, since `Total Amount = Quantity × Price per Unit`.
- `Revenue` ↔ `Total Amount` = **1.000** — the two are the same quantity, which confirms the recomputation is consistent.
- `Quantity` ↔ `Total Amount` = **0.374** — moderate; price varies far more than quantity (25–500 vs 1–4).
- `Age` is essentially uncorrelated with everything (**−0.061 to −0.024**), which rules age out as a revenue driver.
- No meaningful negative correlations appear, so there is no trade-off structure in this dataset.

### 6. Extra visualisation — basket size by age × gender — [`basket_by_age_gender.png`](outputs/figures/basket_by_age_gender.png)

Average basket value spans **398.02 to 515.09** across the ten age × gender cells — a **29% spread** — while transaction counts
stay nearly flat (age groups 169–229; gender 510 vs 490).

| Avg. basket | Female | Male |
|---|---:|---:|
| 18–25 | 487.28 | 512.27 |
| 26–35 | **515.09** | 442.50 |
| 36–45 | 436.78 | 475.11 |
| 46–55 | 447.98 | 430.73 |
| 56–65 | **398.02** | 426.26 |

The gender gap **reverses by age**: women lead in 26–35 (+72.59) and 46–55 (+17.25), men lead in 18–25, 36–45 and 56–65.
Age ordering is not monotonic, and 56–65 is the weakest band for *both* genders.

**This is the non-obvious insight of the analysis:** revenue is driven by *how much customers spend per transaction*, not by
*how many transactions they make* — which is why the recommendations target basket size rather than customer volume.

---

## Business recommendations

### 1. Re-price and re-bundle around unit price, not around volume
**Evidence:** `Price per Unit` correlates with `Total Amount` at **0.852**, far above `Quantity` at **0.374**; unit price spans
25–500 while quantity only spans 1–4. Electronics also beats Clothing on revenue (**156,905 vs 155,580**) despite selling
**fewer units (849 vs 894)**.
**Action:** run price and bundle tests on high-unit-price items rather than pushing more units. Specifically, bundle the
500-unit-price lines (all three categories contain them: 68 Beauty, 67 Electronics, 64 Clothing transactions) with low-price
items so the value of each transaction rises without discounting the high-margin units. A 10% price test on the top unit-price
band is worth more to revenue than a 10% volume increase at the current mix, because price moves basket value roughly 2.3× more
strongly than quantity does.

### 2. Plan stock and campaigns around the Q3 trough and the Q4 peak
**Evidence:** quarterly sales go **108,500 (Q1) → 123,735 (Q2) → 96,045 (Q3) → 126,190 (Q4)**. Q3 is **22.4% below Q2** and
**23.9% below Q4**, and the monthly trough (September, 23,620) is **55.6% below** the May peak (53,150) on only 65 vs 105
transactions.
**Action:** build inventory ahead of the Q4 lift, and run a targeted promotion in the July–September window rather than spreading
campaigns evenly across the year. Because the Q3 fall is driven as much by **fewer transactions (65–94/month) as by lower
basket value**, the Q3 campaign should be a traffic-and-volume play (bundle discounts, free-shipping threshold, win-back email)
rather than a pricing cut.

### 3. Grow basket value for the weakest demographic cells, and drop age as a targeting key
**Evidence:** average basket size spans **398.02 to 515.09** across age × gender cells (a **29% spread**), while transaction
counts are nearly flat (age groups **169–229**; gender **510 Female vs 490 Male**). **Age is uncorrelated with revenue
(−0.061 to −0.024)** and the gender gap **reverses by age**.
**Action:** target the two weakest cells — **Female 56–65 (398.02)** and **Male 56–65 (426.26)** — with threshold offers that
lift basket size (free shipping above a spend floor, a second-item incentive), and treat **age as a poor segmentation variable**
— the evidence says spend per transaction and unit price drive revenue, not who the customer is. A blanket demographic discount
is not justified by these numbers, since demographic *volume* differences are small next to the basket-size spread: the gender
split is **2% of transactions** and the age-group spread from 169 to 229 accounts for only **6%** of volume, versus a **29%**
spread in basket value.

---

## Limitations and assumptions

- **Products exist only at category level.** The file has 3 product categories and no product names, so "top 10 best-selling
  products" collapses to 3 rows. A product-level dataset (e.g. the Kaggle *Superstore* dataset) would be needed to produce a
  genuine top-10 product list; no product names are invented here.
- **One transaction per customer.** There are 1,000 unique `Customer ID`s for 1,000 transactions, so this file cannot support
  repeat-purchase, customer-lifetime or basket-vs-frequency analysis. Demographics are stated as transaction counts.
- **Synthetic-looking, flat data.** The 2023 monthly totals swing between 23,620 and 53,150 with no marketing, price or
  competitor context in the file, so seasonality cannot be attributed to a cause.
- **The final period is partial.** The last data point (2024-01-01, 2 transactions) must be excluded from any seasonal reading;
  it is annotated on the chart.
- **Correlation is not causation.** The heatmap describes associations within these 1,000 rows only.
- **Cleaning was a no-op on this file.** The validation step found 0 duplicate rows, 0 arithmetic inconsistencies, 0 non-positive
  values and 0 out-of-range ages, so the cleaning pipeline removed 0 rows. It is retained because it is the step that proves the
  data is clean, and it is what makes the results safe to quote.

---

## How to reproduce

1. Install the dependencies: `pip install -r requirements.txt` (from the repository root).
2. Place `retail_sales.csv` in `data/raw/`.
3. From `DataAnalytics-L1-EDARetailSales/`, run `jupyter lab notebooks/retail_sales_eda.ipynb` and select **Run All**.
4. The notebook re-generates `data/processed/retail_sales_cleaned.csv` and all five PNGs in `outputs/figures/`.

The notebook is fully deterministic — there is no sampling or random seed involved — so a re-run reproduces every number in
this README exactly.

---

## Outputs

| Artefact | Path |
|---|---|
| Notebook (executed) | [`notebooks/retail_sales_eda.ipynb`](notebooks/retail_sales_eda.ipynb) |
| Cleaned dataset | [`data/processed/retail_sales_cleaned.csv`](data/processed/retail_sales_cleaned.csv) |
| Monthly & quarterly trends | [`outputs/figures/monthly_quarterly_trends.png`](outputs/figures/monthly_quarterly_trends.png) |
| Demographics | [`outputs/figures/demographics.png`](outputs/figures/demographics.png) |
| Revenue by category | [`outputs/figures/revenue_by_category.png`](outputs/figures/revenue_by_category.png) |
| Correlation heatmap | [`outputs/figures/correlation_heatmap.png`](outputs/figures/correlation_heatmap.png) |
| Basket size by age × gender | [`outputs/figures/basket_by_age_gender.png`](outputs/figures/basket_by_age_gender.png) |
| Written EDA report | [`reports/eda_summary.md`](reports/eda_summary.md) |
