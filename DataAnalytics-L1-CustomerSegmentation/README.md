# Level 1 — Customer Segmentation with K-Means

**Internship task:** Oasis Infobyte — Data Analytics, Level 1 (Task 2 of 3)
**Notebook:** [`notebooks/customer_segmentation.ipynb`](notebooks/customer_segmentation.ipynb)

---

## Problem statement and objective

An online retailer has 4,338 identified customers in a year of transaction history, and no way to tell them apart beyond a single
row in a spreadsheet. Treating them as one undifferentiated list means every campaign is the same campaign — which is how
budget gets spent on customers who will never buy again while the ones who generate half the revenue get the same treatment as
the ones who generated a single order a year ago.

The objective is to **group customers by purchasing behaviour**, describe each group in plain language using calculated values,
and turn each group into a **specific, targeted marketing action**.

---

## Dataset

| | |
|---|---|
| **Name** | Online Retail (UCI Machine Learning Repository, dataset 352) |
| **Source** | `https://archive.ics.uci.edu/static/public/352/online+retail.zip` → `Online Retail.xlsx` |
| **Expected filename** | `online_retail.csv` |
| **Placement** | `DataAnalytics-L1-CustomerSegmentation/data/raw/online_retail.csv` |
| **Shape** | 541,909 line items × 8 columns |
| **Period covered** | 2010-12-01 → 2011-12-09 (374 days) |
| **License / usage** | CC BY 4.0, public. No credentials or API key required. |

The UCI download is a `.zip` containing an `.xlsx` workbook. This project stores the **flattened CSV** form
(`online_retail.csv`) so the notebook needs only `pandas`; `openpyxl` is listed in `requirements.txt` for anyone who wants to
regenerate the CSV from the original workbook.

### Key columns

| Column | Type | Meaning |
|---|---|---|
| `InvoiceNo` | str | Invoice number. `C` prefix = cancellation. |
| `StockCode` | str | Product code |
| `Description` | str | Product description |
| `Quantity` | int | Units on the line (negative = return) |
| `InvoiceDate` | str | Timestamp of the invoice |
| `UnitPrice` | float | Price per unit (negative = return) |
| `CustomerID` | float | Customer identifier. **Blank for 135,080 guest rows.** |
| `Country` | str | Customer country, 38 countries |

### If the file is missing

The notebook raises a `FileNotFoundError` naming the exact expected path and the UCI download URL. Place the file at
`data/raw/online_retail.csv` and re-run.

---

## Tools and libraries

Python 3.10+ · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `jupyter`

---

## Folder structure

```text
DataAnalytics-L1-CustomerSegmentation/
├── README.md
├── notebooks/
│   └── customer_segmentation.ipynb   # the deliverable
├── data/
│   ├── raw/
│   │   └── online_retail.csv          # input dataset
│   └── processed/
│       └── customer_segmentation.csv  # 4,338 customers × 15 columns
└── outputs/
    └── figures/
        ├── elbow_method.png
        ├── cluster_scatter.png        # 2 panels: Recency×Monetary, Frequency×Monetary
        ├── customers_per_cluster.png
        └── customers_per_segment.png
```

---

## Setup and run

```bash
# from the repository root
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cd DataAnalytics-L1-CustomerSegmentation
jupyter lab notebooks/customer_segmentation.ipynb
```

Run all cells top to bottom (`Run All`). The notebook is committed executed, so every chart and number below comes from a real
run.

---

## Analysis workflow

1. **Load & inspect** — dimensions, dtypes, null counts, distinct invoices / stock codes / customers / countries, sample rows.
2. **Clean** — a logged pipeline with row counts printed at each step: parse dates → keep identified customers → drop
   cancellation invoices → drop non-positive quantity/price.
3. **Aggregate to customer level** — one row per `CustomerID`, *never* one row per invoice. Derives recency, purchase frequency,
   monetary value, average purchase value, and the CLV proxy.
4. **Standardise** — `StandardScaler` on Recency / Frequency / Monetary; the notebook prints the pre-scaling means and standard
   deviations and asserts the post-scaling means are ~0 and standard deviations ~1.
5. **Choose K** — Elbow Method (WCSS for k = 2…10) plus a silhouette table for the same range, and the alternative partitions
   printed so the K choice can be judged rather than asserted.
6. **Fit K-Means** — `K=4`, `random_state=42`, `n_init=10`.
7. **Visualise** — two scatter plots with different feature pairs, plus a bar chart of customers per cluster.
8. **Profile & act** — mean and median values per cluster, a business name per segment, and one marketing action per segment.
9. **Save** — the customer-level segmentation table and all charts.

---

## Metric definitions

| Metric | Definition (as implemented) |
|---|---|
| **Recency** | Days from the customer's last invoice to the reference date (the day after the last invoice in the file, 2011-12-10). |
| **Purchase Frequency** | Number of **distinct invoices** the customer placed in the 374-day window. |
| **Monetary (Total Revenue)** | Sum of `Quantity × UnitPrice` over the customer's retained line items. |
| **Average Purchase Value (APV)** | `Total Revenue ÷ Purchase Frequency` — the value of a typical order. |
| **Annual Purchase Frequency** | `Purchase Frequency ÷ (374 / 365)` — order count scaled to a per-year rate using the **common** observation window. |
| **CLV (proxy)** | `APV × Annual Purchase Frequency × 2 years` — see the caveat below. |

**The CLV caveat, stated rather than hidden.** A true CLV needs a retention/churn model, a margin assumption and multi-year
purchase history. This dataset covers roughly one trading year and contains no churn labels, so a true CLV **cannot be reliably
calculated here**. The proxy above uses an **assumed 2-year average relationship lifespan**, which is a documented business
assumption and *not* a quantity derived from the data. The notebook also prints a consistency check: because the lifespan
assumption is a single constant, `CLV_proxy` is exactly **1.9519 × Total_Revenue** for every customer — a monotone rescaling of
monetary value that therefore carries **no information beyond the RFM variables**. That is why the CLV proxy is used for
reporting and budget sizing, and **not** as a clustering feature.

---

## Main charts and findings

All numbers below are the notebook's own executed output.

### 1. Cleaning funnel

| Stage | Line items |
|---|---:|
| Raw | 541,909 |
| After dropping unparseable dates | 541,909 |
| After keeping identified customers | 406,829 |
| After removing cancellation invoices | 397,924 |
| After removing non-positive quantity/price | 397,884 |

**4,338 customers** retained, **18,532 distinct invoices**, **8,911,407.90** total clean revenue.

### 2. Customer economics

| Metric | mean | median | max |
|---|---:|---:|---:|
| Recency (days) | 92.54 | 51 | 374 |
| Purchase Frequency | 4.27 | 2 | 209 |
| Total Revenue | 2,054.27 | 674.48 | 280,206.02 |
| Average Purchase Value | 419.17 | 293.90 | 84,236.25 |
| CLV proxy | 4,009.66 | 1,316.51 | 546,926.19 |

The distributions are heavily right-skewed: the median customer places only **2 orders** and **one in four customers buys exactly
once**. The mean order value (419.17) overstates a typical customer's order (293.90) by roughly 40%. The median customer is
active for only **93.5 days** of the 374-day window, and the 25th percentile is **1 day** — most of this base is effectively
one-time buyers.

### 3. Choosing K — [`elbow_method.png`](outputs/figures/elbow_method.png)

WCSS by k: 2 → 9,012.64 · 3 → 5,439.30 · **4 → 4,092.14** · 5 → 3,118.28 · 6 → 2,472.57 · 7 → 2,022.41 · 8 → 1,742.37 ·
9 → 1,445.63 · 10 → 1,303.39. The percentage reduction falls 39.6% (2→3), 24.8% (3→4), 23.8% (4→5), 20.7%, 18.2%, 13.8%…

The silhouette scores are **0.8958 (k=2)**, 0.5942 (k=3), **0.6162 (k=4)**, 0.6165 (k=5), 0.5963, 0.5165, 0.5175, 0.4777, 0.4271.

**The k=2 peak is a small-k artefact and the notebook says so explicitly.** At k=2 the model is nothing but *whales vs everybody
else* — 4,312 of 4,338 customers land in a single cluster. The alternative partitions are printed to make this visible:

- **k=2** — 4,312 customers (74.9% of revenue) vs 26 customers (25.1% of revenue). Useless for marketing.
- **k=3** — adds the dormant group but leaves the 204 high-frequency customers *inside* the active majority.
- **k=5** — rejected: **+0.0003** silhouette for the cost of splitting the whale group into 8 and 6 customers.

**K=4 is the chosen model**: the smallest partition that separates every group that behaves differently enough to be *marketed*
differently. Final model: WCSS **4,092.14**, silhouette **0.6162**, cluster sizes **{0: 3,054, 1: 1,067, 2: 13, 3: 204}**.

### 4. Segment profile — [`cluster_scatter.png`](outputs/figures/cluster_scatter.png)

| Segment | Customers | % base | Mean recency | Mean orders | Mean order value | Mean spend | % revenue | % orders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Whales** | 13 | 0.3% | 7.4 | 82.5 | 8,570.73 | 127,338.31 | 18.6% | 5.8% |
| **Loyal High-Value** | 204 | 4.7% | 15.5 | 22.3 | 1,080.41 | 12,709.09 | 29.1% | 24.6% |
| **Regular Customers** | 3,054 | 70.4% | 43.7 | 3.68 | 376.76 | 1,359.05 | 46.6% | 60.7% |
| **At-Risk / Lapsed** | 1,067 | 24.6% | 248.1 | 1.55 | 314.81 | 480.62 | 5.8% | 8.9% |

**The headline:** the Whales and Loyal High-Value segments together are **217 customers (5.0% of the base) but 47.7% of
revenue and 30.4% of all orders**. Half the business sits with one twentieth of the customers.

The two scatter plots use **different feature pairs** — Recency × Monetary and Frequency × Monetary — and both are on **log
scales** because the data is extremely skewed (spend 3.75 → 280,206; orders 1 → 209). Recency alone would not separate the
Whales from the Loyal group (both are recent), and monetary alone would not separate them either (both are high-value) — it is
the *combination* of high frequency and high value that identifies them. On linear axes the 13 whale points would be
indistinguishable from the 204 loyal customers.

### 5. Customers per cluster — [`customers_per_cluster.png`](outputs/figures/customers_per_cluster.png) · [`customers_per_segment.png`](outputs/figures/customers_per_segment.png)

3,054 (70.4%) · 1,067 (24.6%) · 204 (4.7%) · 13 (0.3%). The class imbalance is extreme, which is the reason the scatter plots
are log-scaled.

---

## Targeted marketing actions

Each action cites the measured value it rests on.

**1. Whales — 13 customers (0.3% of the base, 18.6% of revenue)**
*Evidence: mean recency 7.4 days, mean 82.5 orders, mean spend 127,338, mean order value 8,570.73.*
**Action:** hand-manage these accounts — named account manager, early access to new lines, a service-level promise on delivery.
Do **not** put them on discount promotions: with a mean order value of 8,570.73 against a portfolio mean of 419.17, even a 5%
discount on their orders gives away about **82,770** of revenue — more than the entire lifetime spend of **172 average lapsed
customers**. Their **mean CLV proxy of 248,548.05 per customer** also means a single one of them is worth roughly **half the
entire annual revenue of the whole At-Risk / Lapsed segment** (512,818.85 across 1,067 customers).

**2. Loyal High-Value — 204 customers (4.7% of the base, 29.1% of revenue, 24.6% of orders)**
*Evidence: mean recency 15.5 days, mean 22.3 orders, mean spend 12,709, mean order value 1,080.41.*
**Action:** convert them into a subscription or replenishment programme. They already buy roughly every **17 days** (22.3 orders
across the 374-day window) and their median order value of 8,037.73 is **12× the portfolio median order value of 674.48** — the
profile is almost perfectly suited to an automatic reorder cycle. Concentrate the loyalty-reward budget here.

**3. Regular Customers — 3,054 customers (70.4% of the base, 46.6% of revenue, 60.7% of orders)**
*Evidence: mean recency 43.7 days, mean 3.68 orders, mean spend 1,359 (median 829), mean order value 376.76.*
**Action:** this is the segment with the most room to grow, because it already generates **46.6% of revenue** but each customer
is worth only about 1,359. Run a **second-purchase activation** campaign here, not a blanket one: with a median of only 3 orders
per customer, moving them from 3 to 4 orders is worth more incremental revenue than any reactivation effort. Targeted cross-sell
of adjacent categories is the right mechanism, since their order value (376.76) sits between the lapsed group and the loyal
group, meaning they have room to trade up.

**4. At-Risk / Lapsed — 1,067 customers (24.6% of the base, 5.8% of revenue, 8.9% of orders)**
*Evidence: mean recency 248.1 days (median 243 — more than eight months), mean 1.55 orders (median 1), mean spend 480.62,
mean order value 314.81.*
**Action:** run a low-cost **win-back** email/SMS campaign with a time-boxed incentive, and set a hard budget cap. The economics
are unattractive on a per-customer basis: 1,067 people generating only 5.8% of revenue, whose median spend is just 310.47 and
median order value 314.81. A 10% incentive costs about **31 per activated customer**; if every one of the 1,067 bought one more
order at their median spend, the incentive would cost **33,127** in total in exchange for at most **331,271** of recovered revenue
— a 10:1 gross ratio *before* margin and before the reputational cost of discounting to customers who were never going to return.
Prioritise within this segment by **most recent first** (the 180–240 day band) and let the 300+ day tail lapse: they are the
least likely to return and should be suppressed to protect the email channel's engagement for everyone else.

**Cross-segment insight:** the highest-return use of a marketing budget is (a) protecting the high-value segments, and (b) moving
the *Regular* segment up one order per year — not (c) spending heavily to win back the lapsed.

---

## Limitations and assumptions

- **CLV is a proxy, not a real CLV.** No churn labels, no margin data, one year of history. The 2-year lifespan is an explicit
  assumption, and the notebook proves the resulting figure is a rescaling of monetary value rather than pretending it adds
  information.
- **The customer base is overwhelmingly one-time buyers.** With a median of 2 orders and a 25th percentile of 1 order, "loyalty"
  segments are thin: the whole Loyal High-Value segment is 204 customers. Conclusions about retention should not be
  over-generalised from this sample.
- **The 374-day window is short for recency work.** A customer last seen 374 days ago may have churned, gone on holiday, or
  simply bought once. The At-Risk segment is an *opportunity* list, not a churn prediction.
- **13 customers is a very small group.** The Whales segment drives 18.6% of revenue, so any single whale leaving would move
  the revenue figures noticeably. Treat that segment as "hand-managed, do not automate", not as a statistically stable cluster.
- **Outlier sensitivity.** K-Means is not robust to extreme points, and a customer with 209 orders and 280,206 in revenue pulls
  the Euclidean distance metric hard. That is exactly why the alternative-k partitions are printed rather than hidden.
- **Silhouette at k=2 (0.8958) is higher than at k=4 (0.6162).** This is the small-k artefact, and the notebook deliberately
  shows it rather than quietly picking the flattering number. A reader who prefers a pure silhouette criterion would pick k=2 and
  get a two-way split with no marketing value.
- **Country is not used.** 38 countries are present and the 13 whales include a customer from the Netherlands, but country was
  not used as a feature because the base is far too thin per country to support segmentation.
- **Guest rows are dropped.** 135,080 line items have no `CustomerID`. They are excluded because the task is customer
  segmentation; guest data is nonetheless the largest single block in the file and a guest-to-account conversion project would
  be a natural follow-up.
- **This is association, not causation.** Segment membership is descriptive of past behaviour; the marketing actions are
  hypotheses to be A/B tested, not proven effects.

---

## How to reproduce

1. `pip install -r requirements.txt` (from the repository root).
2. Place `online_retail.csv` in `data/raw/`.
3. From `DataAnalytics-L1-CustomerSegmentation/`, run `jupyter lab notebooks/customer_segmentation.ipynb` → **Run All**.
4. The notebook re-generates `data/processed/customer_segmentation.csv` and all four PNGs in `outputs/figures/`.

The K-Means model is fitted with `random_state=42` and `n_init=10`, so the segmentation is **fully reproducible** and identical
across runs and machines.

---

## Outputs

| Artefact | Path |
|---|---|
| Notebook (executed) | [`notebooks/customer_segmentation.ipynb`](notebooks/customer_segmentation.ipynb) |
| Customer-level segmentation (4,338 × 15) | [`data/processed/customer_segmentation.csv`](data/processed/customer_segmentation.csv) |
| Elbow method | [`outputs/figures/elbow_method.png`](outputs/figures/elbow_method.png) |
| Cluster scatter plots (2 panels) | [`outputs/figures/cluster_scatter.png`](outputs/figures/cluster_scatter.png) |
| Customers per cluster | [`outputs/figures/customers_per_cluster.png`](outputs/figures/customers_per_cluster.png) |
| Customers per segment | [`outputs/figures/customers_per_segment.png`](outputs/figures/customers_per_segment.png) |
