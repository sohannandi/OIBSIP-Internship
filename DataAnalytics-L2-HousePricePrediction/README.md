# Level 2 — Predicting House Prices with Linear Regression

**Internship task:** Oasis Infobyte — Data Analytics, Level 2 (Task 3 of 3)
**Notebook:** [`notebooks/house_price_prediction.ipynb`](notebooks/house_price_prediction.ipynb)

---

## Problem statement and objective

House prices depend on a mix of measurable things — size, age, condition, garage, and above all **where the house is** — but
buyers and sellers routinely treat the estimate as a black box. A model that explains *which factors move the price and by
how much* is more useful than one that merely produces a number, because it tells a buyer what to improve.

The objective of this project is to build a **leakage-safe** linear regression model on the Ames Housing data, evaluate it
honestly on a held-out 20% of the houses, and interpret the coefficients **carefully** — accounting for feature scaling, for
one-hot encoding, and for the baseline category of each encoded feature.

---

## Dataset

| | |
|---|---|
| **Name** | Ames Housing (Dei-Joannes "Ames, Iowa: Residential Sales" data) |
| **Source** | `https://raw.githubusercontent.com/STATCowboy/pbidataflowstalk/master/AmesHousing.csv` |
| **Expected filename** | `AmesHousing.csv` |
| **Placement** | `DataAnalytics-L2-HousePricePrediction/data/raw/AmesHousing.csv` |
| **Shape** | 2,930 rows × 82 columns |
| **Target column** | `SalePrice` |
| **License / usage** | Public GitHub raw file; the underlying Ames data is a public research dataset. No credentials required. |

### Target distribution

`SalePrice` — mean **180,796.06**, median **160,000**, std **79,886.69**, min **12,789**, max **755,000**, **skew 1.744**
(right-skewed). The right skew is why the notebook later flags a log-transformed target as the most promising next step.

### Key columns used

| Column | Type | Why it is in the model |
|---|---|---|
| `OverallQual` | int | Expert overall quality rating, 1–10. The single strongest correlate of price (r = 0.799). |
| `GrLivArea` | int | Above-ground living area in sq ft. r = 0.707. |
| `GarageCars` | int | Garage capacity. r = 0.648. |
| `GarageArea` | int | Garage size in sq ft. r = 0.640. |
| `TotalBsmtSF` | int | Total basement area. r = 0.632. |
| `1stFlrSF` | int | First-floor area. r = 0.622. |
| `YearBuilt` | int | Construction year. r = 0.558. |
| `FullBath` | int | Full bathrooms. r = 0.546. |
| `YearRemod/Add` | int | Most recent renovation year. r = 0.533. |
| `TotRmsAbvGrd` | int | Total rooms above grade. r = 0.495. |
| `LotArea` | int | Lot size. r = 0.267. |
| `Neighborhood` | str | **Location.** The largest single lever in the fitted model. |
| `BldgType` | str | Single-family vs townhouse vs duplex. |
| `HouseStyle` | str | Ranch, split-level, etc. |
| `KitchenQual` | str | Kitchen condition grade. |
| `ExterQual` | str | Exterior condition grade. |
| `HeatingQC` | str | Heating quality grade. |
| `CentralAir` | str | Central air conditioning (Y/N). |
| `GarageType` | str | Garage construction type. |
| `PavedDrive` | str | Paved driveway (Y/N). |

20 features in total, expanding to **68 columns** after one-hot encoding.

### If the file is missing

The notebook raises a `FileNotFoundError` naming the exact expected path and the download URL. Place the file at
`data/raw/AmesHousing.csv` and re-run. The file is read with `na_values="NA"` and column names are stripped of spaces, which
is what makes `1stFlrSF` and `YearRemod/Add` addressable.

---

## Tools and libraries

Python 3.10+ · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `jupyter`

---

## Folder structure

```text
DataAnalytics-L2-HousePricePrediction/
├── README.md
├── notebooks/
│   └── house_price_prediction.ipynb   # the deliverable
├── data/
│   ├── raw/
│   │   └── AmesHousing.csv
│   └── processed/
│       ├── test_set_predictions.csv   # 586 test houses: actual, predicted, residual
│       └── model_coefficients.csv     # 68 rows: feature, coefficient
└── outputs/
    └── figures/
        ├── target_distribution.png
        ├── correlation_heatmap.png
        ├── actual_vs_predicted.png
        ├── residuals.png               # 3 panels
        ├── coefficients.png
        └── model_comparison.png
```

---

## Setup and run

```bash
# from the repository root
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cd DataAnalytics-L2-HousePricePrediction
jupyter lab notebooks/house_price_prediction.ipynb
```

Run all cells top to bottom (`Run All`). The notebook is committed executed.

---

## Analysis workflow

1. **Load & inspect** — shape, all 82 columns, dtypes, null counts (15,748 null cells in total), a target distribution chart
   and descriptive statistics.
2. **Feature selection** — a markdown cell explains the reasoning for each of the 20 features before any model is fitted.
3. **Missing values** — a documented strategy per column type (see below).
4. **Preprocessing pipeline** — `ColumnTransformer` with imputation + scaling + one-hot encoding, fitted **on the training set
   only**.
5. **Correlation heatmap** — numeric features against the target.
6. **80/20 split** — `train_test_split(test_size=0.2, random_state=42)`. Train 2,344 houses, test 586.
7. **Fit `LinearRegression`.**
8. **Evaluate** — MSE, RMSE, R² (plus MAE for context), each explained, each compared against a mean-only baseline.
9. **Actual vs predicted** — scatter with the y = x reference diagonal.
10. **Residuals** — three panels (vs predicted, distribution, vs `GrLivArea`) and a discussion of the patterns.
11. **Coefficient analysis** — with the baseline category of every one-hot feature printed and interpreted.
12. **Bonus** — Ridge and Lasso compared through the same leakage-safe pipeline, with alphas chosen by `cv=5` **inside the
    training set only**.

### Missing-value strategy

| Column | Problem | Strategy |
|---|---|---|
| `TotalBsmtSF` | 1 missing | Median imputation, inside the pipeline |
| `GarageCars` | 1 missing | Median imputation, inside the pipeline |
| `GarageArea` | 1 missing | Median imputation, inside the pipeline |
| `GarageType` | 157 missing | **Structural**: missing means *no garage*, so it is filled with the literal `"None"` **before** the pipeline, preserving the distinction between "no garage" and "attached garage" |

All imputation happens **inside** `Pipeline` / `ColumnTransformer` objects fitted on the training set only. Nothing is computed
on the full dataset before the split — that is the mechanism that prevents leakage here.

---

## Main charts and findings

All numbers below are the notebook's own executed output.

### 1. Target distribution — [`target_distribution.png`](outputs/figures/target_distribution.png)

Mean **180,796.06**, median **160,000**, std **79,886.69**, min **12,789**, max **755,000**, **skew 1.744**. A long right
tail means a small number of expensive houses will dominate a squared-error loss.

### 2. Correlations with the target — [`correlation_heatmap.png`](outputs/figures/correlation_heatmap.png)

`OverallQual` **0.799** · `GrLivArea` **0.707** · `GarageCars` **0.648** · `GarageArea` **0.640** · `TotalBsmtSF` **0.632** ·
`1stFlrSF` **0.622** · `YearBuilt` **0.558** · `FullBath` **0.546** · `YearRemod/Add` **0.533** · `TotRmsAbvGrd` **0.495** ·
`LotArea` **0.267**.

Quality and living area lead, and the size features cluster tightly together (garage and basement space correlate strongly with
each other), which is why the model shares credit among them and why a single coefficient should not be read as a standalone
effect.

### 3. Model performance — [`actual_vs_predicted.png`](outputs/figures/actual_vs_predicted.png)

On the **held-out test set of 586 houses**:

| Metric | Value | What it means |
|---|---|---|
| **MSE** | **1,054,885,803.03** | Mean squared error, in dollars². The unit is hard to read, so RMSE is the number to quote. |
| **RMSE** | **32,479.01** | Typical prediction error in dollars — about **18%** of the mean sale price. |
| **R²** | **0.8684** | The model explains about **86.8%** of the variance in price in the test set. |
| **MAE** | **19,045.39** | Typical error in dollars, *not* exaggerated by the few expensive houses. MAE being well below RMSE confirms the error distribution has heavy tails. |

For reference, a model that simply predicts the training mean for every house gets **RMSE 90,222.37** and **R² −0.0153**. The
fitted model therefore removes about **64%** of the baseline error.

**Train R² 0.8628 vs test R² 0.8684** — a gap of **−0.0056**, with the *test* score marginally higher. There is no sign of
overfitting here.

A good R² is **not** evidence that any single feature causes a price. These are associations measured on 2,930 houses in one
county, and the model is a description of that sample, not a causal model of the housing market.

### 4. Residuals — [`residuals.png`](outputs/figures/residuals.png)

- Residuals are **well centred**: mean **2,095** against a mean price of 180,796 (about 1%), with **284 houses predicted above**
  actual and **302 below**. The **skew of the residuals is −0.064 — essentially zero**, so the model is not making a systematic
  directional error.
- But the residual **spread grows with the size of the house**: mean absolute error rises from **12,174** in the cheapest
  predicted-price quartile to **35,648** in the most expensive — nearly **3×** — while the error *as a percentage of price* stays
  in a narrow **8.2%–11.9%** band. That is **heteroscedasticity**: the error in dollars scales with the value of the house.
- The largest errors run in **both directions**, and the single worst one is an **over-prediction**: residuals are
  `actual − predicted`, and the worst case is a house that sold for **184,750** but was predicted at **512,631** — an
  over-prediction of **327,881**. The largest under-prediction is **+215,244**. The three highest-priced test houses
  (625,000 / 611,657 / 610,000) are predicted at 436,507 / 396,413 / 436,793 — under-predicted by **$173,000 to $215,000**,
  which is the visible pattern in the actual-vs-predicted panel. **6.14% of test houses have an absolute error above $50,000.**
- The residuals-vs-`GrLivArea` panel shows a **gentle curve**, which hints that the true relationship between living area and
  price is not perfectly linear.

**What this implies:** the linear form is a reasonable but imperfect description of this relationship, and the diagnostics point
at two specific remedies — a **log-transformed target** to shrink the error spread and stop the expensive houses from
dominating the loss, and a **non-linear model** (gradient boosting, or polynomial / interaction features) to capture the
curvature. Both are flagged here as the natural next steps, not claimed as results.

### 5. Coefficients — [`coefficients.png`](outputs/figures/coefficients.png)

The notebook prints the **baseline (dropped) category** for every one-hot feature before interpreting any coefficient:
`Neighborhood='Blmngtn'`, `BldgType='1Fam'`, `HouseStyle='1.5Fin'`, `KitchenQual='Ex'`, `ExterQual='Ex'`, `HeatingQC='Ex'`,
`CentralAir='N'`, `GarageType='2Types'`, `PavedDrive='N'`. Every one-hot coefficient is therefore **a difference from that
baseline**, never an absolute price.

Top coefficients by absolute magnitude:

| Feature | Coefficient | Reading |
|---|---:|---|
| `Neighborhood=GrnHill` | **+115,985.78** | vs baseline neighbourhood `Blmngtn` — the largest lever in the model |
| `Neighborhood=StoneBr` | +56,675.11 | |
| `Neighborhood=NoRidge` | +42,300.82 | |
| `KitchenQual=TA` | −41,134.42 | vs baseline `Ex` (Excellent) |
| `KitchenQual=Fa` | −39,318.32 | |
| `KitchenQual=Gd` | −37,123.78 | |
| `BldgType=Twnhs` | −36,456.27 | vs baseline `1Fam` |
| `GrLivArea` (standardised) | **+34,004.83** | per one standard deviation (505 sq ft) |
| `ExterQual=Fa` | −31,633.66 | vs baseline `Ex` |
| `ExterQual=TA` | −29,979.66 | |
| `Neighborhood=NridgHt` | +29,762.67 | |
| `GarageType=None` | +29,472.08 | vs baseline `2Types` — see the caveat |
| `BldgType=TwnhsE` | −28,186.17 | |
| `ExterQual=Gd` | −28,124.00 | |
| `HeatingQC=Po` | −25,921.71 | vs baseline `Ex` |
| `HouseStyle=SFoyer` | +24,205.45 | |
| `Neighborhood=Greens` | +23,827.19 | |
| `Neighborhood=Crawfor` | +21,331.67 | |
| `GarageType=Attchd` | +19,089.60 | |
| `BldgType=Duplex` | −18,999.35 | |

**How to read this table:**

- **Location is the single biggest lever.** `GrnHill` is worth **+115,985.78** against the baseline neighbourhood — more than
  three times the effect of one extra standard deviation of living space. In this market, *where* a house is beats *how big* it
  is. These are differences relative to `Blmngtn`, not absolute prices.
- **`GrLivArea` is the largest numeric coefficient** at **+34,004.83 per standard deviation** (505 sq ft).
- **The negative condition coefficients are not errors.** The baselines for `KitchenQual`, `ExterQual` and `HeatingQC` are all
  `Ex` (Excellent), so **every other level is a discount from Excellent**: `KitchenQual=TA` is −41,134.42, `ExterQual=Fa` is
  −31,633.66, `HeatingQC=Po` is −25,921.71. The negative sign means *cheaper than the top grade*, not *bad*.
- **`BldgType` follows the same pattern.** With `1Fam` as the baseline, `Twnhs` is −36,456.27 and `Duplex` is −18,999.35 —
  townhouses and duplexes sell for less than single-family homes, holding everything else constant.
- **One coefficient deserves a flag: `GarageType=None` is +29,472.08.** The baseline for `GarageType` is `2Types` (the
  alphabetically first level, held by only 23 houses), so this compares against a very unusual category rather than against
  attached garages. Read it as *"houses with no garage differ from houses with a two-type garage"*, not as *"no garage is a
  premium"*. The `drop="first"` baseline is a mechanical choice, and for `GarageType` it is not a very natural reference
  category.
- **These are associations, not causal effects.** Credit is shared among correlated features (garage capacity, garage area and
  basement area all measure size), and the coefficients are measured on this sample of 2,930 houses.

### 6. Bonus — Ridge vs Lasso — [`model_comparison.png`](outputs/figures/model_comparison.png)

All three models are fitted on the **same preprocessed training matrix** and evaluated on the **same 586-house test set**.
`alpha` is chosen by `cv=5` **inside the training set only**, so the test set is untouched until final evaluation.

| Model | MSE | RMSE | R² | MAE | alpha | coefficients zeroed |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 1,054,885,803 | **32,479.01** | 0.8684 | 19,045.39 | — | 0 of 68 |
| Ridge (L2) | 1,055,475,000 | 32,488.08 | 0.8684 | **18,986.29** | 1.0 | 0 of 68 |
| Lasso (L1) | 1,057,224,000 | 32,514.99 | 0.8681 | 19,001.64 | 10.0 | **5 of 68** |

- **The three models are essentially tied.** Test RMSE spans only **36 dollars, about 0.1%**, and R² agrees to three decimals.
  With 20 features and 2,344 training rows there is not enough overfitting for a penalty to fix — which the tiny train/test R²
  gap confirms.
- **Lasso selected `alpha = 10.0`**, a real penalty, and zeroed **5 of 68** coefficients: a genuine but small feature-selection
  effect that nonetheless holds accuracy essentially unchanged.
- **Ridge selected `alpha = 1.0`** and zeroed **0 of 68** — exactly the expected behaviour of an L2 penalty, which shrinks
  coefficients towards zero but never sets any exactly to zero.
- **Ridge wins on MAE** (18,986.29 vs 19,045.39 and 19,001.64) even though it is *not* the best on RMSE. MAE is the more
  trustworthy of the two here because it is less dominated by the handful of expensive houses.
- **The honest conclusion:** the choice among these three linear models barely matters for this dataset. If the goal were
  accuracy, the effort is better spent on the **model form** — a log-transformed target or a non-linear learner — than on the
  choice of regulariser.

---

## Limitations and assumptions

- **A linear additive model is a simplification.** It cannot represent thresholds ("above 2,000 sq ft, extra space is worth
  less per sq ft") or interactions (a good kitchen matters more in an expensive neighbourhood). The curved residuals vs
  `GrLivArea` are direct evidence of this.
- **Squared-error loss is dominated by expensive houses.** The three priciest test houses are under-predicted by $173,000–$215,000
  each, and MAE (19,045) is far below RMSE (32,479) as a result. If the business cares about the typical house rather than the
  expensive one, a different loss or a log target would be more appropriate.
- **The target is right-skewed (1.744)** and the model predicts on the original scale, so errors are not symmetric in percentage
  terms. A log-transformed target is the single most promising improvement.
- **Only 20 of 82 columns are used.** Deliberate: the full Ames set has many near-duplicate and heavily collinear columns, and
  the selected set already yields R² 0.87. A more thorough feature-engineering pass (all 82 columns, interaction terms) would
  likely improve accuracy further and is out of scope here.
- **One county, one era.** Ames, Iowa, 2006–2010 sales. The coefficients describe that market; they do not transfer to another
  city or decade.
- **Sale price is not a clean measure of value.** It embeds the seller's timing, motivation and the market conditions of the
  sale, none of which the model can see.
- **The `GarageType` baseline is mechanical, not meaningful.** `drop="first"` picks `2Types` (23 houses), so that column's
  coefficients are harder to interpret than the rest. Changing the encoder to drop a more common level would read better.
- **Prediction intervals are not produced.** A point estimate and its residual spread are given, but a proper interval would need
  bootstrap or quantile-regression treatment.
- **Correlation is not causation.** Nothing here shows that spending money on a kitchen *causes* a price increase; it shows that
  houses with better kitchens sold for more *in this sample*.

---

## How to reproduce

1. `pip install -r requirements.txt` (from the repository root).
2. Place `AmesHousing.csv` in `data/raw/`.
3. From `DataAnalytics-L2-HousePricePrediction/`, run `jupyter lab notebooks/house_price_prediction.ipynb` → **Run All**.
4. The notebook re-generates `data/processed/test_set_predictions.csv`, `data/processed/model_coefficients.csv` and all six PNGs.

**Determinism:** `train_test_split(test_size=0.2, random_state=42)` and `LinearRegression` (a closed-form solver) mean the model,
the metrics and the coefficient table are **identical on every run and every machine**. `RidgeCV` / `LassoCV` are also
deterministic here. No metric in this README depends on a lucky seed.

---

## Outputs

| Artefact | Path |
|---|---|
| Notebook (executed) | [`notebooks/house_price_prediction.ipynb`](notebooks/house_price_prediction.ipynb) |
| Test-set predictions (586 rows) | [`data/processed/test_set_predictions.csv`](data/processed/test_set_predictions.csv) |
| Coefficient table (68 rows) | [`data/processed/model_coefficients.csv`](data/processed/model_coefficients.csv) |
| Target distribution | [`outputs/figures/target_distribution.png`](outputs/figures/target_distribution.png) |
| Correlation heatmap | [`outputs/figures/correlation_heatmap.png`](outputs/figures/correlation_heatmap.png) |
| Actual vs predicted | [`outputs/figures/actual_vs_predicted.png`](outputs/figures/actual_vs_predicted.png) |
| Residuals (3 panels) | [`outputs/figures/residuals.png`](outputs/figures/residuals.png) |
| Coefficients | [`outputs/figures/coefficients.png`](outputs/figures/coefficients.png) |
| Model comparison | [`outputs/figures/model_comparison.png`](outputs/figures/model_comparison.png) |
