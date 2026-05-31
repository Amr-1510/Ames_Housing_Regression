# Ames Housing — Regression Project

A from-scratch regression project on the Ames Housing dataset. Linear regression models are implemented manually (no sklearn for model fitting) using three approaches: Numerical View (Normal Equations), Statistical View (p-values & coefficients table), and Gradient Descent View. Both Ridge and Lasso regularization are explored across three model specifications.

---

## Dataset

**File:** `AmesHousing.csv`

**Target Variable:** `SalePrice` → log-transformed to `price_log` (`np.log1p`) to handle right skew and outliers.

---

## Selected Features

| Feature | Type | Reason for Selection |
|---|---|---|
| `Overall Qual` | Numeric | Strongest single predictor (Spearman ≈ 0.80) |
| `Gr Liv Area` | Numeric | Above-grade living area — key size driver |
| `Garage Cars` | Numeric | Garage capacity; `Garage Area` dropped (collinear) |
| `Total Bsmt SF` | Numeric | Total basement sq ft |
| `Year Built` | Numeric | Construction year; `Garage Yr Blt` dropped (correlated) |
| `Year Remod/Add` | Numeric | Most recent remodel year |
| `1st Flr SF` | Numeric | First-floor sq footage |
| `Neighborhood` | Categorical | Location — strongest real-estate driver |
| `Bsmt Qual` | Categorical | Basement quality rating |

Feature selection used Pearson correlation (for reference), **Spearman rank correlation** (main selector for skewed target), and **Mutual Information** for categorical features.

---

## Pipeline

### 1. Feature Selection
- Pearson correlation (reference only — suppressed by skew)
- Spearman rank correlation → numeric features selected
- Mutual Information (`mutual_info_regression`) → categorical features selected

### 2. Target Transformation
- `SalePrice` is right-skewed with outliers → `price_log = np.log1p(SalePrice)`
- Outliers handled via log transformation (not row removal)

### 3. Data Preprocessing
- Duplicate removal
- Null handling: numeric nulls → median; categorical nulls → mode
- IQR outlier check (informational only before split; capping applied on train set only)
- Train/Test split
- One-Hot Encoding for `Neighborhood` and `Bsmt Qual`
- StandardScaler (fit on train only)

---

## Model Specifications

Three specifications compared across two regularization types:

| Spec | Features | Models |
|---|---|---|
| Spec 1 — Best Single Predictor | `Overall Qual` only | Ridge (λ=1.0 / 0.01), Lasso (λ=0.1 / 0.001) |
| Spec 2 — Full Model | All selected features | Ridge, Lasso |
| Spec 3 — Domain Knowledge | `Overall Qual`, `Gr Liv Area`, `Garage Cars`, `Total Bsmt SF`, `Year Built` | Ridge, Lasso |

---

## Three Implementation Views

### Numerical View (Normal Equations)
Custom classes implementing closed-form solutions:

- `SimpleLinearRegressionNumericalView` — Ridge-penalised Normal Equations, single predictor
- `LassoNumericalView` — Coordinate Descent + Soft-Thresholding
- `MultipleLinearRegressionNumericalView` — multi-feature Ridge

Includes step-by-step derivation output, slope shrinkage tables, and λ sensitivity plots.

### Statistical View
Custom classes with coefficient tables, standard errors, t-statistics, and p-values:

- `MultiLinearRegressionStatisticalView` — Ridge
- `LassoMLRStatisticalView` — Lasso

Model comparison table with R² and Adjusted R² for all 6 spec/regularization combinations.

### Gradient Descent View
Custom `LinearRegressionGD` class:

- Supports Ridge (L2) and Lasso (L1) penalties
- Configurable learning rate, epochs
- Convergence (loss curve) plots
- Interactive Plotly visualizations: correlation heatmap, feature vs price_log scatter, residual plots, coefficient comparison bar chart

---

## Model Comparison Output

Final table comparing all 6 combinations (3 specs × 2 regularizations) across both implementation views (Statistical and Gradient Descent), showing R² and Adjusted R².

---

## Interactive Visualizations (Plotly)

- Correlation heatmap (numeric features)
- Interactive scatter with dropdown: any feature vs `price_log` + regression line
- Residual plots: OLS vs Ridge vs Lasso
- Coefficient comparison bar chart: OLS vs Ridge vs Lasso

---

## How to Run

1. Place `AmesHousing.csv` at the referenced path (update if needed).
2. Run `regressionFinall.ipynb` top to bottom.
3. All custom model classes are defined inline — no external model library needed.

---

## Requirements

```
numpy pandas matplotlib seaborn scikit-learn plotly
```
