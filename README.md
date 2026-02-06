# DS Time Series

A Python library for hierarchical time series forecasting, designed for Customer-Material sales prediction with fiscal calendar awareness.

## Installation

```bash
# Clone and install
git clone https://github.com/your-org/ds_timeseries.git
cd ds_timeseries

# Create environment with uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# For Prophet support
uv pip install prophet
```

## Quick Start

```python
from ds_timeseries.data import download_toy_dataset
from ds_timeseries.models import LightGBMForecaster
from ds_timeseries.evaluation.metrics import wape, mae

# Load data (M5 Walmart sample)
df = download_toy_dataset("m5_sample", n_series=100)

# Train/test split (chronological - no random splits!)
cutoff = df["ds"].max() - pd.Timedelta(weeks=4)
train = df[df["ds"] <= cutoff]
test = df[df["ds"] > cutoff]

# Fit and predict
model = LightGBMForecaster()
model.fit(train)
forecasts = model.predict(horizon=4)

# Evaluate
merged = test.merge(forecasts, on=["unique_id", "ds"])
print(f"WAPE: {wape(merged['y'], merged['yhat']):.2%}")
```

---

## Data Format

All data must follow this schema:

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | str | Time series identifier (e.g., "CUST001_MAT123") |
| `ds` | datetime | Date (week start date, Monday) |
| `y` | float | Target variable (sales quantity/value) |

**Optional hierarchy columns**: `customer_id`, `material_id`, `store_id`, `item_id`, `cat_id`, etc.

---

## Available Models

### 1. Baselines

```python
from ds_timeseries.models import (
    NaiveForecaster,           # Last value
    MovingAverageForecaster,   # Rolling mean
    SeasonalNaiveForecaster,   # Same week last year
)

# Simple baseline
model = MovingAverageForecaster(window=4)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### 2. Statistical Models

```python
from ds_timeseries.models import ETSForecaster

# Exponential Smoothing (auto-selects best ETS spec)
model = ETSForecaster(season_length=52)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### 3. LightGBM (Recommended)

Best for structured business data. Uses Tweedie objective for intermittent demand.

```python
from ds_timeseries.models import LightGBMForecaster
from ds_timeseries.features import FeatureConfig, FiscalCalendarConfig

# Configure features
feature_config = FeatureConfig(
    lags=[1, 4, 13, 52],           # See "Choosing Lag Features" below
    rolling_windows=[4, 13, 26],
    rolling_aggs=["mean", "std"],
    fiscal_config=FiscalCalendarConfig(
        fiscal_year_start_month=11,  # November
        week_pattern="5-4-4",
    ),
)

model = LightGBMForecaster(
    feature_config=feature_config,
    use_tweedie=True,  # Better for zeros in data
)
model.fit(train)
forecasts = model.predict(horizon=4)

# View feature importance
print(model.get_feature_importance().head(10))
```

### 4. XGBoost

```python
from ds_timeseries.models import XGBoostForecaster

model = XGBoostForecaster(feature_config=feature_config)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### 5. CatBoost

CatBoost handles categorical features natively and is robust to overfitting.

```python
from ds_timeseries.models import CatBoostForecaster

model = CatBoostForecaster(feature_config=feature_config)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### 6. Direct vs Recursive Forecasting (M5 Winner Approach)

LightGBM, XGBoost, and CatBoost support both strategies:

```python
# Recursive (default): predict one step, use prediction as feature for next step
recursive_model = LightGBMForecaster(strategy="recursive")
recursive_model.fit(train)

# Direct: train separate model for each horizon step
direct_model = LightGBMForecaster(strategy="direct")
direct_model.fit(train, horizon=4)  # Must specify horizon at fit time

# M5 1st place: averaged both approaches
```

### 7. Prophet

Best for strong seasonality and holiday effects.

```python
from ds_timeseries.models import ProphetForecaster

model = ProphetForecaster(
    yearly_seasonality=True,
    weekly_seasonality=False,
)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### 8. Prophet + XGBoost Hybrid (Advanced)

Prophet captures trend/seasonality, XGBoost learns residual patterns.

```python
from ds_timeseries.models import ProphetXGBoostHybrid

model = ProphetXGBoostHybrid(feature_config=feature_config)
model.fit(train)
forecasts = model.predict(horizon=4)
```

---

## Neural Network Models (Optional)

State-of-the-art deep learning models. Install with:
```bash
uv pip install -e ".[neural]"
```

### N-BEATS (M4 Competition Winner)

Pure deep learning approach with interpretable trend/seasonality decomposition.

```python
from ds_timeseries.models import NBEATSForecaster, NeuralConfig

config = NeuralConfig(
    input_size=52,  # 1 year lookback
    max_steps=500,
    batch_size=32,
)

model = NBEATSForecaster(horizon=4, config=config)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### NHITS (AAAI 2023)

Improved N-BEATS with hierarchical interpolation for long-horizon forecasting.

```python
from ds_timeseries.models import NHITSForecaster

model = NHITSForecaster(horizon=4)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### Temporal Fusion Transformer (TFT)

Google's state-of-the-art interpretable model (36-69% better than DeepAR).

```python
from ds_timeseries.models import TFTForecaster

model = TFTForecaster(horizon=4, num_heads=4)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### DeepAR (Amazon)

Probabilistic autoregressive model with cross-series learning.

```python
from ds_timeseries.models import DeepARForecaster

model = DeepARForecaster(horizon=4)
model.fit(train)
forecasts = model.predict(horizon=4)
```

### Neural Ensemble

Combine multiple neural architectures:

```python
from ds_timeseries.models import NeuralEnsembleForecaster

# Average predictions from N-BEATS, NHITS, and LSTM
model = NeuralEnsembleForecaster(
    horizon=4,
    models=["nbeats", "nhits", "lstm"],
)
model.fit(train)
forecasts = model.predict(horizon=4)
```

---

## Adding External Variables (Exogenous Features)

### For ML Models (LightGBM, XGBoost, Hybrid)

Add columns to your DataFrame before fitting. The model will automatically include them:

```python
# Add external features to your data
df["price"] = ...           # Numeric feature
df["promotion"] = ...       # 0/1 flag
df["holiday_flag"] = ...    # Boolean

# These become features automatically
model = LightGBMForecaster(feature_config=feature_config)
model.fit(df)

# For prediction, include future values of exogenous variables
future_df["price"] = ...    # Must provide future values
```

**Note**: For recursive prediction, you need future values of exogenous variables.

### For Prophet

Use Prophet's built-in regressor API:

```python
from prophet import Prophet

# Fit with regressors
model = Prophet()
model.add_regressor("price")
model.add_regressor("promotion")
model.fit(train[["ds", "y", "price", "promotion"]])

# Predict (must include future regressor values)
future = model.make_future_dataframe(periods=4, freq="W-MON")
future["price"] = ...       # Future prices
future["promotion"] = ...   # Future promotions
forecast = model.predict(future)
```

---

## Choosing Lag Features

### Guidelines

| Your Data Frequency | Recommended Lags | Reasoning |
|---------------------|------------------|-----------|
| **Weekly** | `[1, 2, 4, 8, 13, 26, 52]` | 1w, 2w, 1mo, 2mo, quarter, half-year, year |
| **Daily** | `[1, 7, 14, 28, 365]` | 1d, 1w, 2w, 1mo, 1yr |
| **Monthly** | `[1, 3, 6, 12]` | 1mo, quarter, half-year, year |

### For Weekly Sales Data (Recommended)

```python
feature_config = FeatureConfig(
    # Core lags (always include)
    lags=[
        1,    # Last week (captures momentum)
        4,    # Last month
        52,   # Same week last year (seasonality)
    ],

    # Extended lags (if you have enough history)
    # lags=[1, 2, 4, 8, 13, 26, 52],

    # Rolling windows (match your business cycles)
    rolling_windows=[
        4,    # Monthly average
        13,   # Quarterly average
    ],
)
```

### Rule of Thumb

1. **Always include lag 1**: Captures recent momentum
2. **Include seasonal lag**: 52 for weekly (same week last year)
3. **Include business cycle lags**: 4 (month), 13 (quarter) for weekly
4. **Need 2x history**: If using lag 52, need at least 104 weeks of data

### Feature Importance Analysis

After fitting, check which lags matter:

```python
model = LightGBMForecaster(feature_config=config)
model.fit(train)

importance = model.get_feature_importance()
print(importance.head(15))

# If y_lag_52 has low importance, your data may not have strong yearly seasonality
# If y_lag_1 dominates, try adding more rolling features
```

---

## Fiscal Calendar Features

For businesses with fiscal calendars (not calendar year):

```python
from ds_timeseries.features import FiscalCalendarConfig, FeatureConfig

# November fiscal year start, 5-4-4 week pattern
fiscal_config = FiscalCalendarConfig(
    fiscal_year_start_month=11,  # November
    week_pattern="5-4-4",        # 5 weeks, 4 weeks, 4 weeks per quarter
)

feature_config = FeatureConfig(
    fiscal_config=fiscal_config,
    lags=[1, 4, 13, 52],
)

# This adds features like:
# - fiscal_year, fiscal_quarter, fiscal_month
# - fiscal_week_in_month, fiscal_week_in_quarter
# - is_fiscal_month_end (critical for "hockey stick" patterns!)
# - is_fiscal_quarter_end, is_fiscal_year_end
```

### Bringing Your Own Fiscal Calendar

If your company already has a fiscal calendar (from Finance, SAP, etc.), you can pass it directly instead of having the library generate one. Your calendar needs **one row per week** with `ds` dates matching your data's weekly frequency.

#### Required Schema

| Column | Type | Description |
|--------|------|-------------|
| `ds` | datetime | **Week-end date**, must match your data's frequency (e.g., Saturdays for W-SAT). This is the join key. |
| `fiscal_year` | int | Fiscal year identifier (e.g., 2024). |
| `fiscal_month` | int | Fiscal month 1-12 within the fiscal year. |

#### Optional Columns (used by feature engineering)

| Column | Type | Description |
|--------|------|-------------|
| `fiscal_quarter` | int | Quarter 1-4. If missing, rollup still works but output won't include it. |
| `fiscal_week` | int | Week number 1-52 within the fiscal year. |
| `fiscal_week_in_month` | int | Week position within the month (1-5). |
| `fiscal_week_in_quarter` | int | Week position within the quarter (1-13). |
| `is_fiscal_month_end` | bool | `True` for the last week of each fiscal month. **Critical** for hockey stick modeling. |
| `is_fiscal_quarter_end` | bool | `True` for the last week of each fiscal quarter. |
| `is_fiscal_year_end` | bool | `True` for the last week of the fiscal year. |

The more columns you provide, the more features the ML models can use. At minimum you need `ds`, `fiscal_year`, and `fiscal_month`. For hockey stick detection, also include `is_fiscal_month_end`.

#### Example: Loading Your Calendar

```python
import pandas as pd
from ds_timeseries.features import (
    engineer_features,
    rollup_to_fiscal_month,
    add_fiscal_features,
    FeatureConfig,
)

# Load your company's fiscal calendar
my_cal = pd.read_csv("our_fiscal_calendar.csv", parse_dates=["ds"])

# Verify: one row per week, dates match your data
assert my_cal["ds"].diff().dropna().eq(pd.Timedelta(days=7)).all()
print(my_cal.head())
#          ds  fiscal_year  fiscal_quarter  fiscal_month  is_fiscal_month_end
# 2023-11-04         2024               1             1                False
# 2023-11-11         2024               1             1                False
# ...

# Use with feature engineering (adds fiscal columns to your data)
df_features = engineer_features(df, fiscal_calendar=my_cal)

# Use with monthly rollup (derives weeks_expected from your calendar)
monthly_fcst = rollup_to_fiscal_month(forecasts, fiscal_calendar=my_cal)
```

#### Key Requirements

1. **Date alignment**: Your calendar's `ds` values must land on the same weekday as your data. If your data uses W-SAT (Saturday week-ends), your calendar must too. Rows that don't match will be silently unmatched.

2. **Complete coverage**: The calendar should cover all dates in your data. Missing weeks will trigger a warning and those rows will be excluded from the rollup.

3. **`weeks_expected` is automatic**: When you pass your own calendar, `rollup_to_fiscal_month` counts the weeks per fiscal month directly from the calendar — no need to declare a 5-4-4 or 4-4-5 pattern.

### Rolling Up Weekly Forecasts to Fiscal Months

Aggregate weekly predictions into fiscal monthly totals. Handles months with 4 or 5 weeks based on the pattern.

```python
from ds_timeseries.features import rollup_to_fiscal_month, FiscalCalendarConfig

# Option A: Auto-generate calendar from config
config = FiscalCalendarConfig(fiscal_year_start_month=11, week_pattern="5-4-4")
monthly_fcst = rollup_to_fiscal_month(forecasts, config, value_cols="yhat")

# Option B: Use your own fiscal calendar
monthly_fcst = rollup_to_fiscal_month(forecasts, fiscal_calendar=my_cal)

# Filter to only complete months (all expected weeks present)
monthly_fcst = monthly_fcst[monthly_fcst["is_complete"]]

# Monthly-level evaluation
merged = actuals.merge(forecasts, on=["unique_id", "ds"])
monthly = rollup_to_fiscal_month(merged, config, value_cols=["y", "yhat"])
monthly = monthly[monthly["is_complete"]]

from ds_timeseries.evaluation.metrics import wape
print(f"Monthly WAPE: {wape(monthly['y'], monthly['yhat']):.2%}")
```

The output includes `weeks_expected`, `weeks_present`, and `is_complete` so you can decide how to handle partial months at the edges of your forecast horizon.

---

## Cross-Validation

Use time series CV (no random splits!):

```python
from ds_timeseries.evaluation import cross_validate, cv_score
from ds_timeseries.evaluation.metrics import wape

# Full CV with predictions
results = cross_validate(
    model,
    df,
    n_folds=5,
    horizon=4,
    min_train_size=104,  # At least 2 years
)

# Quick metric summary
scores = cv_score(model, df, metric_fn=wape, n_folds=5, horizon=4)
print(f"WAPE: {scores['mean']:.2%} ± {scores['std']:.2%}")
```

---

## Complete Guide: Apply to Your Own Data

This guide walks you through forecasting your company's data end-to-end, assuming you have hierarchical sales data similar to M5 (Customer-Material, Store-SKU, etc.).

### Step 1: Prepare Your Data

**Required columns:**
- `unique_id`: Unique identifier for each time series (e.g., "CUST001_MAT123")
- `ds`: Date column (datetime, ideally week start date)
- `y`: Target variable (sales quantity or revenue)

**Optional hierarchy columns:**
- `customer_id`, `material_id`, `store_id`, `category`, `region`, etc.

```python
import pandas as pd
import numpy as np

# Load your raw data
raw_df = pd.read_csv("your_sales_data.csv")

# Convert to required format
df = raw_df.rename(columns={
    "customer_code": "customer_id",
    "material_code": "material_id",
    "week_start_date": "ds",
    "quantity_sold": "y",
})

# Create unique_id from hierarchy
df["unique_id"] = df["customer_id"] + "_" + df["material_id"]

# Ensure datetime
df["ds"] = pd.to_datetime(df["ds"])

# Sort by series and date
df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

# Check your data
print(f"Total rows: {len(df):,}")
print(f"Unique series: {df['unique_id'].nunique():,}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
print(f"Zeros in target: {(df['y'] == 0).mean():.1%}")
```

### Step 2: Exploratory Data Analysis

```python
# Check data quality
from ds_timeseries.data import classify_demand

# Classify your demand patterns
# ADI > 1.32 and CV² > 0.49 = lumpy (hardest to forecast)
classification = classify_demand(df)
print(classification.value_counts())

# Check history length per series
history_weeks = df.groupby("unique_id")["ds"].nunique()
print(f"Min history: {history_weeks.min()} weeks")
print(f"Max history: {history_weeks.max()} weeks")
print(f"Median history: {history_weeks.median():.0f} weeks")

# Filter to series with enough history (need 2x your longest lag)
MIN_HISTORY = 104  # 2 years for lag 52
valid_series = history_weeks[history_weeks >= MIN_HISTORY].index
df = df[df["unique_id"].isin(valid_series)]
print(f"Series with {MIN_HISTORY}+ weeks: {len(valid_series):,}")
```

### Step 3: Choose Models Based on Demand Pattern

```python
from ds_timeseries.models import (
    # For regular demand (low zeros)
    LightGBMForecaster,
    XGBoostForecaster,
    ETSForecaster,

    # For intermittent demand (>50% zeros)
    CrostonForecaster,
    SBAForecaster,      # Recommended: bias-corrected Croston
    TSBForecaster,      # Best if obsolescence risk
)

# Check zero percentage to choose model type
zero_pct = (df["y"] == 0).mean()
print(f"Zero percentage: {zero_pct:.1%}")

if zero_pct > 0.8:
    print("→ Highly intermittent: Use SBA or TSB")
    base_model = SBAForecaster()
elif zero_pct > 0.5:
    print("→ Intermittent: Use SBA or LightGBM with Tweedie")
    base_model = LightGBMForecaster(use_tweedie=True)
else:
    print("→ Regular demand: Use LightGBM or XGBoost")
    base_model = LightGBMForecaster()
```

### Step 4: Configure Features

```python
from ds_timeseries.features import FeatureConfig, FiscalCalendarConfig

# Configure your fiscal calendar (if applicable)
fiscal_config = FiscalCalendarConfig(
    fiscal_year_start_month=11,  # November FY start
    week_pattern="5-4-4",        # Or "4-4-5", "4-5-4"
)

# Configure lag features based on your data
feature_config = FeatureConfig(
    # Lags: recent (1), monthly (4), quarterly (13), yearly (52)
    lags=[1, 4, 13, 52],

    # Rolling windows for smoothing
    rolling_windows=[4, 13],  # 1 month, 1 quarter
    rolling_aggs=["mean", "std"],

    # Your fiscal calendar
    fiscal_config=fiscal_config,

    # Difference features for trend
    diff_periods=[1, 52],
)

# Create model with features
model = LightGBMForecaster(
    feature_config=feature_config,
    use_tweedie=True,  # Recommended for sales data
)
```

### Step 5: Establish Baselines

```python
from ds_timeseries.models import NaiveForecaster, MovingAverageForecaster, ETSForecaster
from ds_timeseries.evaluation import cross_validate, cv_score
from ds_timeseries.evaluation.metrics import wape, mae

# Always compare against baselines!
baselines = {
    "Naive": NaiveForecaster(),
    "MA(4)": MovingAverageForecaster(window=4),
    "ETS": ETSForecaster(),
}

HORIZON = 4  # 4-week forecast
N_FOLDS = 3  # 3-fold CV

baseline_results = {}
for name, model in baselines.items():
    scores = cv_score(model, df, metric_fn=wape, n_folds=N_FOLDS, horizon=HORIZON)
    baseline_results[name] = scores["mean"]
    print(f"{name}: WAPE = {scores['mean']:.2%} ± {scores['std']:.2%}")

best_baseline = min(baseline_results, key=baseline_results.get)
print(f"\nBest baseline: {best_baseline} ({baseline_results[best_baseline]:.2%})")
```

### Step 6: Train ML Models

```python
from ds_timeseries.models import LightGBMForecaster, XGBoostForecaster, CatBoostForecaster

# Train multiple ML models
ml_models = {
    "LightGBM": LightGBMForecaster(feature_config=feature_config, use_tweedie=True),
    "XGBoost": XGBoostForecaster(feature_config=feature_config),
    "CatBoost": CatBoostForecaster(feature_config=feature_config),
}

ml_results = {}
for name, model in ml_models.items():
    scores = cv_score(model, df, metric_fn=wape, n_folds=N_FOLDS, horizon=HORIZON)
    ml_results[name] = scores["mean"]
    print(f"{name}: WAPE = {scores['mean']:.2%} ± {scores['std']:.2%}")

# Compare to baseline
best_ml = min(ml_results, key=ml_results.get)
improvement = (baseline_results[best_baseline] - ml_results[best_ml]) / baseline_results[best_baseline]
print(f"\nBest ML: {best_ml} ({ml_results[best_ml]:.2%})")
print(f"Improvement over {best_baseline}: {improvement:.1%}")
```

### Step 7: Ensemble for Production

```python
from ds_timeseries.models import SimpleEnsemble, WeightedEnsemble, M5WinnerEnsemble

# Option A: Simple ensemble (average of top models)
ensemble = SimpleEnsemble([
    LightGBMForecaster(feature_config=feature_config),
    XGBoostForecaster(feature_config=feature_config),
])

# Option B: M5 winner architecture (if you have hierarchy columns)
if "store_id" in df.columns:
    ensemble = M5WinnerEnsemble(
        feature_config=feature_config,
        pooling_levels=[["store_id"], ["store_id", "category"]],
    )
    ensemble.fit(df, horizon=HORIZON)
else:
    ensemble.fit(df)

# Evaluate
scores = cv_score(ensemble, df, metric_fn=wape, n_folds=N_FOLDS, horizon=HORIZON)
print(f"Ensemble WAPE: {scores['mean']:.2%}")
```

### Step 8: Generate Final Forecasts

```python
# Train on all data
final_model = LightGBMForecaster(feature_config=feature_config)
final_model.fit(df)

# Generate forecasts
forecasts = final_model.predict(horizon=HORIZON)

# View forecasts
print(forecasts.head(20))

# Export for business use
forecasts.to_csv("forecasts.csv", index=False)

# Or merge with hierarchy info
forecasts_with_info = forecasts.merge(
    df[["unique_id", "customer_id", "material_id"]].drop_duplicates(),
    on="unique_id",
)
forecasts_with_info.to_csv("forecasts_detailed.csv", index=False)
```

### Step 9: Visualize Results

```python
from ds_timeseries.evaluation import (
    plot_forecast,
    plot_forecast_grid,
    plot_model_comparison,
    plot_feature_importance,
)

# Plot a single series
fig = plot_forecast(
    actuals=df,
    forecasts=forecasts,
    series_id=df["unique_id"].iloc[0],
    n_history=52,  # Show last year
)
fig.savefig("forecast_example.png", dpi=150, bbox_inches="tight")

# Plot grid of multiple series
fig = plot_forecast_grid(df, forecasts, n_series=9)
fig.savefig("forecast_grid.png", dpi=150, bbox_inches="tight")

# Check feature importance
importance = final_model.get_feature_importance()
fig = plot_feature_importance(importance, top_n=15)
fig.savefig("feature_importance.png", dpi=150, bbox_inches="tight")

print("Top features:")
print(importance.head(10))
```

### Step 10: Monitor and Iterate

```python
# When new data arrives, backtest to monitor performance
from ds_timeseries.evaluation import cross_validate

def monitor_forecast_quality(df, model, horizon=4):
    """Run weekly to track forecast accuracy over time."""
    results = cross_validate(model, df, n_folds=1, horizon=horizon)

    metrics = {
        "date": df["ds"].max(),
        "wape": wape(results["y"], results["yhat"]),
        "mae": mae(results["y"], results["yhat"]),
        "bias": (results["yhat"].mean() - results["y"].mean()) / results["y"].mean(),
    }

    return metrics

# Track metrics over time
tracking = monitor_forecast_quality(df, final_model)
print(f"Current WAPE: {tracking['wape']:.2%}")
print(f"Bias: {tracking['bias']:+.1%}")

# Alert if performance degrades
if tracking["wape"] > baseline_results[best_baseline] * 1.1:
    print("⚠️ WARNING: Model performance has degraded. Consider retraining.")
```

### Checklist for Production

- [ ] Data validated: no missing dates, correct hierarchy
- [ ] Baseline established: know what "good" looks like
- [ ] CV evaluation: don't trust train-set metrics
- [ ] Feature importance reviewed: no data leakage (future features)
- [ ] Forecasts sanity-checked: no negative values, reasonable ranges
- [ ] Monitoring set up: track accuracy over time
- [ ] Retraining schedule: weekly/monthly refresh

---

## Ensemble Models

Combine multiple models for improved accuracy:

```python
from ds_timeseries.models import (
    SimpleEnsemble,
    WeightedEnsemble,
    StackingEnsemble,
    HierarchicalEnsemble,
    DRFAMEnsemble,
    M5WinnerEnsemble,
    LightGBMForecaster,
    XGBoostForecaster,
    ETSForecaster,
)

# Simple average of predictions
ensemble = SimpleEnsemble(
    models=[
        LightGBMForecaster(feature_config=feature_config),
        XGBoostForecaster(feature_config=feature_config),
        ETSForecaster(),
    ]
)
ensemble.fit(train)
forecasts = ensemble.predict(horizon=4)

# CV-optimized weights (learns best combination)
weighted = WeightedEnsemble(
    models=[
        LightGBMForecaster(feature_config=feature_config),
        XGBoostForecaster(feature_config=feature_config),
    ],
    cv_folds=3,
)
weighted.fit(train)

# Stacking with meta-learner (Ridge regression)
stacking = StackingEnsemble(
    models=[
        LightGBMForecaster(feature_config=feature_config),
        XGBoostForecaster(feature_config=feature_config),
    ],
)
stacking.fit(train)
```

### M5 Competition Winner Ensembles (Advanced)

```python
# DRFAM: Direct + Recursive Forecast Averaging Method (M5 1st place)
# Averages predictions from both strategies at multiple pooling levels
drfam = DRFAMEnsemble(
    model_class=LightGBMForecaster,
    model_params={"feature_config": feature_config},
    pooling_levels=["store_id", "dept_id"],
    use_direct=True,
    use_recursive=True,
)
drfam.fit(train, horizon=4)
forecasts = drfam.predict(horizon=4)

# Full M5 winner architecture (220 models!)
# - 3 pooling levels (store, store-cat, store-dept)
# - 2 strategies (direct, recursive)
# - Tweedie objective, optimized hyperparameters
m5_winner = M5WinnerEnsemble(
    feature_config=feature_config,
    pooling_levels=[
        ["store_id"],
        ["store_id", "cat_id"],
        ["store_id", "dept_id"],
    ],
)
m5_winner.fit(train, horizon=4)
forecasts = m5_winner.predict(horizon=4)
print(f"Total models: {m5_winner.get_model_count()}")

# Hierarchical ensemble: different models per hierarchy level
hierarchical = HierarchicalEnsemble(
    model_class=LightGBMForecaster,
    levels=["store_id", "cat_id"],
)
hierarchical.fit(train)
```

---

## Hyperparameter Tuning

Built-in tuning with time series cross-validation:

```python
from ds_timeseries.models.tuning import (
    grid_search_cv,
    random_search_cv,
    tune_lightgbm,
    tune_xgboost,
)

# Grid search
param_grid = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.05, 0.1],
}
result = grid_search_cv(
    LightGBMForecaster,
    param_grid,
    df,
    n_folds=3,
    horizon=4,
)
print(f"Best params: {result.best_params}")
print(f"Best WAPE: {result.best_score:.2%}")

# Random search (faster for large search spaces)
param_distributions = {
    "num_leaves": [31, 63, 127, 255],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_samples": [10, 20, 50, 100],
}
result = random_search_cv(
    LightGBMForecaster,
    param_distributions,
    df,
    n_iter=20,
    n_folds=3,
)

# Convenience functions with sensible search spaces
lgb_result = tune_lightgbm(df, feature_config, n_iter=20)
xgb_result = tune_xgboost(df, feature_config, n_iter=20)

# Use tuned model
best_model = lgb_result.best_model
forecasts = best_model.predict(horizon=4)
```

### Optuna Integration (Optional)

```python
from ds_timeseries.models.tuning import optuna_tune

# Requires: uv pip install optuna
result = optuna_tune(
    LightGBMForecaster,
    df,
    feature_config=feature_config,
    n_trials=50,
    n_folds=3,
)
```

---

## Visualization

Visualize forecasts and compare models:

```python
from ds_timeseries.evaluation import (
    plot_forecast,
    plot_forecast_grid,
    plot_model_comparison,
    plot_metrics_comparison,
    plot_residuals,
    plot_feature_importance,
)

# Single series forecast vs actuals
fig = plot_forecast(
    actuals=test,
    forecasts=forecasts,
    series_id="FOODS_1_001_CA_1",
    n_history=52,  # Show last 52 weeks of history
)
fig.savefig("forecast.png")

# Grid of multiple series
fig = plot_forecast_grid(actuals, forecasts, n_series=9)

# Compare multiple models on same series
model_forecasts = {
    "LightGBM": lgb_forecasts,
    "XGBoost": xgb_forecasts,
    "ETS": ets_forecasts,
}
fig = plot_model_comparison(actuals, model_forecasts, series_id="FOODS_1_001_CA_1")

# Compare metrics across models
metrics_df = pd.DataFrame({
    "model": ["XGBoost", "LightGBM", "ETS"],
    "wape": [0.3678, 0.3716, 0.3995],
    "mae": [5.55, 5.60, 6.03],
})
fig = plot_metrics_comparison(metrics_df, metric="wape")

# Residual analysis
fig = plot_residuals(actuals, forecasts)

# Feature importance (for ML models)
importance_df = model.get_feature_importance()
fig = plot_feature_importance(importance_df, top_n=20)
```

---

## Hierarchical Reconciliation

Ensure forecasts sum correctly across hierarchy levels:

```python
from ds_timeseries.reconciliation import (
    HierarchySpec,
    reconcile_forecasts,
    bottom_up_reconcile,
    top_down_reconcile,
    mintrace_reconcile,
)

# Define hierarchy
hierarchy = HierarchySpec(
    levels=["item_id", "dept_id", "cat_id", "store_id", "state_id"]
)

# Bottom-up: aggregate from most granular level
reconciled = bottom_up_reconcile(forecasts, actuals, hierarchy)

# Top-down: disaggregate from total using historical proportions
reconciled = top_down_reconcile(forecasts, actuals, hierarchy)

# MinTrace (optimal): minimize trace of forecast error covariance
reconciled = mintrace_reconcile(forecasts, actuals, hierarchy, method="ols")
reconciled = mintrace_reconcile(forecasts, actuals, hierarchy, method="wls")

# Generic interface
reconciled = reconcile_forecasts(
    forecasts, actuals, hierarchy,
    method="mintrace",  # or "bottom_up", "top_down"
)
```

---

## Benchmark Results (M5 Sample, 50 series, 4-week horizon)

### Single Models

| Model | WAPE | MAE | Notes |
|-------|------|-----|-------|
| **XGBoost** | **36.78%** | **5.55** | Best single model |
| LightGBM (Tweedie) | 37.16% | 5.60 | Fast, handles zeros well |
| ETS | 39.95% | 6.03 | Best statistical baseline |
| MovingAvg(4) | 52.42% | 7.91 | Simple baseline |
| Prophet | 56.30% | 8.49 | Better for aggregate data |
| Prophet+XGB | 58.52% | 8.83 | Hybrid underperforms here |

### Ensemble Models

| Model | WAPE | Notes |
|-------|------|-------|
| XGBoost | 36.78% | Best single model |
| Weighted Ensemble | 36.78% | Learned 100% XGBoost weight |
| LightGBM | 37.16% | |
| Simple Ensemble | 37.17% | Average of 3 models |

**Key Insights:**
- ML models (XGBoost, LightGBM) beat ETS baseline by ~7-8% relative improvement
- XGBoost slightly edges out LightGBM on this dataset
- Weighted ensemble learned to weight XGBoost at 100% (single model was best)
- Prophet underperforms on granular item-level data (better for aggregate forecasts)

---

## Project Structure

```
ds_timeseries/
├── src/ds_timeseries/
│   ├── data/                    # Data loading (M5, synthetic)
│   │   ├── toy_datasets.py      # Download M5 sample data
│   │   └── preprocessing.py     # Data cleaning, aggregation
│   ├── features/                # Feature engineering
│   │   ├── calendar.py          # Fiscal calendar (5-4-4, 4-4-5 patterns)
│   │   ├── lags.py              # Lag, rolling, diff features
│   │   └── engineering.py       # Feature pipeline
│   ├── models/                  # Forecasting models
│   │   ├── baselines.py         # Naive, MovingAverage, SeasonalNaive
│   │   ├── statistical.py       # ETS, ARIMA
│   │   ├── ml.py                # LightGBM, XGBoost, Prophet, Hybrid
│   │   ├── ensemble.py          # Simple, Weighted, Stacking, Hierarchical
│   │   └── tuning.py            # Grid/random search, Optuna
│   ├── evaluation/              # Model evaluation
│   │   ├── metrics.py           # WAPE, MAE, RMSE
│   │   ├── cross_validation.py  # Time series CV
│   │   └── plots.py             # Visualization functions
│   └── reconciliation/          # Hierarchical reconciliation
│       ├── hierarchy.py         # Hierarchy definition
│       └── methods.py           # Bottom-up, Top-down, MinTrace
├── tests/                       # Unit tests (37 tests)
└── data/raw/                    # Downloaded datasets
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run benchmarks
python scripts/benchmark_ml.py --n-series 100

# Add dependencies
uv add <package>
uv add --dev <dev-package>
```

See [CLAUDE.md](CLAUDE.md) for detailed project documentation.
