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

### 5. Prophet

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

### 6. Prophet + XGBoost Hybrid (Advanced)

Prophet captures trend/seasonality, XGBoost learns residual patterns.

```python
from ds_timeseries.models import ProphetXGBoostHybrid

model = ProphetXGBoostHybrid(feature_config=feature_config)
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

## Applying to Your Own Data

### Step 1: Prepare Your Data

```python
import pandas as pd

# Your data should have: unique_id, ds, y
df = pd.DataFrame({
    "unique_id": [...],  # e.g., "CUST001_MAT123"
    "ds": pd.to_datetime([...]),  # Week start dates
    "y": [...],  # Sales values
})

# Optional: Add hierarchy columns
df["customer_id"] = ...
df["material_id"] = ...
```

### Step 2: Configure Features for Your Domain

```python
from ds_timeseries.features import FeatureConfig, FiscalCalendarConfig

# Customize for your business
feature_config = FeatureConfig(
    # Adjust lags based on your data frequency and history length
    lags=[1, 4, 13, 52] if len(df) > 52 else [1, 4],

    # Your fiscal calendar
    fiscal_config=FiscalCalendarConfig(
        fiscal_year_start_month=11,  # Your FY start
        week_pattern="5-4-4",        # Your week pattern
    ),
)
```

### Step 3: Train and Evaluate

```python
from ds_timeseries.models import LightGBMForecaster
from ds_timeseries.evaluation import cross_validate
from ds_timeseries.evaluation.metrics import wape

# Train
model = LightGBMForecaster(feature_config=feature_config)

# Evaluate with CV
results = cross_validate(model, df, n_folds=3, horizon=4)
print(f"WAPE: {wape(results['y'], results['yhat']):.2%}")
```

---

## Benchmark Results (M5 Sample)

| Model | WAPE | MAE | Notes |
|-------|------|-----|-------|
| **LightGBM (Tweedie)** | **41%** | **4.27** | Best overall |
| ETS | 41% | 4.26 | Strong baseline |
| XGBoost | 42% | 4.35 | Competitive |
| MovingAverage(8) | 45% | 4.67 | Simple baseline |

---

## Project Structure

```
ds_timeseries/
├── src/ds_timeseries/
│   ├── data/           # Data loading (M5, synthetic)
│   ├── features/       # Feature engineering, fiscal calendar
│   ├── models/         # Baselines, ETS, LightGBM, XGBoost, Prophet
│   ├── evaluation/     # Metrics (WAPE, MAE), cross-validation
│   └── reconciliation/ # Hierarchical reconciliation (Phase 3)
├── scripts/            # Benchmark scripts
├── tests/              # Unit tests
└── data/raw/           # Downloaded datasets
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
