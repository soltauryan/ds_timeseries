# DS Time Series - Project Manifesto

## Project Overview

A robust Python library for hierarchical time series forecasting, specifically designed to predict weekly sales at the Customer-Material level with fiscal calendar awareness.

## Core Objectives

### Business Goal
Generate accurate weekly sales forecasts for Customer-Material pairs that are reconcilable across the hierarchy.

### Key Business Logic

1. **Hierarchical Structure**: Sales forecasts must be reconcilable (sum of Material forecasts = Customer forecast = Total forecast)
2. **Fiscal Calendar**: Data follows fiscal month patterns (4-4-5 or similar), NOT standard calendar months
3. **Hockey Stick Pattern**: Expect sales surges at end of fiscal periods - models must capture this
4. **Intermittent Demand**: Handle zeros in data gracefully (common at granular Customer-Material level)

## Technical Requirements

### Architecture Principles
- **Modular Design**: Functional library structure, NOT monolithic scripts
- **Separation of Concerns**: Data, models, evaluation, and reconciliation as separate modules
- **Reproducibility**: All experiments must be reproducible with fixed random seeds

### Validation Strategy
- **Time Series Cross-Validation**: Rolling window / expanding window approach
- **NO RANDOM SPLITS**: Strictly chronological train/test splits to prevent data leakage
- **Walk-Forward Validation**: Train on past, predict future, roll forward

### Primary Metrics
- **WAPE** (Weighted Absolute Percentage Error): Primary metric - handles intermittent demand well
- **MAE** (Mean Absolute Error): Secondary metric - interpretable in business units
- Avoid MAPE for zero-heavy data (division by zero issues)

```python
# WAPE Formula
WAPE = sum(|actual - forecast|) / sum(actual)
```

## Phased Development Approach

### Phase 1: Baselines ✅ COMPLETE
- [x] Moving Average (simple, weighted)
- [x] Seasonal Naive (same week last year)
- [x] Exponential Smoothing (ETS)
- [x] Time Series Cross-Validation
- [x] Establish benchmark performance

#### Baseline Results (M5 Sample, 100 series, 3-fold CV, 4-week horizon)
| Model | WAPE | MAE | Notes |
|-------|------|-----|-------|
| **ETSForecaster** | **59.39%** | **5.32** | Best baseline, auto-selects ETS spec |
| MovingAverage(8) | 63.04% | 5.65 | Simple, fast |
| MovingAverage(4) | 63.47% | 5.69 | |
| NaiveForecaster | 65.17% | 5.84 | Last value repeated |
| SeasonalNaive(52) | 95.28% | 8.54 | Poor - needs more history |

**Insight**: Seasonal Naive underperforms because M5 data has only ~5 years and patterns shift over time. ETS adapts better.

### Phase 2: Advanced Models ✅ COMPLETE
- [x] Feature Engineering
  - Lag features (1, 4, 13, 52 weeks)
  - Rolling statistics (mean, std over 4, 13, 26, 52 weeks)
  - Fiscal calendar features (5-4-4 pattern, Nov start)
  - `is_fiscal_month_end` flag (critical for hockey stick)
  - Cyclical encoding (week_sin, week_cos)
- [x] ML Models: LightGBM (Tweedie), XGBoost
- [x] Prophet + Prophet-XGBoost Hybrid

#### Phase 2 Benchmark Results (M5 Sample, 50 series, 4-week horizon)
| Model | WAPE | MAE | Notes |
|-------|------|-----|-------|
| **XGBoost** | **36.78%** | **5.55** | Best overall |
| LightGBM (Tweedie) | 37.16% | 5.60 | Fast, handles zeros well |
| ETS | 39.95% | 6.03 | Best statistical baseline |
| MovingAvg(4) | 52.42% | 7.91 | Simple baseline |
| Prophet | 56.30% | 8.49 | Better for trend, not granular data |
| Prophet+XGB | 58.52% | 8.83 | Hybrid underperforms on this data |

**Key insight**: ML models beat ETS by 7-8% relative improvement. XGBoost slightly edges out LightGBM.

#### Fiscal Calendar Configuration
```python
# November start, 5-4-4 pattern (5 weeks, 4 weeks, 4 weeks per quarter)
from ds_timeseries.features import FiscalCalendarConfig, engineer_features

config = FiscalCalendarConfig(fiscal_year_start_month=11, week_pattern="5-4-4")
# Pre-generated calendar: data/raw/fiscal_calendar_544_nov.parquet
```

#### Feature Groups (33 features)
| Group | Features |
|-------|----------|
| Fiscal | fiscal_year, fiscal_quarter, fiscal_month, fiscal_week, fiscal_week_in_month, fiscal_week_in_quarter, is_fiscal_month_end, is_fiscal_quarter_end, is_fiscal_year_end |
| Calendar | calendar_year, calendar_month, calendar_week, calendar_quarter, week_sin, week_cos, month_sin, month_cos |
| Lags | y_lag_1, y_lag_4, y_lag_13, y_lag_52 |
| Rolling | y_roll_4_mean, y_roll_4_std, y_roll_13_mean, y_roll_13_std |
| Diff | y_diff_1, y_diff_4, y_diff_52 |
| Pct Change | y_pct_1, y_pct_52 |
| Expanding | y_expanding_mean, y_expanding_std |

### Phase 3: Hierarchical Reconciliation & Ensembles ✅ COMPLETE
- [x] Bottom-Up reconciliation
- [x] Top-Down reconciliation
- [x] MinTrace (OLS/WLS) optimal reconciliation
- [x] Ensemble Models:
  - SimpleEnsemble (average)
  - WeightedEnsemble (CV-optimized weights)
  - StackingEnsemble (meta-learner)
  - HierarchicalEnsemble (M5-winning approach)
- [x] Hyperparameter Tuning:
  - grid_search_cv, random_search_cv
  - tune_lightgbm, tune_xgboost
  - Optuna support (if installed)
- [x] Visualization:
  - plot_forecast, plot_forecast_grid
  - plot_model_comparison
  - plot_residuals
  - plot_feature_importance

#### Ensemble Results
| Model | WAPE | Notes |
|-------|------|-------|
| XGBoost | 36.78% | Best single model |
| Weighted Ensemble | 36.78% | Learned 100% XGBoost weight |
| LightGBM | 37.16% | |
| Simple Ensemble | 37.17% | Average of 3 models |

## Recommended Libraries

### Core Dependencies
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
```

### Time Series Specific
```
statsforecast          # Fast statistical models + hierarchical support (Nixtla)
mlforecast             # ML models for time series (Nixtla)
hierarchicalforecast   # Reconciliation methods (Nixtla)
lightgbm               # Gradient boosting
xgboost                # Gradient boosting
```

### Optional
```
prophet                # Holiday/seasonality (slower, use sparingly)
pytorch-forecasting    # Deep learning approaches (Phase 4+)
```

## Folder Structure

```
ds_timeseries/
├── CLAUDE.md              # This file - project manifesto
├── README.md              # User-facing documentation
├── pyproject.toml         # Package configuration
├── src/
│   └── ds_timeseries/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py      # Data loading utilities
│       │   ├── toy_datasets.py # Download sample datasets
│       │   └── preprocessing.py # Data cleaning, aggregation
│       ├── features/
│       │   ├── __init__.py
│       │   ├── calendar.py     # Fiscal calendar features
│       │   ├── lags.py         # Lag and rolling features
│       │   └── engineering.py  # Feature pipelines
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract base forecaster
│       │   ├── baselines.py    # Naive, Moving Average, Seasonal Naive
│       │   ├── statistical.py  # ETS, ARIMA
│       │   └── ml.py           # LightGBM, XGBoost wrappers
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py      # WAPE, MAE, etc.
│       │   ├── cross_validation.py  # Time series CV
│       │   └── backtesting.py  # Walk-forward testing
│       ├── reconciliation/
│       │   ├── __init__.py
│       │   ├── hierarchy.py    # Hierarchy definition
│       │   └── methods.py      # Bottom-up, MinTrace, etc.
│       └── utils/
│           ├── __init__.py
│           └── logging.py
├── notebooks/
│   └── 01_exploratory.ipynb
├── tests/
│   └── ...
└── data/
    └── raw/                # Downloaded toy datasets
```

## Development Dataset: M5 Sample

**Primary dataset**: `m5_sample` - Walmart hierarchical sales from M5 competition

### Hierarchy Structure
```
Total
├── State (CA, TX, WI)
│   └── Store (CA_1, CA_2, TX_1, etc.)
│       └── Category (FOODS, HOUSEHOLD, HOBBIES)
│           └── Department (FOODS_1, FOODS_2, FOODS_3, etc.)
│               └── Item (unique SKU)
```

### Schema Mapping to Our Domain
| M5 Column   | Our Domain Equivalent |
|-------------|----------------------|
| `item_id`   | Material ID          |
| `store_id`  | Customer/Location    |
| `cat_id`    | Product Category     |
| `dept_id`   | Product Department   |
| `state_id`  | Region               |

### Usage
```python
from ds_timeseries.data import download_toy_dataset

df = download_toy_dataset("m5_sample")
# Returns: unique_id, ds, y, item_id, store_id, cat_id, dept_id, state_id
```

## Coding Conventions

### Data Format
All time series data should follow this schema:
```python
# Required columns
- unique_id: str  # Customer-Material identifier (e.g., "CUST001_MAT123")
- ds: datetime    # Date (week start date)
- y: float        # Target variable (sales quantity/value)

# Hierarchy columns (when applicable)
- customer_id: str
- material_id: str
- category: str (optional)
```

### Model Interface
All models should implement:
```python
class BaseForecaster(ABC):
    def fit(self, df: pd.DataFrame) -> 'BaseForecaster': ...
    def predict(self, horizon: int) -> pd.DataFrame: ...
    def fit_predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame: ...
```

## Running Commands

```bash
# Create virtual environment with uv
uv venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Install in development mode with uv
uv pip install -e ".[dev]"

# Or use uv sync for lockfile-based installs
uv sync --dev

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_metrics.py -v

# Add new dependencies
uv add pandas numpy  # runtime deps
uv add --dev pytest  # dev deps
```

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-02 | Use Nixtla stack (statsforecast, mlforecast) | Fastest implementations, native hierarchical support |
| 2026-02-02 | WAPE as primary metric | Handles zeros, weighted by volume |
| 2026-02-02 | Fiscal calendar as first-class feature | Business requirement, drives hockey stick patterns |
| 2026-02-02 | Use m5_sample as primary dev dataset | Gold standard hierarchical retail data, maps to Customer-Material pattern |
| 2026-02-02 | ETS as best baseline (59.39% WAPE) | Beat to target for Phase 2 ML models |
| 2026-02-02 | XGBoost best overall (36.78% WAPE) | 7% improvement over ETS baseline |
| 2026-02-02 | LightGBM with Tweedie objective | Handles intermittent demand (zeros) well |
| 2026-02-02 | Prophet underperforms on granular data | Better for aggregate-level forecasting |

## References

- [Nixtla StatsForecast](https://github.com/Nixtla/statsforecast)
- [M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [Forecasting: Principles and Practice (Hyndman)](https://otexts.com/fpp3/)
- [Hierarchical Forecasting Paper](https://robjhyndman.com/papers/mint.pdf)
