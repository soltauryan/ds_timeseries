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

### Phase 4: Extended Models ✅ COMPLETE
- [x] CatBoost forecaster (native categorical handling)
- [x] Direct forecasting mode (separate model per horizon step)
- [x] Neural network wrappers via NeuralForecast:
  - N-BEATS (M4 winner, interpretable)
  - NHITS (hierarchical interpolation)
  - DeepAR (Amazon probabilistic)
  - TFT (Temporal Fusion Transformer, Google SOTA)
  - LSTM baseline
  - NeuralEnsemble (combine multiple NN architectures)
  - AutoNeuralForecaster (Optuna-based tuning)
- [x] DRFAM Ensemble (M5 1st place: Direct + Recursive averaging)
- [x] MultiLevelPoolingEnsemble
- [x] M5WinnerEnsemble (full 220-model architecture)

#### Available Model Classes
| Category | Models |
|----------|--------|
| Baselines | NaiveForecaster, MovingAverageForecaster, SeasonalNaiveForecaster |
| Statistical | ETSForecaster |
| Gradient Boosting | LightGBMForecaster, XGBoostForecaster, CatBoostForecaster |
| Prophet | ProphetForecaster, ProphetXGBoostHybrid |
| Neural (optional) | NBEATSForecaster, NHITSForecaster, DeepARForecaster, TFTForecaster, LSTMForecaster |
| Ensembles | SimpleEnsemble, WeightedEnsemble, StackingEnsemble, HierarchicalEnsemble, DRFAMEnsemble, M5WinnerEnsemble |

#### Direct vs Recursive Forecasting
```python
# Recursive (default): predict step by step, use predictions as features
model = LightGBMForecaster(strategy="recursive")

# Direct: separate model for each horizon step (M5 winner approach)
model = LightGBMForecaster(strategy="direct")
model.fit(train, horizon=4)  # Must specify horizon

# DRFAM: averages both strategies (M5 1st place)
ensemble = DRFAMEnsemble(model_class=LightGBMForecaster, use_direct=True, use_recursive=True)
```

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
catboost               # Gradient boosting with native categoricals
```

### Optional
```
prophet                # Holiday/seasonality (slower, use sparingly)
neuralforecast         # Neural networks (N-BEATS, NHITS, TFT, DeepAR)
pytorch-forecasting    # Deep learning approaches
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
│       │   ├── ml.py           # LightGBM, XGBoost, CatBoost, Prophet
│       │   ├── neural.py       # N-BEATS, NHITS, TFT, DeepAR (optional)
│       │   ├── ensemble.py     # Simple, Weighted, Stacking, DRFAM, M5Winner
│       │   └── tuning.py       # Hyperparameter tuning
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

## Presenting to Leadership (Non-Technical Stakeholders)

Leadership wants to see **impact**, not technical details. Focus on business value and intuitive visuals.

### Recommended Visualizations

#### 1. Before/After Forecast Accuracy Chart
Show forecast vs actuals for a few key products. Use simple line charts.
```python
from ds_timeseries.evaluation import plot_forecast

# Pick a high-volume product leadership knows
fig = plot_forecast(actuals, forecasts, series_id="TOP_PRODUCT_001", n_history=26)
fig.suptitle("Forecast vs Actual - Top Product (Last 6 Months)", fontsize=14)
```
**Key message**: "Our new model follows the actual pattern much more closely"

#### 2. Accuracy Improvement Summary (One Number)
```python
# WAPE is intuitive: "On average, our forecast is X% off from actual"
old_wape = 0.45  # Previous model or naive baseline
new_wape = 0.37  # Your new model

improvement = (old_wape - new_wape) / old_wape * 100
print(f"Forecast accuracy improved by {improvement:.0f}%")
# "Forecast accuracy improved by 18%"
```

#### 3. Dollar Impact Visualization
Convert forecast error to dollars. Leadership cares about money.
```python
avg_order_value = 500  # dollars
annual_units = 100000
error_reduction = old_wape - new_wape  # e.g., 0.08

annual_savings = avg_order_value * annual_units * error_reduction
print(f"Estimated annual savings: ${annual_savings:,.0f}")
```

#### 4. Inventory Impact Chart
Show projected inventory reductions or stockout improvements.
```
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Stockouts/month | 45 | 32 | -29% |
| Excess inventory | $2.1M | $1.7M | -19% |
| Forecast WAPE | 45% | 37% | +18% accuracy |
```

#### 5. Simple Model Comparison Bar Chart
```python
import matplotlib.pyplot as plt

models = ["Current System", "Moving Average", "New ML Model"]
accuracy = [55, 63, 73]  # % accuracy (100 - WAPE*100)

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracy, color=["gray", "lightblue", "green"])
plt.ylabel("Forecast Accuracy (%)", fontsize=12)
plt.title("Forecast Accuracy Comparison", fontsize=14)
plt.ylim(0, 100)

# Add value labels
for bar, val in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val}%", ha="center")

plt.tight_layout()
plt.savefig("model_comparison_leadership.png", dpi=150)
```

### Key Talking Points for Leadership

1. **"We reduced forecast error by X%"** - Single headline metric
2. **"This translates to $Y in potential savings"** - Business impact
3. **"The model automatically detects seasonal patterns"** - Intelligence
4. **"We can forecast 4 weeks ahead with Z% accuracy"** - Actionable timeline
5. **"Here's how it predicted [recent event] correctly"** - Concrete example

### What NOT to Show Leadership
- WAPE/MASE/RMSE formulas
- Model architecture diagrams
- Code snippets
- Training/validation curves
- Feature importance lists (unless they ask)

### Dashboard Recommendations

For ongoing monitoring, create a simple dashboard with:

1. **Traffic Light Summary**
   - 🟢 Green: WAPE < 35% (good)
   - 🟡 Yellow: WAPE 35-50% (acceptable)
   - 🔴 Red: WAPE > 50% (needs attention)

2. **Week-over-Week Trend**
   - Simple line chart of accuracy over time
   - Shows the model is stable/improving

3. **Top 10 Products Table**
   - Product name, Last week's forecast, Actual, Difference
   - Familiar format leadership can scan quickly

4. **Exception Report**
   - Products where forecast was way off
   - Actionable for operations team

### Tools for Leadership Dashboards
- **Streamlit**: Quick Python dashboards
- **Power BI/Tableau**: If company already uses these
- **Google Sheets**: Simple, shareable, no IT involvement
- **Matplotlib exports to PPT**: For monthly reviews

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
| 2026-02-03 | Add CatBoost alongside LightGBM/XGBoost | Native categorical handling, robust to overfitting |
| 2026-02-03 | Implement direct + recursive forecasting | M5 1st place averaged both strategies (DRFAM) |
| 2026-02-03 | Add NeuralForecast models as optional | N-BEATS, NHITS, TFT for users who need neural networks |
| 2026-02-03 | M5WinnerEnsemble implements 220-model arch | Full reproduction of M5 1st place approach |
| 2026-02-03 | Add intermittent demand models (Croston, SBA, TSB) | Critical for spare parts/retail with >50% zeros |
| 2026-02-03 | Add demand classification (Syntetos-Boylan scheme) | Automatically recommend models based on data characteristics |
| 2026-02-03 | Add additional metrics (RMSSE, sMAPE, Winkler) | Complete M5 metric suite, interval evaluation |
| 2026-02-03 | Add data validation and leakage detection | Prevent common mistakes in production |

## References

- [Nixtla StatsForecast](https://github.com/Nixtla/statsforecast)
- [M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [Forecasting: Principles and Practice (Hyndman)](https://otexts.com/fpp3/)
- [Hierarchical Forecasting Paper](https://robjhyndman.com/papers/mint.pdf)
