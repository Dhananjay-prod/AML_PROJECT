# S&P 500 Stock Market Predictor

A machine learning model that predicts whether the S&P 500 index will go **up or down** the next trading day. Uses a Random Forest classifier trained on historical price data with rolling window features, evaluated through a walk-forward backtesting system to simulate real trading conditions.

---

## How It Works

1. Downloads S&P 500 historical data via `yfinance` (cached locally as `sp500.csv`)
2. Engineers a binary target — `1` if tomorrow's close is higher than today's, `0` otherwise
3. Builds rolling window features across 5 time horizons (2, 5, 60, 250, 1000 days)
4. Trains a Random Forest on expanding historical windows (walk-forward backtest)
5. Predicts using `predict_proba` with a 0.6 confidence threshold to reduce false positives
6. Evaluates using **precision score** — the right metric when false positives are costly

---

## Project Structure

```
├── main.py        # Full pipeline: data loading, feature engineering, training, backtesting
├── sp500.csv      # Cached historical data (auto-generated on first run)
└── README.md
```

---

## Features

| Feature | Description |
|---|---|
| `Close_Ratio_{horizon}` | Current close price divided by rolling mean close over N days |
| `Trend_{horizon}` | Sum of past target values (up/down days) over N-day window |

Horizons used: **2, 5, 60, 250, 1000** days

---

## Model

- **Algorithm**: `RandomForestClassifier` (scikit-learn)
- **Trees**: 200 estimators
- **Min samples split**: 50
- **Prediction threshold**: 0.6 (via `predict_proba` — only predicts "up" when confidence ≥ 60%)
- **Evaluation metric**: Precision Score

---

## Backtesting

Walk-forward backtesting is used to simulate real trading — the model never sees future data during training.

```
Start: first 2500 days as training set
Step:  250 days at a time (roughly 1 trading year)

[-------- train --------][- test -]
[--------- train ----------][- test -]
[------------ train ------------][- test -]
...
```

This approach gives a realistic estimate of how the model would perform if deployed over time.

---

## Setup

```bash
pip install yfinance pandas scikit-learn
```

```bash
python main.py
```

On first run, historical S&P 500 data is downloaded and saved to `sp500.csv`. Subsequent runs use the cached file.

---

## Results

The model is evaluated using **precision** — out of all the days it predicted "market goes up", how many were actually correct. A random baseline would sit around 53-54% (markets go up slightly more than down historically). The 0.6 threshold filters out low-confidence predictions to push precision higher at the cost of fewer total predictions.

---

## Key Concepts Demonstrated

- Random Forest for binary classification
- Time series feature engineering with rolling windows
- Walk-forward backtesting (no data leakage)
- Threshold tuning via `predict_proba`
- Precision-focused model evaluation
- yfinance data fetching with local CSV caching
Dhananjay Kumar 1BI20AI009
Harshil Pathria 1BI20AI016
