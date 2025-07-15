# Portfolio Optimization System

A comprehensive portfolio optimization framework that combines multiple forecasting models with advanced optimization strategies for NIFTY 50 stocks.

## Features

- ARIMA-GARCH and XGBoost forecasting models
- Mean-Variance optimization with GMIR and GMV objectives
- Monthly rebalancing strategy
- Performance metrics calculation
- Professional plotting and visualization

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install the package:
```bash
pip install -e .
```

## Usage

Run the portfolio optimization:
```bash
python3 run.py
```

Or use the command line interface:
```bash
portfolio-optimize
```

## Configuration

The system uses the following configuration parameters:
- Start Date: 2023-01-01
- End Date: 2025-07-01
- Rebalance Frequency: Monthly
- Estimation Window: 252 trading days
- Maximum Weight per Asset: 10%
