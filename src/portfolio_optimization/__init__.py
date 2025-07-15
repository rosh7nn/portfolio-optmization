"""
Portfolio Optimization Package

This package contains modules for portfolio optimization and backtesting.
"""

from portfolio_optimization.backtest import BacktestEngine
from portfolio_optimization.optimizer import PortfolioOptimizer
from portfolio_optimization.settings import (
    MAX_WEIGHT,
    RISK_FREE_RATE,
    START_DATE,
    END_DATE,
    REBALANCE_FREQ,
    ESTIMATION_WINDOW,
    DATA_DIR,
    PLOTS_DIR
)
from .forecasting.models import ARIMAGARCH, XGBoost
