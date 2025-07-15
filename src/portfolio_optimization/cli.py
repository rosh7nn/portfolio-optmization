"""
Portfolio Optimization Command Line Interface

This module provides a command line interface for running portfolio optimization.
"""

import argparse
from datetime import datetime
from typing import Dict, List
import warnings

from .backtesting.engine import BacktestEngine
from .config import (
    START_DATE, END_DATE, REBALANCE_FREQ,
    NIFTY50_TICKERS, DATA_DIR, PLOTS_DIR
)

warnings.filterwarnings('ignore')

def main():
    """
    Main entry point for portfolio optimization.
    
    Parses command line arguments and runs the backtest.
    """
    parser = argparse.ArgumentParser(
        description='Portfolio Optimization System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--no-cache', 
        action='store_true',
        help='Force download fresh data'
    )
    parser.add_argument(
        '--start-date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=START_DATE,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=END_DATE,
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Run backtest
    engine = BacktestEngine(use_cached_data=not args.no_cache)
    
    # Prepare data
    _, log_returns, rebalance_dates = engine.prepare_data()
    
    # Define strategies
    strategies = [
        ('arima', 'gmir', 'ARIMA-GARCH GMIR'),
        ('arima', 'gmv', 'ARIMA-GARCH GMV'),
        ('xgboost', 'gmir', 'XGBoost GMIR'),
        ('xgboost', 'gmv', 'XGBoost GMV'),
    ]
    
    # Run backtest for each strategy
    all_results = {}
    for model_type, objective, strategy_name in strategies:
        print(f"\n{'='*80}\nRunning {strategy_name} Backtest")
        results = engine.run_backtest(
            log_returns=log_returns,
            rebalance_dates=rebalance_dates,
            model_type=model_type,
            objective=objective
        )
        
        # Calculate metrics
        metrics = engine.calculate_performance_metrics(
            portfolio_values=results['portfolio_value'],
            daily_returns=results['daily_returns'],
            weights_history=results['weights']
        )
        
        # Plot results
        engine.plot_results(results, strategy_name, strategy_name)
        
        # Store results
        all_results[strategy_name] = {
            'results': results,
            'metrics': metrics
        }
        
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value}")
    
    # Print summary
    print("\nBacktest Summary:")
    print("=" * 80)
    for strategy, data in all_results.items():
        print(f"\n{strategy}:")
        print("-" * 80)
        metrics = data['metrics']
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
