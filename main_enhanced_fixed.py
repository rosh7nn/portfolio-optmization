"""
Enhanced main script with comprehensive analysis features including:
- Benchmark portfolio performance
- MN75% and MN90% concentration metrics
- Transaction cost sensitivity analysis
- Parameter sensitivity analysis
- Consolidated reporting
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
from typing import Dict, Tuple, List, Any, Optional
import analysis_utils
from analysis_utils import (
    plot_mn_concentration,
    run_transaction_cost_analysis,
    run_sensitivity_analysis,
    plot_metric_comparison,
    plot_equity_curve_comparison,
    generate_final_report
)

# Import necessary functions from main.py
from main import (
    prepare_data,
    run_backtest,
    calculate_performance_metrics,
    save_backtest_results,
    apply_transaction_costs
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Configuration
START_DATE = '2023-01-01'  # Start date for backtest
END_DATE = '2025-07-01'    # End date for backtest
ESTIMATION_WINDOW = 252    # 12 months of trading days
REBALANCE_FREQ = 'M'       # Monthly rebalancing
MAX_WEIGHT = 0.1           # Maximum weight per asset
BOOTSTRAP_PATHS = 1000     # Number of bootstrap paths for Monte Carlo
FORECAST_HORIZON = 21      # 1 month forecast horizon (21 trading days)

def main(use_cached_data: bool = True):
    """
    Enhanced main function with comprehensive analysis features.
    """
    # Initialize dictionaries to store results and metrics for all strategies
    all_results = {}
    all_metrics_list = []
    
    print("Preparing data...")
    _, log_returns, _ = prepare_data(use_cached=use_cached_data)
    
    # Set first rebalance date after estimation window
    first_rebalance_date = log_returns.index[ESTIMATION_WINDOW]
    rebalance_dates = pd.date_range(start=first_rebalance_date, end=END_DATE, freq=REBALANCE_FREQ)
    
    # 1. Calculate benchmark (equal-weighted) portfolio
    print("\n1. Calculating benchmark portfolio performance...")
    equal_weight_returns = log_returns.mean(axis=1).loc[first_rebalance_date:]
    equal_weight_cum = (1 + equal_weight_returns).cumprod()
    
    benchmark_metrics = calculate_performance_metrics(
        portfolio_values=equal_weight_cum,
        daily_returns=equal_weight_returns
    )
    benchmark_metrics['strategy'] = 'Equal-Weight Benchmark'
    all_metrics_list.append(benchmark_metrics)
    
    # Store benchmark results
    benchmark_results = {
        'portfolio_value': equal_weight_cum,
        'daily_returns': equal_weight_returns,
        'metrics': benchmark_metrics
    }
    all_results['Benchmark'] = benchmark_results
    
    # Define strategy configurations
    strategies = [
        ('arima_garch', 'gmir', 'ARIMA-GARCH GMIR'),
        ('arima_garch', 'gmv', 'ARIMA-GARCH GMV'),
        ('xgboost', 'gmir', 'XGBoost GMIR'),
        ('xgboost', 'gmv', 'XGBoost GMV'),
    ]
    
    # 2. Run backtests for each strategy
    for model_type, objective, strategy_name in strategies:
        print(f"\n{'='*50}")
        print(f"2. Running backtest for {strategy_name}...")
        print(f"Model: {model_type}, Objective: {objective.upper()}")
        print(f"{'='*50}")
        
        # Run backtest
        results = run_backtest(
            log_returns=log_returns,
            rebalance_dates=rebalance_dates,
            model_type=model_type,
            objective=objective
        )
        
        # 3. Calculate performance metrics with benchmark returns
        print(f"\n3. Calculating performance metrics for {strategy_name}...")
        metrics = calculate_performance_metrics(
            portfolio_values=results['portfolio_value'],
            daily_returns=results['daily_returns'],
            benchmark_returns=equal_weight_returns,
            weights_history=results.get('weights')
        )
        
        # Add strategy metadata to metrics
        metrics['strategy'] = strategy_name
        metrics['model'] = model_type
        metrics['objective'] = objective
        
        # Store results and metrics
        all_results[strategy_name] = results
        all_metrics_list.append(metrics)
        results['metrics'] = metrics
        
        # 4. Save detailed results
        print(f"\n4. Saving results for {strategy_name}...")
        save_backtest_results(results, strategy_name, first_rebalance_date)
        
        # 5. Generate concentration metrics plots
        print(f"\n5. Generating concentration metrics for {strategy_name}...")
        plot_mn_concentration(results, strategy_name)
        
        # 6. Run transaction cost sensitivity analysis
        print(f"\n6. Running transaction cost analysis for {strategy_name}...")
        tc_metrics = run_transaction_cost_analysis(
            results=results,
            rebalance_dates=rebalance_dates,
            strategy_name=strategy_name
        )
        all_metrics_list.extend(tc_metrics)
        
        # Print metrics
        print(f"\n{strategy_name} Performance:")
        print("-" * 50)
        for metric, value in metrics.items():
            if metric not in ['strategy', 'model', 'objective']:
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
    
    # 7. Run sensitivity analysis
    print("\n7. Running parameter sensitivity analysis...")
    sensitivity_metrics = run_sensitivity_analysis(
        log_returns=log_returns,
        rebalance_dates=rebalance_dates,
        end_date=END_DATE,
        rebalance_freq=REBALANCE_FREQ
    )
    all_metrics_list.extend(sensitivity_metrics)
    
    # 8. Generate final analysis and reports
    print("\n8. Generating final analysis and reports...")
    metrics_df = pd.DataFrame(all_metrics_list)
    
    # Save combined metrics
    metrics_df.to_csv('data/all_metrics.csv', index=False)
    print("\n✓ Saved combined metrics to data/all_metrics.csv")
    
    # Generate comparison plots
    plot_metric_comparison(metrics_df, 'Total Return', 'Total Return Comparison')
    plot_metric_comparison(metrics_df, 'Sharpe Ratio', 'Sharpe Ratio Comparison')
    plot_metric_comparison(metrics_df, 'Max Drawdown', 'Max Drawdown Comparison')
    plot_equity_curve_comparison(all_results)
    
    # Generate final performance report
    report_path = generate_final_report(metrics_df)
    print(f"\n✓ Generated final performance report: {report_path}")
    
    # Print final comparison table
    print("\nFinal Strategy Comparison:")
    print("-" * 100)
    columns = ['strategy', 'Total Return', 'Annualized Return', 
               'Sharpe Ratio', 'Max Drawdown', 'Information Ratio']
    columns = [col for col in columns if col in metrics_df.columns]
    print(metrics_df[columns].sort_values('Sharpe Ratio', ascending=False).to_string(index=False))
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run enhanced portfolio backtest with comprehensive analysis.')
    parser.add_argument('--no-cache', action='store_true', help='Force download fresh data')
    args = parser.parse_args()
    
    # Run the enhanced backtest
    main(use_cached_data=not args.no_cache)
