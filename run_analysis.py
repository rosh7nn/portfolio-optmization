#!/usr/bin/env python3
"""
Portfolio Optimization Analysis Pipeline

This script runs the complete analysis pipeline, including:
1. Data preparation and preprocessing
2. Model training and parameter tuning
3. Backtesting with different strategies
4. Performance evaluation and visualization
"""
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from config import (
    START_DATE, END_DATE, ESTIMATION_WINDOW, 
    REBALANCE_FREQ, DATA_DIR, PLOTS_DIR, TICKERS,
    ARIMA_ORDER, XGB_LAGS, RISK_FREE_RATE
)
from data_handling import download_data, preprocess_data, generate_rebalance_dates
from backtesting.engine import run_all_strategies
from visualization.model_comparison import compare_models
from visualization.plots import plot_transaction_cost_sensitivity
from utils.model_selection import (
    tune_arima_parameters, 
    tune_xgb_parameters,
    save_model_parameters
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run portfolio optimization analysis.')
    parser.add_argument('--start-date', type=str, default=START_DATE,
                      help=f'Start date in YYYY-MM-DD format (default: {START_DATE})')
    parser.add_argument('--end-date', type=str, default=END_DATE,
                      help=f'End date in YYYY-MM-DD format (default: {END_DATE})')
    parser.add_argument('--estimation-window', type=int, default=ESTIMATION_WINDOW,
                      help=f'Number of days for estimation window (default: {ESTIMATION_WINDOW})')
    parser.add_argument('--rebalance-freq', type=str, default=REBALANCE_FREQ,
                      help=f'Rebalancing frequency (default: {REBALANCE_FREQ})')
    parser.add_argument('--tune-models', action='store_true',
                      help='Tune model parameters (ARIMA and XGBoost)')
    parser.add_argument('--no-cache', action='store_true',
                      help='Force download fresh data instead of using cached data')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Output directory for results (default: results/)')
    return parser.parse_args()

def setup_directories(output_dir: str) -> None:
    """Create necessary directories for output."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

def tune_models(returns: pd.DataFrame, output_dir: str) -> None:
    """
    Tune model parameters and save the results.
    
    Args:
        returns: DataFrame of asset returns
        output_dir: Output directory for saving results
    """
    print("\n" + "="*50)
    print("TUNING MODEL PARAMETERS")
    print("="*50)
    
    # Tune ARIMA parameters
    print("\nTuning ARIMA parameters...")
    arima_params = tune_arima_parameters(returns)
    save_model_parameters(
        arima_params, 
        os.path.join(output_dir, 'models', 'arima_parameters.json')
    )
    
    # Tune XGBoost parameters
    print("\nTuning XGBoost parameters...")
    xgb_params = tune_xgb_parameters(returns)
    save_model_parameters(
        xgb_params,
        os.path.join(output_dir, 'models', 'xgboost_parameters.json')
    )

def run_analysis(args):
    """Run the complete analysis pipeline."""
    # Setup output directories
    setup_directories(args.output_dir)
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION ANALYSIS")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Estimation Window: {args.estimation_window} days")
    print(f"Rebalance Frequency: {args.rebalance_freq}")
    print(f"Output Directory: {args.output_dir}")
    print("-"*70)
    
    # Step 1: Download and preprocess data
    print("\n1. DOWNLOADING AND PREPROCESSING DATA")
    print("-"*70)
    
    prices = download_data(
        tickers=TICKERS,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cached=not args.no_cache
    )
    
    returns = preprocess_data(prices)
    rebalance_dates = generate_rebalance_dates(
        prices.index, 
        freq=args.rebalance_freq
    )
    
    # Save processed data
    prices.to_csv(os.path.join(args.output_dir, 'data', 'prices.csv'))
    returns.to_csv(os.path.join(args.output_dir, 'data', 'returns.csv'))
    
    # Step 2: Tune model parameters if requested
    if args.tune_models:
        tune_models(returns, args.output_dir)
    
    # Step 3: Run backtests
    print("\n2. RUNNING BACKTESTS")
    print("-"*70)
    
    all_results = run_all_strategies(
        log_returns=returns,
        rebalance_dates=rebalance_dates
    )
    
    # Step 4: Compare models and generate visualizations
    print("\n3. GENERATING VISUALIZATIONS")
    print("-"*70)
    
    # Compare all strategies
    compare_models(
        data_dir=os.path.join(args.output_dir, 'data'),
        save_dir=os.path.join(args.output_dir, 'plots'),
        filename='strategy_comparison.png'
    )
    
    # Generate transaction cost sensitivity plot
    strategies = [
        'arima_garch_gmir', 'arima_garch_gmv',
        'xgboost_gmir', 'xgboost_gmv'
    ]
    
    plot_transaction_cost_sensitivity(
        strategies=strategies,
        data_dir=os.path.join(args.output_dir, 'data'),
        save_path=os.path.join(args.output_dir, 'plots', 'transaction_cost_sensitivity.png')
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")
    print(f"- Data files: {os.path.abspath(os.path.join(args.output_dir, 'data'))}")
    print(f"- Plots: {os.path.abspath(os.path.join(args.output_dir, 'plots'))}")
    print(f"- Model parameters: {os.path.abspath(os.path.join(args.output_dir, 'models'))}")

def main():
    """Main function to run the analysis."""
    args = parse_arguments()
    run_analysis(args)

if __name__ == "__main__":
    main()
