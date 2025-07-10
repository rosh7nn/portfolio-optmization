"""
Analysis utilities for portfolio backtesting framework.
Includes functions for transaction cost analysis, sensitivity analysis, and performance visualization.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

def plot_mn_concentration(results: dict, strategy_name: str, save_dir: str = 'plots') -> None:
    """
    Plot MN75% and MN90% concentration metrics over time.
    
    Args:
        results: Dictionary containing backtest results with 'mn75' and 'mn90' keys
        strategy_name: Name of the strategy (used for plot title and filename)
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(14, 6))
    if 'mn75' in results and 'mn90' in results and not results['mn75'].empty and not results['mn90'].empty:
        results['mn75'].plot(label='MN75%', color='blue')
        results['mn90'].plot(label='MN90%', color='green')
        plt.title(f"Portfolio Concentration Metrics for {strategy_name}", fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of stocks', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        os.makedirs(save_dir, exist_ok=True)
        safe_name = strategy_name.lower().replace(' ', '_')
        plt.savefig(f"{save_dir}/{safe_name}_mn_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved concentration metrics plot for {strategy_name}")
    else:
        print(f"Warning: Concentration metrics not available for {strategy_name}")

def run_transaction_cost_analysis(
    results: dict, 
    rebalance_dates: pd.DatetimeIndex, 
    strategy_name: str,
    transaction_costs: List[float] = None
) -> List[dict]:
    """
    Run transaction cost sensitivity analysis for a strategy.
    
    Args:
        results: Dictionary containing backtest results
        rebalance_dates: Dates when rebalancing occurred
        strategy_name: Name of the strategy
        transaction_costs: List of transaction costs to test (as decimals)
        
    Returns:
        List of performance metrics for each transaction cost scenario
    """
    if transaction_costs is None:
        transaction_costs = [0.0005, 0.001, 0.002]  # 5bps, 10bps, 20bps
        
    tc_metrics = []
    
    for tc in transaction_costs:
        # Apply transaction costs
        tc_portfolio = apply_transaction_costs(
            portfolio_values=results['portfolio_value'],
            turnover_series=results.get('turnover', pd.Series(0, index=results['portfolio_value'].index)),
            rebalance_dates=rebalance_dates,
            tc_rate=tc
        )
        
        # Calculate metrics
        metrics = calculate_performance_metrics(
            portfolio_values=tc_portfolio,
            daily_returns=results['daily_returns'],
            weights_history=results.get('weights')
        )
        metrics['strategy'] = f"{strategy_name} ({int(tc*10000)}bps TC)"
        metrics['transaction_cost_bps'] = int(tc * 10000)
        tc_metrics.append(metrics)
    
    return tc_metrics

def plot_metric_comparison(metrics_df: pd.DataFrame, metric: str, title: str, save_dir: str = 'plots') -> str:
    """
    Plot comparison of a specific metric across strategies.
    
    Args:
        metrics_df: DataFrame containing performance metrics
        metric: Name of the metric to plot
        title: Plot title
        save_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x='strategy', y=metric, data=metrics_df, hue='objective')
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), 
                   textcoords='offset points')
    
    os.makedirs(save_dir, exist_ok=True)
    safe_title = title.lower().replace(' ', '_')
    save_path = f"{save_dir}/{safe_title}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_equity_curve_comparison(all_results: Dict[str, dict], save_dir: str = 'plots') -> str:
    """
    Plot comparison of equity curves for all strategies.
    
    Args:
        all_results: Dictionary mapping strategy names to their results
        save_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(14, 8))
    
    for strategy_name, results in all_results.items():
        if 'portfolio_value' in results and not results['portfolio_value'].empty:
            plt.plot(results['portfolio_value'], label=strategy_name, linewidth=2)
    
    plt.title('Strategy Comparison - Equity Curves', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/strategy_equity_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def run_sensitivity_analysis(
    log_returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    end_date: str,
    rebalance_freq: str = 'M',
    estimation_windows: List[int] = None
) -> List[dict]:
    """
    Run sensitivity analysis for different parameters.
    
    Args:
        log_returns: DataFrame of log returns
        rebalance_dates: Original rebalance dates
        end_date: End date for the backtest
        rebalance_freq: Rebalancing frequency
        estimation_windows: List of estimation windows to test (in days)
        
    Returns:
        List of performance metrics for each parameter combination
    """
    if estimation_windows is None:
        estimation_windows = [126, 252, 504]  # 6m, 1y, 2y
        
    all_metrics = []
    
    for window in estimation_windows:
        print(f"\nTesting estimation window: {window} days")
        first_rebalance_date = log_returns.index[window]
        window_rebalance_dates = pd.date_range(
            start=first_rebalance_date, 
            end=end_date, 
            freq=rebalance_freq
        )
        
        # Test with one strategy for sensitivity
        try:
            results = run_backtest(
                log_returns=log_returns,
                rebalance_dates=window_rebalance_dates,
                model_type='arima_garch',
                objective='gmv'
            )
            
            metrics = calculate_performance_metrics(
                portfolio_values=results['portfolio_value'],
                daily_returns=results['daily_returns']
            )
            metrics['strategy'] = f"EstWindow_{window}d"
            metrics['sensitivity_param'] = 'estimation_window'
            metrics['param_value'] = window
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error running sensitivity analysis for window {window}: {str(e)}")
    
    return all_metrics

def generate_final_report(metrics_df: pd.DataFrame, save_dir: str = 'reports') -> str:
    """
    Generate a final performance report with key metrics.
    
    Args:
        metrics_df: DataFrame containing performance metrics
        save_dir: Directory to save the report
        
    Returns:
        Path to the saved report
    """
    # Select and order columns
    columns = ['strategy', 'objective', 'Total Return', 'Annualized Return', 
              'Sharpe Ratio', 'Max Drawdown', 'Information Ratio']
    columns = [col for col in columns if col in metrics_df.columns]
    
    # Format the DataFrame
    final_table = metrics_df[columns].copy()
    final_table = final_table.sort_values(by='Sharpe Ratio', ascending=False)
    
    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    report_path = f"{save_dir}/final_performance_report.csv"
    final_table.to_csv(report_path, index=False)
    
    return report_path
