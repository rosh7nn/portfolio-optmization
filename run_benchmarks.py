"""
Script to run and compare benchmark strategies with existing portfolio strategies.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from benchmarks import calculate_benchmark_returns, plot_benchmark_comparison
from main import plot_equity_and_drawdown

def load_strategy_results(strategy_name: str, benchmark_returns: pd.Series = None) -> dict:
    """Load results from a saved strategy and optionally calculate Information Ratio.
    
    Args:
        strategy_name: Name of the strategy to load
        benchmark_returns: Optional Series of benchmark returns for Information Ratio calculation
        
    Returns:
        Dictionary containing portfolio values, returns, and optional Information Ratio
    """
    results_path = Path('data') / f'portfolio_{strategy_name}.csv'
    if not results_path.exists():
        return None
    
    try:
        # Read the CSV file
        data = pd.read_csv(results_path, index_col=0, parse_dates=True)
        
        # If it's a Series (single column), convert to DataFrame
        if isinstance(data, pd.Series):
            portfolio_values = data.rename('portfolio_value')
        else:
            # If it's a DataFrame, ensure we have a portfolio_value column
            if 'portfolio_value' in data.columns:
                portfolio_values = data['portfolio_value']
            else:
                # Take the first column if portfolio_value doesn't exist
                portfolio_values = data.iloc[:, 0].rename('portfolio_value')
        
        # Calculate daily returns
        daily_returns = portfolio_values.pct_change().dropna()
        
        result = {
            'portfolio_value': portfolio_values,
            'daily_returns': daily_returns
        }
        
        # Calculate Information Ratio if benchmark returns are provided
        if benchmark_returns is not None and not daily_returns.empty:
            # Align the dates between strategy and benchmark returns
            common_dates = daily_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                strategy_aligned = daily_returns[common_dates]
                benchmark_aligned = benchmark_returns[common_dates]
                
                # Ensure we have matching dates
                aligned_data = pd.DataFrame({
                    'strategy': strategy_aligned,
                    'benchmark': benchmark_returns[strategy_aligned.index]
                }).dropna()
                
                if len(aligned_data) > 0:
                    strategy_ret = aligned_data['strategy']
                    benchmark_ret = aligned_data['benchmark']
                    
                    # Calculate active returns (strategy - benchmark)
                    active_returns = strategy_ret - benchmark_ret
                    
                    # Calculate annualized excess return (strategy return - benchmark return)
                    excess_return_annualized = (strategy_ret.mean() - benchmark_ret.mean()) * 252
                    
                    # Calculate tracking error (std of active returns, annualized)
                    tracking_error = active_returns.std() * np.sqrt(252)
                    
                    # Calculate Information Ratio and Modified Information Ratio
                    if tracking_error > 1e-10:  # Avoid division by zero
                        information_ratio = excess_return_annualized / tracking_error
                        result['information_ratio'] = information_ratio
                        
                        # Calculate Modified Information Ratio (adjusting for non-normality)
                        from scipy.stats import skew, kurtosis
                        
                        # Calculate higher moments of active returns
                        active_skew = skew(active_returns, nan_policy='omit')
                        active_kurt = kurtosis(active_returns, nan_policy='omit') + 3  # scipy returns excess kurtosis
                        
                        # Adjust IR for skewness and kurtosis
                        # Using the formula: IR_modified = IR * (1 + (skew/6)*IR - ((kurt-3)/24)*IR^2)
                        ir_squared = information_ratio ** 2
                        skew_adj = (active_skew / 6) * information_ratio
                        kurt_adj = ((active_kurt - 3) / 24) * ir_squared
                        
                        modified_ir = information_ratio * (1 + skew_adj - kurt_adj)
                        result['modified_information_ratio'] = modified_ir
                        
                        # Debug output
                        print(f"\nStrategy: {strategy_name}")
                        print(f"  Period: {len(aligned_data)} days")
                        print(f"  Strategy Annualized Return: {strategy_ret.mean() * 252:.2%}")
                        print(f"  Benchmark Annualized Return: {benchmark_ret.mean() * 252:.2%}")
                        print(f"  Excess Return (Annualized): {excess_return_annualized:.2%}")
                        print(f"  Tracking Error (Annualized): {tracking_error:.2%}")
                        print(f"  Information Ratio: {information_ratio:.2f}")
                        print(f"  Modified Information Ratio: {modified_ir:.2f}")
                        print(f"  Active Returns Skewness: {active_skew:.2f}")
                        print(f"  Active Returns Kurtosis: {active_kurt:.2f}")
                    else:
                        print(f"Warning: Tracking error too small for {strategy_name}")
                else:
                    print(f"Warning: No overlapping dates found for {strategy_name}")
        
        return result
        
    except Exception as e:
        print(f"Error loading strategy {strategy_name}: {str(e)}")
        return None

def combine_strategy_benchmark_results(benchmark_results: dict, strategy_results: dict) -> dict:
    """Combine benchmark and strategy results into a single dictionary."""
    combined = {}
    
    # Add benchmarks
    for strategy in ['buy_hold', 'equal_weight']:
        if strategy in benchmark_results:
            combined[f'benchmark_{strategy}'] = benchmark_results[strategy]
    
    # Add strategies
    for strategy, result in strategy_results.items():
        if result is not None and 'portfolio_value' in result:
            combined[f'strategy_{strategy}'] = result
    
    return combined

def plot_all_strategies(benchmark_results: dict, strategy_results: dict, save_path: str = None):
    """Plot comparison of all strategies with performance table."""
    plt.figure(figsize=(16, 9))
    
    # Define consistent colors and styles
    colors = {
        'buy_hold': '#3498db',
        'equal_weight': '#2ecc71',
        'arima_garch_gmir': '#e74c3c',
        'arima_garch_gmv': '#f39c12',
        'xgboost_gmir': '#9b59b6',
        'xgboost_gmv': '#1abc9c'
    }
    
    all_metrics = []
    strategy_names = []
    
    # Combine all results
    combined_results = combine_strategy_benchmark_results(benchmark_results, strategy_results)
    
    # Plot all series
    for key, result in combined_results.items():
        if result is None or 'portfolio_value' not in result:
            continue
            
        data = result['portfolio_value']
        strategy_type, strategy_name = key.split('_', 1)
        
        # Determine line style based on strategy type
        linestyle = '--' if strategy_type == 'benchmark' else '-'
        linewidth = 2.2 if strategy_type == 'benchmark' else 2.0
        alpha = 0.9 if strategy_type == 'benchmark' else 1.0
        
        # Get color based on strategy name
        color = colors.get(strategy_name, '#7f8c8d')
        
        # Plot the series
        plt.plot(
            data.index, 
            data.values,
            label=f'{strategy_type.title()}: {strategy_name.replace("_", " ").title()}',
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha
        )
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics(data)
        metrics['Strategy'] = f'{strategy_type.title()}: {strategy_name.replace("_", " ").title()}'
        metrics['Type'] = strategy_type
        
        # Add information ratios if available
        if 'information_ratio' in result:
            metrics['information_ratio'] = result['information_ratio']
        if 'modified_information_ratio' in result:
            metrics['modified_information_ratio'] = result['modified_information_ratio']
            
        all_metrics.append(metrics)
        strategy_names.append(metrics['Strategy'])
    
    # Style the plot
    plt.title('Portfolio Value: Benchmarks vs Strategies', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Portfolio Value (Log Scale)', fontsize=12, labelpad=10)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, which='both')
    
    # Add watermark
    plt.figtext(0.5, 0.01, f'Generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
               ha='center', fontsize=8, color='gray', alpha=0.7)
    
    # Create a legend with two columns
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved strategy comparison plot to {save_path}")
    
    # Create and save metrics table
    if all_metrics:
        create_metrics_table(all_metrics, save_path)
    
    plt.show()

def create_metrics_table(metrics_list: list, save_path: str = None):
    """Create a formatted table of performance metrics."""
    # Convert to DataFrame
    df = pd.DataFrame(metrics_list).set_index('Strategy')
    
    # Define column order and formatting
    columns = [
        ('total_return', 'Total Return', '{:.2%}'),
        ('annualized_return', 'Ann. Return', '{:.2%}'),
        ('annualized_volatility', 'Ann. Vol', '{:.2%}'),
        ('sharpe_ratio', 'Sharpe', '{:.2f}'),
        ('max_drawdown', 'Max DD', '{:.2%}'),
        ('information_ratio', 'Info Ratio', '{:.2f}'),
        ('modified_information_ratio', 'Mod. IR', '{:.2f}')
    ]
    
    # Filter available columns
    available_columns = [col for col, _, _ in columns if col in df.columns]
    display_columns = [disp for col, disp, _ in columns if col in df.columns]
    
    # Create formatted DataFrame
    df_formatted = pd.DataFrame(index=df.index)
    for col, _, fmt in columns:
        if col in df.columns:
            df_formatted[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 4 + len(df) * 0.4))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df_formatted[available_columns].values,
        rowLabels=df_formatted.index,
        colLabels=display_columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight best values
    for i, (col, _, _) in enumerate(columns):
        if col not in df.columns:
            continue
            
        if col in ['sharpe_ratio', 'information_ratio']:
            best_idx = df[col].idxmax()
        else:
            best_idx = df[col].idxmin() if col == 'max_drawdown' else df[col].idxmax()
        
        if best_idx in df.index:
            row_idx = df.index.get_loc(best_idx)
            cell = table[(row_idx + 1, i)]
            cell.set_facecolor('#e6f7e6')
    
    # Add title and adjust layout
    plt.title('Performance Metrics Comparison', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save metrics table
    if save_path:
        metrics_path = save_path.replace('.png', '_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics table to {metrics_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print("-" * 105)
    print(f"{'Strategy':<40} | {'Ann. Return':>10} | {'Ann. Vol':>8} | {'Sharpe':>6} | {'Max DD':>6} | {'Info Ratio':>10} | {'Mod. IR':>8}")
    print("-" * 105)
    
    for idx, row in df.iterrows():
        print(f"{idx:<40} | "
              f"{row.get('annualized_return', 0)*100:>9.2f}% | "
              f"{row.get('annualized_volatility', 0)*100:>7.2f}% | "
              f"{row.get('sharpe_ratio', 0):>5.2f} | "
              f"{row.get('max_drawdown', 0)*100:>5.2f}% | "
              f"{row.get('information_ratio', 'N/A'):>10} | "
              f"{row.get('modified_information_ratio', 'N/A'):>7}")
    print("-" * 105)

def calculate_portfolio_metrics(portfolio_data, risk_free_rate: float = 0.05) -> dict:
    """Calculate performance metrics for a portfolio."""
    try:
        # Handle different input types
        if isinstance(portfolio_data, pd.DataFrame):
            if 'portfolio_value' in portfolio_data.columns:
                portfolio_values = portfolio_data['portfolio_value']
            else:
                portfolio_values = portfolio_data.iloc[:, 0]
        elif isinstance(portfolio_data, pd.Series):
            portfolio_values = portfolio_data
        else:
            portfolio_values = pd.Series(portfolio_data)
        
        # Ensure we have enough data
        if len(portfolio_values) < 2:
            return {}
            
        # Calculate daily returns
        daily_returns = portfolio_values.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return {}
        
        # Calculate metrics
        total_return = float(portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)
        annualized_return = float((1 + total_return) ** (252 / len(daily_returns)) - 1)
        annualized_vol = float(daily_returns.std() * np.sqrt(252))
        
        # Calculate max drawdown (as a fraction, will be formatted as percentage later)
        cum_returns = (1 + daily_returns).cumprod()
        peak = cum_returns.cummax()
        drawdowns = (cum_returns / peak) - 1
        max_drawdown = float(drawdowns.min())
        
        # Calculate Sharpe ratio
        if pd.notna(annualized_vol) and annualized_vol > 1e-10:
            sharpe_ratio = float((annualized_return - risk_free_rate) / annualized_vol)
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
    except Exception as e:
        print(f"Error calculating portfolio metrics: {str(e)}")
        return {}

def load_price_data(use_cached: bool = True) -> pd.DataFrame:
    """Load price data, ensuring we get the actual prices even if log returns are cached."""
    # First try to load prices directly
    if use_cached:
        try:
            prices = pd.read_csv('data/stock_data.csv', index_col=0, parse_dates=True)
            print("Loaded price data from cache")
            return prices
        except FileNotFoundError:
            pass
    
    # If not found in cache or use_cached is False, download fresh data
    print("Downloading price data...")
    from main import download_prices, save_data
    prices = download_prices()
    save_data(prices)
    return prices

def main():
    # Create necessary directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load price data
    prices = load_price_data(use_cached=True)
    
    if prices is None or prices.empty:
        print("Error: Could not load price data.")
        return
        
    # Subset to benchmark start date (2024-01-01)
    benchmark_start_date = '2024-01-01'
    prices = prices[prices.index >= benchmark_start_date]
    
    # Get rebalance dates (aligned with main strategy)
    rebalance_dates = prices.resample('M').last().index
    
    print(f"\n{'='*80}")
    print(f"PORTFOLIO BACKTESTING COMPARISON")
    print(f"{'='*80}")
    print(f"Date Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Number of Assets: {len(prices.columns)}")
    print(f"Rebalance Frequency: Monthly")
    print(f"Benchmark Start Date: {benchmark_start_date}")
    print(f"{'='*80}\n")
    
    print("Calculating benchmark strategies...")
    benchmark_results = calculate_benchmark_returns(
        prices,
        rebalance_freq='M',
        risk_free_rate=0.05,
        benchmark='equal_weight'
    )
    
    # Load benchmark returns for Information Ratio calculation
    benchmark_returns = benchmark_results.get('equal_weight', {}).get('daily_returns')
    
    print("\nLoading strategy results...")
    strategy_results = {}
    
    # Load ARIMA-GARCH strategies
    for strategy in ['arima_garch_gmir', 'arima_garch_gmv']:
        print(f"Loading {strategy}...")
        results = load_strategy_results(strategy, benchmark_returns=benchmark_returns)
        if results is not None:
            strategy_results[strategy] = results
            print(f"  ✓ Loaded {strategy} with {len(results.get('portfolio_value', []))} data points")
    
    # Load XGBoost strategies
    for strategy in ['xgboost_gmir', 'xgboost_gmv']:
        print(f"Loading {strategy}...")
        results = load_strategy_results(strategy, benchmark_returns=benchmark_returns)
        if results is not None:
            strategy_results[strategy] = results
            print(f"  ✓ Loaded {strategy} with {len(results.get('portfolio_value', []))} data points")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_all_strategies(
        benchmark_results,
        strategy_results,
        save_path='plots/strategy_comparison_equity.png'
    )
    
    # Generate benchmark comparison plot
    plot_benchmark_comparison(
        benchmark_results,
        save_path='plots/benchmark_comparison.png'
    )
    
    # Generate individual benchmark plots
    print("\nGenerating individual benchmark plots...")
    for strategy in ['buy_hold', 'equal_weight']:
        if strategy in benchmark_results:
            plot_equity_and_drawdown(
                benchmark_results[strategy],
                title=f'Benchmark: {strategy.replace("_", " ").title()}',
                safe_name=f'benchmark_{strategy}',
                first_rebalance_date=rebalance_dates[0] if len(rebalance_dates) > 0 else None
            )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to the 'plots' directory.")
    print(f"- Strategy Comparison: plots/strategy_comparison_equity.png")
    print(f"- Metrics Table: plots/strategy_comparison_equity_metrics.png")
    print(f"- Benchmark Comparison: plots/benchmark_comparison.png")

if __name__ == "__main__":
    main()
