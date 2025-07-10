"""
Script to run and compare benchmark strategies with existing portfolio strategies.
"""
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
                    
                    # Calculate Information Ratio
                    if tracking_error > 1e-10:  # Avoid division by zero
                        information_ratio = excess_return_annualized / tracking_error
                        result['information_ratio'] = information_ratio
                        
                        # Debug output
                        print(f"\nStrategy: {strategy_name}")
                        print(f"  Period: {len(aligned_data)} days")
                        print(f"  Strategy Annualized Return: {strategy_ret.mean() * 252:.2%}")
                        print(f"  Benchmark Annualized Return: {benchmark_ret.mean() * 252:.2%}")
                        print(f"  Excess Return (Annualized): {excess_return_annualized:.2%}")
                        print(f"  Tracking Error (Annualized): {tracking_error:.2%}")
                        print(f"  Information Ratio: {information_ratio:.2f}")
                    else:
                        print(f"Warning: Tracking error too small for {strategy_name}")
                else:
                    print(f"Warning: No overlapping dates found for {strategy_name}")
        
        return result
        
    except Exception as e:
        print(f"Error loading strategy {strategy_name}: {str(e)}")
        return None

def plot_all_strategies(benchmark_results: dict, strategy_results: dict, save_path: str = None):
    """Plot comparison of all strategies with performance table."""

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])

    all_metrics = []
    strategy_names = []

    # Color palette
    colors = {
        'buy_hold': '#1f77b4',
        'equal_weight': '#ff7f0e',
        'arima_garch_gmir': '#2ca02c',
        'arima_garch_gmv': '#d62728',
        'xgboost_gmir': '#9467bd',
        'xgboost_gmv': '#17becf'
    }

    # Plot benchmarks (dashed)
    for strategy in ['buy_hold', 'equal_weight']:
        if strategy in benchmark_results:
            data = benchmark_results[strategy]['portfolio_value']
            ax1.plot(
                data.index, data.values,
                label=f'Benchmark: {strategy.replace("_", " ").title()}',
                color=colors[strategy],
                linestyle='--',
                linewidth=2.2
            )
            metrics = calculate_portfolio_metrics(data)
            metrics['Strategy'] = f'Benchmark: {strategy.replace("_", " ").title()}'
            # Add information ratio if available
            if 'information_ratio' in benchmark_results[strategy]:
                metrics['information_ratio'] = benchmark_results[strategy]['information_ratio']
            all_metrics.append(metrics)
            strategy_names.append(metrics['Strategy'])

    # Plot strategy results (solid)
    for name, result in strategy_results.items():
        if result:
            data = result['portfolio_value']
            ax1.plot(
                data.index, data.values,
                label=f'Strategy: {name.upper()}',
                color=colors.get(name, '#333333'),
                linewidth=2.5
            )
            metrics = calculate_portfolio_metrics(data)
            metrics['Strategy'] = f'Strategy: {name.upper()}'
            # Add information ratio if available (for strategies loaded from files, we might not have it)
            if 'information_ratio' in result:
                metrics['information_ratio'] = result['information_ratio']
            all_metrics.append(metrics)
            strategy_names.append(metrics['Strategy'])

    # Title & labels
    ax1.set_title('Strategy Performance Comparison (2024-01-01 to Present)', fontsize=16, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value (Initial = 1.0)', fontsize=12)
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.text(0.99, 0.01, 'Generated ' + datetime.now().strftime('%Y-%m-%d'),
             transform=ax1.transAxes, ha='right', fontsize=8, color='gray')

    # Metrics table below plot
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df = df.set_index('Strategy')
        
        # Include information ratio in display if available
        columns = ['total_return', 'annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']
        if 'information_ratio' in df.columns:
            columns.append('information_ratio')
        
        display_df = df[columns]

        # Format display
        display_df = display_df.copy()
        for col in ['total_return', 'annualized_return', 'annualized_volatility', 'max_drawdown']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.2%}" if pd.notna(x) else "N/A")
        
        if 'sharpe_ratio' in display_df.columns:
            display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "N/A")
            
        if 'information_ratio' in display_df.columns:
            display_df['information_ratio'] = display_df['information_ratio'].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "N/A")

        table = ax2.table(
            cellText=display_df.values,
            rowLabels=display_df.index,
            colLabels=['Total Return', 'Ann. Return', 'Ann. Vol', 'Sharpe', 'Max DD'] + (['Info Ratio'] if 'information_ratio' in df.columns else []),
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved strategy comparison plot with table to {save_path}")

    plt.show()

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
    
    print(f"Loaded price data for {len(prices.columns)} stocks from {prices.index[0].date()} to {prices.index[-1].date()}")
    
    print("Calculating benchmark strategies...")
    benchmark_results = calculate_benchmark_returns(
        prices,
        rebalance_freq='M',  # Monthly rebalancing for equal weight
        risk_free_rate=0.05,  # 5% risk-free rate for Sharpe ratio
        benchmark='equal_weight'  # Use equal weight as the benchmark for Information Ratio
    )
    
    # Load benchmark returns for Information Ratio calculation
    benchmark_returns = benchmark_results.get('equal_weight', {}).get('daily_returns')
    
    # Load existing strategy results
    strategy_results = {}
    # ARIMA-GARCH strategies
    for strategy in ['arima_garch_gmir', 'arima_garch_gmv']:
        results = load_strategy_results(strategy, benchmark_returns=benchmark_returns)
        if results is not None:
            strategy_results[f"arima_garch_{strategy.split('_')[-1]}"] = results
    
    # XGBoost strategies
    for strategy in ['xgboost_gmir', 'xgboost_gmv']:
        results = load_strategy_results(strategy, benchmark_returns=benchmark_returns)
        if results is not None:
            strategy_results[strategy] = results
    
    # Plot comparison
    plot_all_strategies(
        benchmark_results,
        strategy_results,
        save_path='plots/strategy_comparison_equity.png'
    )
    
    # Plot benchmark comparison
    plot_benchmark_comparison(
        benchmark_results,
        save_path='plots/benchmark_comparison.png'
    )
    
    # Save individual benchmark plots
    for strategy in ['buy_hold', 'equal_weight']:
        if strategy in benchmark_results:
            plot_equity_and_drawdown(
                benchmark_results[strategy],
                title=f'Benchmark: {strategy.replace("_", " ").title()}',
                safe_name=f'benchmark_{strategy}',
                first_rebalance_date=rebalance_dates[0] if len(rebalance_dates) > 0 else None
            )

if __name__ == "__main__":
    main()
