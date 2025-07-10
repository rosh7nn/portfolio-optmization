"""
Script to run and compare benchmark strategies with existing portfolio strategies.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from benchmarks import calculate_benchmark_returns, plot_benchmark_comparison
from main import prepare_data, plot_equity_and_drawdown

def load_strategy_results(strategy_name: str) -> dict:
    """Load results from a saved strategy."""
    results_path = Path('data') / f'portfolio_{strategy_name}.csv'
    if not results_path.exists():
        return None
    
    portfolio_values = pd.read_csv(results_path, index_col=0, parse_dates=True).squeeze()
    return {
        'portfolio_value': portfolio_values,
        'daily_returns': portfolio_values.pct_change().fillna(0)
    }

def plot_all_strategies(benchmark_results: dict, strategy_results: dict, save_path: str = None):
    """Plot comparison of all strategies."""
    sns.set_style('whitegrid')
    plt.figure(figsize=(14, 7))
    
    # Plot benchmark strategies
    colors = ['#1f77b4', '#ff7f0e']  # Different colors for benchmarks
    for i, strategy in enumerate(['buy_hold', 'equal_weight']):
        if strategy in benchmark_results:
            plt.plot(
                benchmark_results[strategy]['portfolio_value'], 
                label=f'Benchmark: {str.replace(strategy, "_", " ").title()}',
                color=colors[i],
                linestyle='--',
                linewidth=2
            )
    
    # Plot other strategies
    colors = ['#2ca02c', '#d62728', '#9467bd']  # Different colors for other strategies
    for i, (name, result) in enumerate(strategy_results.items()):
        if result is not None and 'portfolio_value' in result:
            plt.plot(
                result['portfolio_value'],
                label=f'Strategy: {name.upper()}',
                color=colors[i % len(colors)],
                linewidth=2
            )
    
    plt.title('Strategy Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (Initial = 1.0)', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved strategy comparison plot to {save_path}")
    
    plt.show()

def calculate_portfolio_metrics(portfolio_values: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Calculate performance metrics for a portfolio."""
    daily_returns = portfolio_values.pct_change().dropna()
    
    if len(daily_returns) == 0:
        return {}
    
    # Calculate metrics
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    annualized_vol = daily_returns.std() * np.sqrt(252)
    
    # Calculate max drawdown
    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.cummax()
    drawdowns = (cum_returns / peak - 1) * 100
    max_drawdown = drawdowns.min()
    
    # Calculate Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
    
    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_vol),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown)
    }

def print_metrics_table(benchmark_results: dict, strategy_results: dict):
    """Print a table comparing performance metrics."""
    # Prepare data
    metrics_data = []
    
    # Add benchmark metrics
    for strategy in ['buy_hold', 'equal_weight']:
        if strategy in benchmark_results and 'portfolio_value' in benchmark_results[strategy]:
            portfolio_values = benchmark_results[strategy]['portfolio_value']
            metrics = calculate_portfolio_metrics(portfolio_values)
            if metrics:
                metrics['Strategy'] = f'Benchmark: {str.replace(strategy, "_", " ").title()}'
                metrics_data.append(metrics)
    
    # Add strategy metrics
    for name, result in strategy_results.items():
        if result is not None and 'portfolio_value' in result:
            metrics = calculate_portfolio_metrics(result['portfolio_value'])
            if metrics:
                metrics['Strategy'] = f'Strategy: {name.upper()}'
                metrics_data.append(metrics)
    
    # Create and display DataFrame
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        df = df.set_index('Strategy')
        
        # Format for display
        formatted_df = df.copy()
        for col in ['total_return', 'annualized_return', 'annualized_volatility', 'sharpe_ratio']:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
        
        if 'max_drawdown' in formatted_df.columns:
            formatted_df['max_drawdown'] = formatted_df['max_drawdown'].apply(
                lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
            )
        
        print("\nPerformance Metrics Comparison:")
        print("-" * 80)
        print(formatted_df.to_string())
        print("-" * 80)
        
        # Save to CSV
        metrics_path = 'data/strategy_metrics_comparison.csv'
        df.to_csv(metrics_path)
        print(f"\nSaved detailed metrics to {metrics_path}")

def load_price_data(use_cached: bool = True) -> pd.DataFrame:
    """Load price data, ensuring we get the actual prices even if log returns are cached."""
    # First try to load prices directly
    prices = None
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
    
    # Get rebalance dates (aligned with main strategy)
    rebalance_dates = prices.resample('M').last().index
    
    print(f"Loaded price data for {len(prices.columns)} stocks from {prices.index[0].date()} to {prices.index[-1].date()}")
    
    print("Calculating benchmark strategies...")
    benchmark_results = calculate_benchmark_returns(
        prices,
        rebalance_freq='M',  # Monthly rebalancing for equal weight
        risk_free_rate=0.05  # 5% risk-free rate for Sharpe ratio
    )
    
    # Save benchmark results
    for strategy in ['buy_hold', 'equal_weight']:
        if strategy in benchmark_results:
            df = pd.DataFrame({
                'portfolio_value': benchmark_results[strategy]['portfolio_value'],
                'daily_returns': benchmark_results[strategy]['daily_returns']
            })
            df.to_csv(f'data/benchmark_{strategy}.csv')
    
    # Load existing strategy results
    strategy_results = {}
    for strategy in ['arima_garch_gmir', 'arima_garch_gmv']:  # Add your strategy names here
        results = load_strategy_results(strategy)
        if results is not None:
            strategy_results[strategy] = results
    
    # Plot comparison
    plot_all_strategies(
        benchmark_results,
        strategy_results,
        save_path='plots/strategy_comparison_equity.png'
    )
    
    # Print metrics table
    print_metrics_table(benchmark_results, strategy_results)
    
    # Save individual benchmark plots
    for strategy in ['buy_hold', 'equal_weight']:
        if strategy in benchmark_results:
            plot_equity_and_drawdown(
                benchmark_results[strategy],
                title=f'Benchmark: {str.replace(strategy, "_", " ").title()}',
                safe_name=f'benchmark_{strategy}',
                first_rebalance_date=rebalance_dates[0] if len(rebalance_dates) > 0 else None
            )

if __name__ == "__main__":
    import numpy as np  # Import here to avoid circular imports
    main()
