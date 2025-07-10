"""
Benchmark strategies for portfolio comparison.
"""
import pandas as pd
import numpy as np

def calculate_benchmark_returns(prices: pd.DataFrame, rebalance_freq: str = 'M', 
                              risk_free_rate: float = 0.05, 
                              benchmark: str = 'equal_weight') -> dict:
    """
    Calculate benchmark strategy returns.
    
    Args:
        prices: DataFrame with dates as index and assets as columns
        rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q', 'A')
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        
    Returns:
        Dictionary with benchmark strategy results
    """
    results = {}
    
    # 1. Buy and Hold (Equal Weight at Start)
    if not prices.empty:
        # Equal weight at start
        initial_weights = pd.Series(1/len(prices.columns), index=prices.columns)
        
        # Calculate portfolio value over time
        normalized_prices = prices.div(prices.iloc[0])
        portfolio_value = (normalized_prices * initial_weights).sum(axis=1)
        
        # Calculate daily returns
        daily_returns = portfolio_value.pct_change().fillna(0)
        
        results['buy_hold'] = {
            'portfolio_value': portfolio_value,
            'daily_returns': daily_returns,
            'weights': initial_weights
        }
    
    # 2. Equal Weight with periodic rebalancing
    rebalance_dates = prices.resample(rebalance_freq).last().index
    portfolio_value = pd.Series(1.0, index=prices.index)
    current_weights = pd.Series(1/len(prices.columns), index=prices.columns)

    for i in range(1, len(prices)):
        daily_ret = prices.iloc[i] / prices.iloc[i-1] - 1
        portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + (current_weights * daily_ret).sum())
        if prices.index[i] in rebalance_dates or i == 1:
            current_weights = pd.Series(1/len(prices.columns), index=prices.columns)

    daily_returns = portfolio_value.pct_change().fillna(0)
    results['equal_weight'] = {
        'portfolio_value': portfolio_value,
        'daily_returns': daily_returns,
        'weights': current_weights
    }
    
    # Calculate information ratio for each strategy relative to benchmark
    if benchmark in results:
        benchmark_returns = results[benchmark]['daily_returns']
        for strategy in results:
            if strategy != benchmark:
                strategy_returns = results[strategy]['daily_returns']
                active_returns = strategy_returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)  # Annualized tracking error
                if tracking_error > 1e-10:  # Avoid division by zero
                    information_ratio = (strategy_returns.mean() - benchmark_returns.mean()) * 252 / tracking_error
                else:
                    information_ratio = np.nan
                results[strategy]['information_ratio'] = information_ratio
        results[benchmark]['information_ratio'] = 0.0  # Benchmark's IR is 0 by definition
    
    # 2. Equal Weight (Rebalanced periodically)
    if not prices.empty and rebalance_freq:
        # Resample to get rebalance dates
        rebalance_dates = prices.resample(rebalance_freq).last().index
        
        # Initialize portfolio value and weights
        portfolio_value = pd.Series(1.0, index=prices.index)
        current_weights = pd.Series(1/len(prices.columns), index=prices.columns)
        
        # Track weights history
        weights_history = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        for i in range(1, len(prices)):
            current_date = prices.index[i]
            prev_date = prices.index[i-1]
            
            # Update portfolio value based on previous day's weights and returns
            daily_ret = prices.iloc[i] / prices.iloc[i-1] - 1
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + (current_weights * daily_ret).sum())
            
            # Rebalance at specified frequency
            if current_date in rebalance_dates or i == 1:
                current_weights = pd.Series(1/len(prices.columns), index=prices.columns)
            
            weights_history.loc[current_date] = current_weights
        
        # Calculate daily returns
        daily_returns = portfolio_value.pct_change().fillna(0)
        
        results['equal_weight_rebalanced'] = {
            'portfolio_value': portfolio_value,
            'daily_returns': daily_returns,
            'weights': weights_history
        }
    
    return results

def plot_benchmark_comparison(benchmark_results: dict, save_path: str = None):
    """
    Plot comparison of benchmark strategies with performance metrics.
    
    Args:
        benchmark_results: Dictionary with benchmark results from calculate_benchmark_returns()
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    # Create figure with two subplots: one for the plot, one for the table
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    
    # Calculate metrics for each strategy
    metrics = []
    for strategy, result in benchmark_results.items():
        if 'daily_returns' in result and 'portfolio_value' in result:
            daily_returns = result['daily_returns']
            portfolio_value = result['portfolio_value']
            
            # Calculate metrics
            total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
            annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
            annualized_vol = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.05) / annualized_vol if annualized_vol > 0 else 0
            
            # Calculate max drawdown
            cum_returns = (1 + daily_returns).cumprod()
            peak = cum_returns.cummax()
            drawdowns = (cum_returns / peak) - 1
            max_drawdown = drawdowns.min()
            
            # Get information ratio (calculated in calculate_benchmark_returns)
            info_ratio = result.get('information_ratio', 0.0)
            
            metrics.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'Total Return': total_return,
                'Ann. Return': annualized_return,
                'Ann. Vol': annualized_vol,
                'Sharpe': sharpe_ratio,
                'Max DD': max_drawdown,
                'Info Ratio': info_ratio
            })
            
            # Plot the strategy
            ax1.plot(
                portfolio_value,
                label=f'Benchmark: {str.replace(strategy, "_", " ").title()}',
                linewidth=2
            )
    
    # Format the main plot
    ax1.set_title('Benchmark Strategy Comparison', fontsize=16, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value (Initial = 1.0)', fontsize=12)
    ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.2)
    
    # Add metrics table
    if metrics:
        ax2 = plt.subplot(gs[1])
        ax2.axis('off')
        
        # Convert metrics to DataFrame and format for display
        metrics_df = pd.DataFrame(metrics)
        display_df = metrics_df.set_index('Strategy')
        
        # Format numbers for display
        for col in ['Total Return', 'Ann. Return', 'Ann. Vol', 'Max DD']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
        display_df['Info Ratio'] = display_df['Info Ratio'].apply(lambda x: f"{x:.2f}")
        
        # Create table
        table = ax2.table(
            cellText=display_df.values,
            rowLabels=display_df.index,
            colLabels=display_df.columns,
            cellLoc='center',
            loc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header row
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header row
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4B4B4B')
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
