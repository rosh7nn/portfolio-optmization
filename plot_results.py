import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.dates as mdates

# Set the style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def load_metrics() -> Dict[str, Dict[str, float]]:
    """Load all metrics from CSV files
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping strategy names to their metrics
    """
    metrics_dir = Path('data')
    metrics = {}
    
    for file in metrics_dir.glob('metrics_*.csv'):
        strategy = file.stem.replace('metrics_', '')
        try:
            df = pd.read_csv(file)
            if not df.empty:
                metrics[strategy] = df.iloc[0].to_dict()  # Get metrics as dict
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return metrics

def ensure_plots_dir():
    """Ensure the plots directory exists"""
    os.makedirs('plots', exist_ok=True)

def plot_equity_curves():
    """Plot equity curves for all strategies with transaction cost variants"""
    plt.figure(figsize=(14, 7))
    
    # Get all portfolio data files
    data_dir = Path('data')
    if not data_dir.exists():
        print("Data directory not found. Please run the backtest first.")
        return
        
    portfolio_files = list(data_dir.glob('portfolio_*.csv'))
    
    if not portfolio_files:
        print("No portfolio data files found. Please run the backtest first.")
        return
    
    for file in portfolio_files:
        strategy_name = file.stem.replace('portfolio_', '')
        try:
            data = pd.read_csv(file, parse_dates=['date'], index_col='date')
            display_name = ' '.join(word.capitalize() for word in strategy_name.split('_'))
            
            # Plot base portfolio value (0 bps)
            plt.plot(data.index, data['portfolio_value'], 
                    label=display_name, linewidth=2)
            
            # Plot transaction cost variants if they exist
            for tc in [10, 20]:
                tc_col = f'portfolio_value_after_tc_{tc}bps'
                if tc_col in data.columns:
                    plt.plot(data.index, data[tc_col], '--', 
                            label=f'{display_name} ({tc}bps)', alpha=0.7, linewidth=1.5)
            
        except Exception as e:
            print(f"Error plotting {file}: {e}")
    
    plt.title('Portfolio Value Over Time', fontsize=14, pad=15)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Portfolio Value ($)', fontsize=12, labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    # Save high-quality figure
    ensure_plots_dir()
    plt.savefig('plots/equity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved equity curves to plots/equity_curves.png")

def plot_drawdowns():
    """Plot drawdown curves for all strategies from saved data"""
    # Create figure with a larger height to accommodate the legend
    plt.figure(figsize=(14, 6))
    
    # Check if data directory exists
    data_dir = Path('data')
    if not data_dir.exists():
        print("Data directory not found. Please run the backtest first.")
        return
    
    # Find all portfolio files
    portfolio_files = list(data_dir.glob('portfolio_*.csv'))
    
    if not portfolio_files:
        print("No portfolio data files found. Please run the backtest first.")
        return
    
    print(f"Found portfolio data for {len(portfolio_files)} strategies")
    
    # Use a consistent color palette
    colors = plt.cm.tab10.colors
    
    for i, file in enumerate(portfolio_files):
        strategy_name = file.stem.replace('portfolio_', '')
        try:
            # Load data
            data = pd.read_csv(file, parse_dates=['date'], index_col='date')
            
            # Check if drawdown data exists
            if 'drawdown' in data.columns:
                display_name = ' '.join(word.capitalize() for word in strategy_name.split('_'))
                
                # Convert to percentage and plot
                drawdowns = data['drawdown'] * 100
                
                # Get a consistent color for this strategy
                color = colors[i % len(colors)]
                
                # Plot the drawdown curve
                plt.plot(data.index, drawdowns, 
                         label=display_name, 
                         color=color,
                         linewidth=1.5)
                
                # Fill under the drawdown curve
                plt.fill_between(data.index, drawdowns, 0, 
                                color=color, alpha=0.1)
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Add a zero line for reference
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
    
    # Format the plot
    plt.title('Portfolio Drawdown Over Time', fontsize=14, pad=15)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Drawdown (%)', fontsize=12, labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.2)
    
    # Ensure y-axis shows negative values properly
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=min(ymin, -5), ymax=5)  # Ensure we see at least -5% to 5%
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure plots directory exists and save the figure
    ensure_plots_dir()
    save_path = 'plots/drawdowns.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved drawdown plot to {save_path}")

def plot_concentration_metrics():
    """Plot concentration metrics (MN75% and MN90%) for all strategies from saved data"""
    data_dir = Path('data')
    
    # Check if data directory exists
    if not data_dir.exists():
        print("Data directory not found. Please run the backtest first.")
        return
    
    # Find all concentration metrics files
    conc_files = list(data_dir.glob('concentration_*.csv'))
    
    if not conc_files:
        print("No concentration metrics files found. Please run the backtest first.")
        return
    
    print(f"Found concentration metrics for {len(conc_files)} strategies")
    
    # Create subplots for MN75% and MN90%
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Use a consistent color palette
    colors = plt.cm.tab10.colors
    
    for i, file in enumerate(conc_files):
        strategy_name = file.stem.replace('concentration_', '')
        try:
            # Load data
            data = pd.read_csv(file, parse_dates=['date'], index_col='date')
            display_name = ' '.join(word.capitalize() for word in strategy_name.split('_'))
            
            # Get a consistent color for this strategy
            color = colors[i % len(colors)]
            
            # Plot MN75% if available
            if 'MN75%' in data.columns:
                ax1.plot(data.index, data['MN75%'], 
                        label=display_name, 
                        color=color,
                        linewidth=1.5)
            
            # Plot MN90% if available
            if 'MN90%' in data.columns:
                ax2.plot(data.index, data['MN90%'], 
                        label=display_name, 
                        color=color,
                        linewidth=1.5)
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Configure MN75% subplot
    ax1.set_title('MN75% - Number of Assets Contributing to 75% of Portfolio', 
                 fontsize=14, pad=12)
    ax1.set_ylabel('Number of Assets', fontsize=12, labelpad=8)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(bottom=0)  # Start y-axis at 0
    
    # Configure MN90% subplot
    ax2.set_title('MN90% - Number of Assets Contributing to 90% of Portfolio', 
                 fontsize=14, pad=12)
    ax2.set_xlabel('Date', fontsize=12, labelpad=10)
    ax2.set_ylabel('Number of Assets', fontsize=12, labelpad=8)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(bottom=0)  # Start y-axis at 0
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure plots directory exists and save the figure
    ensure_plots_dir()
    save_path = 'plots/concentration_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved concentration metrics plot to {save_path}")

def plot_metrics_comparison():
    """
    Plot comparison of key metrics across all strategies.
    
    This function loads metrics from saved CSV files and creates a comprehensive
    visualization comparing different performance metrics across strategies.
    """
    print("Loading metrics for comparison...")
    
    # Load all metrics
    metrics = load_metrics()
    if not metrics:
        print("❌ No metrics data found. Please run the backtest first.")
        return
        
    print(f"Found metrics for {len(metrics)} strategies")
    
    # Convert to DataFrame for easier manipulation
    metrics_df = pd.DataFrame(metrics).T
    
    # Define the metrics we want to display and their order
    selected_metrics = [
        'annualized_return',
        'annualized_volatility',
        'sharpe_ratio',
        'max_drawdown',
        'calmar_ratio',
        'sortino_ratio',
        'information_ratio',
        'annualized_turnover'
    ]
    
    # Filter to only include metrics that exist in our data
    display_metrics = [m for m in selected_metrics if m in metrics_df.columns]
    
    if not display_metrics:
        print("❌ No valid metrics found for comparison")
        return
        
    print(f"Generating comparison for {len(display_metrics)} metrics...")
    
    # Create pretty display names
    pretty_names = {
        'annualized_return': 'Annual Return',
        'annualized_volatility': 'Volatility',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'calmar_ratio': 'Calmar Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'information_ratio': 'Information Ratio',
        'annualized_turnover': 'Turnover'
    }
    
    # Set up the figure with appropriate size
    n_metrics = len(display_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Adjust figure size based on number of metrics
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, fig_height))
    
    # Flatten axes if needed
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Define colors
    colors = plt.cm.tab10.colors
    
    for i, metric in enumerate(display_metrics):
        ax = axes[i]
        metric_data = metrics_df[metric].dropna()
        
        if metric_data.empty:
            print(f"  - Warning: No data for metric '{metric}'") 
            continue
            
        display_name = pretty_names.get(metric, metric.replace('_', ' ').title())
        
        # Determine if higher is better for this metric
        higher_is_better = 'return' in metric or 'ratio' in metric
        
        # Sort values appropriately
        if higher_is_better:
            metric_data = metric_data.sort_values(ascending=False)
            color = 'skyblue'
        else:
            if metric != 'turnover':
                metric_data = metric_data.sort_values(ascending=True)
            color = 'lightcoral'
        
        # Create bar plot
        bars = ax.bar(metric_data.index, metric_data, color=color, alpha=0.8)
        
        # Format y-axis label
        ylabel = display_name
        if metric in ['annualized_return', 'annualized_volatility']:
            ylabel += ' (%)'
        elif metric == 'max_drawdown':
            ylabel += ' (%)'
        ax.set_ylabel(ylabel, fontsize=11)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if metric in ['annualized_return', 'annualized_volatility', 'max_drawdown']:
                value_str = f'{-height:.1f}%' if metric == 'max_drawdown' else f'{height:.1f}%'
            else:
                value_str = f'{height:.2f}'
                
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   value_str,
                   ha='center', va='bottom',
                   fontsize=9, rotation=45)
        
        # Highlight best performer
        if len(metric_data) > 0:
            best_idx = metric_data.idxmax() if higher_is_better else metric_data.idxmin()
            for j, (idx, val) in enumerate(zip(metric_data.index, metric_data)):
                if idx == best_idx:
                    bars[j].set_edgecolor('green')
                    bars[j].set_linewidth(2)
        
        # Style the plot
        ax.set_title(display_name, fontsize=12, pad=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Adjust y-limits to make room for labels
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)  # Add 15% more space at the top
    
    # Remove any empty subplots
    for j in range(len(display_metrics), len(axes)):
        fig.delaxes(axes[j])
    
    # Add a main title
    plt.suptitle('Performance Metrics Comparison', fontsize=16, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Ensure plots directory exists and save the figure
    ensure_plots_dir()
    save_path = 'plots/metrics_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved metrics comparison to {save_path}")

def main():
    """
    Main function to run all plotting functions.
    
    This function coordinates the generation of all visualizations by calling
    individual plotting functions in the appropriate order.
    """
    print("=" * 80)
    print("PORTFOLIO BACKTEST VISUALIZATION TOOL")
    print("=" * 80)
    print("This tool generates visualizations from saved backtest results.\n")
    
    # Create plots directory if it doesn't exist
    ensure_plots_dir()
    
    # Check if data directory exists
    data_dir = Path('data')
    if not data_dir.exists():
        print("❌ Error: 'data' directory not found.")
        print("Please run the backtest first to generate the required data files.")
        return
    
    # Check if there are any portfolio files
    portfolio_files = list(data_dir.glob('portfolio_*.csv'))
    if not portfolio_files:
        print("❌ Error: No portfolio data files found in the 'data' directory.")
        print("Please run the backtest first to generate the required data files.")
        return
    
    print(f"Found data for {len(portfolio_files)} strategy(ies)")
    print("-" * 80)
    
    try:
        # 1. Generate equity curves
        print("\n1. Generating equity curves...")
        plot_equity_curves()
        
        # 2. Generate drawdown plots
        print("\n2. Generating drawdown plots...")
        plot_drawdowns()
        
        # 3. Generate concentration metrics plots
        print("\n3. Generating concentration metrics plots...")
        plot_concentration_metrics()
        
        # 4. Generate metrics comparison
        print("\n4. Generating performance metrics comparison...")
        plot_metrics_comparison()
        
        # Print completion message
        print("\n" + "=" * 80)
        print("✅ VISUALIZATION COMPLETE")
        print("=" * 80)
        print("\nGenerated the following visualizations in the 'plots' directory:")
        print("  • Equity curves with transaction cost variants (equity_curves.png)")
        print("  • Drawdown analysis (drawdowns.png)")
        print("  • Portfolio concentration metrics (concentration_metrics.png)")
        print("  • Performance metrics comparison (metrics_comparison.png)")
        print("\nYou can find all plots in the 'plots' directory.")
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("❌ ERROR: An unexpected error occurred during visualization")
        print("!" * 80)
        print(f"\nError details: {str(e)}")
        print("\nPlease check that all required data files are present in the 'data' directory")
        print("and that they are properly formatted.")
        
        # Print traceback for debugging
        import traceback
        print("\nTechnical details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
