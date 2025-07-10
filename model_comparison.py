"""
Comprehensive comparison of all forecasting models against benchmarks for NIFTY 50 stocks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import os

def setup_plot_style():
    """Set up consistent plot styling with professional formatting."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Custom color palette
    colors = {
        'Buy & Hold': '#2E86C1',
        'Equal Weight': '#E67E22',
        'ARIMA-GARCH': '#27AE60',
        'XGBoost': '#C0392B',
        'text': '#2C3E50',
        'grid': '#BDC3C7',
        'background': '#FFFFFF'
    }
    
    # Update rcParams with custom styling
    plt.rcParams.update({
        'figure.figsize': (18, 10),
        'axes.facecolor': colors['background'],
        'figure.facecolor': colors['background'],
        'grid.color': colors['grid'],
        'grid.alpha': 0.3,
        'axes.edgecolor': colors['grid'],
        'axes.labelcolor': colors['text'],
        'text.color': colors['text'],
        'xtick.color': colors['text'],
        'ytick.color': colors['text'],
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 
                           'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': colors['background'],
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })

def load_strategy_data() -> Dict[str, pd.Series]:
    """Load all strategy results from saved files."""
    # Define all strategy files to load
    strategy_files = {
        'Buy & Hold': 'data/benchmark_buy_hold.csv',
        'Equal Weight': 'data/benchmark_equal_weight.csv',
        'ARIMA-GARCH GMIR': 'data/portfolio_arima_garch_gmir.csv',
        'ARIMA-GARCH GMV': 'data/portfolio_arima_garch_gmv.csv',
        'XGBoost GMIR': 'data/portfolio_xgboost_gmir.csv',
        'XGBoost GMV': 'data/portfolio_xgboost_gmv.csv'
    }
    
    results = {}
    for name, filepath in strategy_files.items():
        path = Path(filepath)
        if path.exists():
            try:
                data = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]  # Take first column if multiple
                results[name] = data
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Error loading {name}: {str(e)}")
        else:
            print(f"✗ File not found: {filepath}")
    
    return results

def calculate_metrics(values: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Calculate performance metrics for a value series."""
    if len(values) < 2:
        return {}
    
    # Ensure we're working with numeric values
    values = pd.to_numeric(values, errors='coerce').dropna()
    if len(values) < 2:
        return {}
    
    # Calculate daily returns from portfolio values
    daily_returns = values.pct_change().dropna()
    
    # Calculate cumulative returns
    cum_returns = (1 + daily_returns).cumprod()
    
    # Basic metrics
    total_return = values.iloc[-1] / values.iloc[0] - 1
    
    # Calculate annualized return
    years = len(daily_returns) / 252  # 252 trading days in a year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Calculate volatility (annualized)
    annualized_vol = daily_returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
    
    # Drawdown metrics
    peak = cum_returns.cummax()
    drawdowns = (cum_returns / peak - 1)
    max_drawdown_pct = drawdowns.min() * 100  # Convert to percentage
    
    # Sortino ratio (downside risk only)
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    # Calmar ratio (return vs max drawdown)
    calmar_ratio = annualized_return / abs(max_drawdown_pct/100) if max_drawdown_pct < 0 else 0
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Max Drawdown (%)': max_drawdown_pct,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Final Value': values.iloc[-1],
        'Initial Value': values.iloc[0],
        'Trading Days': len(daily_returns)
    }

def add_performance_insights(ax, strategy_data, metrics_df):
    """Add performance insights and explanations to the plot."""
    insights = [
        "Performance Insights:\n" + "-"*40,
        "1. Buy & Hold benefits from perfect hindsight and no turnover",
        "2. Forecasting models include transaction costs and prediction uncertainty",
        "3. Models may underperform in strong bull markets",
        "4. Risk-adjusted returns (Sharpe Ratio) often favor models"
    ]
    
    # Add strategy-specific notes
    if 'Buy & Hold' in metrics_df.index:
        bh_ret = metrics_df.loc['Buy & Hold', 'Total Return']
        insights.append(f"\nBuy & Hold returned {float(bh_ret.strip('%'))/100:.1%} (annualized)")
    
    if 'XGBoost GMV' in metrics_df.index:
        xg_ret = metrics_df.loc['XGBoost GMV', 'Total Return']
        insights.append(f"Best model (XGBoost GMV) returned {float(xg_ret.strip('%'))/100:.1%}")
    
    # Add the text box
    ax.text(
        1.02, 0.5,
        "\n".join(insights),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

def plot_strategy_comparison(strategy_data: Dict[str, pd.Series], metrics_df: pd.DataFrame = None, save_path: str = None):
    """Plot comparison of all strategies."""
    setup_plot_style()
    
    # Define colors and styles for different strategy types
    style_map = {
        'Buy & Hold': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.9},
        'Equal Weight': {'color': '#ff7f0e', 'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9},
        'ARIMA-GARCH GMIR': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        'ARIMA-GARCH GMV': {'color': '#27ae60', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8},
        'XGBoost GMIR': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        'XGBoost GMV': {'color': '#e74c3c', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8}
    }
    
    # Normalize all series to start at 1.0 for fair comparison
    normalized_data = {}
    for name, values in strategy_data.items():
        if len(values) > 0:
            normalized_data[name] = values / values.iloc[0]
        else:
            normalized_data[name] = values
    
    # Create figure with single subplot for equity curve
    fig, ax1 = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor(plt.rcParams['figure.facecolor'])
    
    # Plot equity curves
    for name, values in strategy_data.items():
        # Determine style based on strategy type
        style = None
        for key in style_map:
            if key in name:
                style = style_map[key].copy()
                break
        
        if style is None:
            style = {'color': '#7f7f7f', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7}
        
        # Plot equity curve
        if len(normalized_data[name]) > 0:
            ax1.plot(
                normalized_data[name].index,
                normalized_data[name],
                label=name,
                **style
            )
            
    # Format the plot with enhanced styling
    ax1.set_title('NIFTY 50 Strategy Performance Comparison', 
                 fontsize=20, pad=20, fontweight='bold', color=plt.rcParams['text.color'])
    
    # Add subtitle with date range
    if len(strategy_data) > 0:
        dates = next(iter(strategy_data.values())).index
        date_range = f"{dates[0].strftime('%b %Y')} - {dates[-1].strftime('%b %Y')}"
        ax1.text(0.5, 1.02, date_range, 
                transform=ax1.transAxes, ha='center', 
                fontsize=13, color=plt.rcParams['text.color'], alpha=0.8)
    
    # Format axes
    ax1.set_xlabel('Date', fontsize=14, labelpad=10, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=14, labelpad=10, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}×"))
    
    # Customize grid and spines
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add reference line
    ax1.axhline(1.0, color=plt.rcParams['grid.color'], linestyle='--', alpha=0.7)
    
    # Add legend with better formatting
    legend = ax1.legend(
        loc='upper left', 
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        framealpha=0.9,
        edgecolor=plt.rcParams['grid.color'],
        facecolor=plt.rcParams['axes.facecolor'],
        fontsize=12,
        labelspacing=1
    )
    
    # Add watermark
    fig.text(0.99, 0.01, 'Generated by FCK Analysis', 
             fontsize=10, color=plt.rcParams['grid.color'],
             ha='right', va='bottom', alpha=0.7)
    
    # Add performance metrics in a table format
    metrics_data = []
    for name, values in strategy_data.items():
        metrics = calculate_metrics(values)
        if metrics:
            metrics_data.append({
                'Strategy': name,
                'Return': metrics['Total Return'],
                'Volatility': metrics['Annualized Volatility'],
                'Sharpe': metrics['Sharpe Ratio']
            })
    
    if metrics_data:
        # Create a table
        cell_text = []
        for row in metrics_data:
            cell_text.append([
                row['Strategy'],
                f"{row['Return']:+.1%}",
                f"{row['Volatility']:.1%}",
                f"{row['Sharpe']:.2f}"
            ])
        
        # Add table at the bottom
        table = ax1.table(
            cellText=cell_text,
            colLabels=['Strategy', 'Total Return', 'Volatility', 'Sharpe'],
            loc='bottom',
            bbox=[0, -0.3, 1, 0.2],
            cellLoc='center',
            colColours=['#f5f5f5']*4,
            cellColours=[['#f9f9f9' if i%2==0 else '#ffffff' for _ in range(4)] 
                        for i in range(len(cell_text))]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Adjust layout to make room for the table
        plt.subplots_adjust(bottom=0.25)
    
    # Add performance insights if metrics are provided
    if metrics_df is not None:
        add_performance_insights(ax1, strategy_data, metrics_df)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for insights
    
    # Save figure if path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()

def generate_metrics_table(strategy_data: Dict[str, pd.Series]) -> pd.DataFrame:
    """Generate a table with performance metrics for all strategies."""
    metrics_data = []
    
    for name, values in strategy_data.items():
        metrics = calculate_metrics(values)
        if metrics:
            metrics['Strategy'] = name
            metrics_data.append(metrics)
    
    if not metrics_data:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    df = df.set_index('Strategy')
    
    # Format percentages
    percent_cols = ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Max Drawdown (%)']
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.1%}" if not pd.isna(x) else "N/A")
    
    # Format ratios
    ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    
    return df

def main():
    # Create output directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("Loading strategy data...")
    strategy_data = load_strategy_data()
    
    if not strategy_data:
        print("No strategy data found. Please run the backtest first.")
        return
    
    print("\nGenerating performance metrics...")
    metrics_table = generate_metrics_table(strategy_data)
    
    print("\nGenerating comparison plot...")
    plot_strategy_comparison(
        strategy_data,
        metrics_df=metrics_table,
        save_path='plots/forecasting_models_vs_benchmarks.png'
    )
    
    if metrics_table is not None:
        print("\nPerformance Metrics:")
        print("-" * 100)
        print(metrics_table)
        
        # Save metrics to CSV
        metrics_path = 'data/forecasting_models_metrics.csv'
        metrics_table.to_csv(metrics_path)
        
        # Print key metrics comparison
        print("\nKey Performance Comparison:")
        print("-" * 80)
        if 'Buy & Hold' in metrics_table.index and 'XGBoost GMV' in metrics_table.index:
            bh_ret = float(metrics_table.loc['Buy & Hold', 'Total Return'].strip('%'))/100
            xg_ret = float(metrics_table.loc['XGBoost GMV', 'Total Return'].strip('%'))/100
            print(f"Buy & Hold Return: {bh_ret:.1%}")
            print(f"Best Model Return: {xg_ret:.1%}")
            print(f"Performance Gap: {abs(bh_ret - xg_ret):.1%} in favor of {'Buy & Hold' if bh_ret > xg_ret else 'Model'}")
        
        print(f"\nSaved detailed metrics to {metrics_path}")
        
        print("\nDone!")

if __name__ == "__main__":
    main()
