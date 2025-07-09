import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

class ResultsPlotter:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.strategy_names = [
            'arima_garch_gmir',
            'arima_garch_gmv',
            'xgboost_gmir',
            'xgboost_gmv'
        ]
        
    def load_results(self) -> Dict[str, pd.DataFrame]:
        """Load all strategy results from CSV files"""
        results = {}
        for strategy in self.strategy_names:
            values_file = self.data_dir / f"{strategy}_values.csv"
            returns_file = self.data_dir / f"{strategy}_returns.csv"
            
            if values_file.exists() and returns_file.exists():
                values = pd.read_csv(values_file, index_col=0, parse_dates=True).squeeze()
                returns = pd.read_csv(returns_file, index_col=0, parse_dates=True).squeeze()
                results[strategy] = pd.DataFrame({
                    'values': values,
                    'returns': returns
                })
        return results
    
    def load_metrics(self) -> pd.DataFrame:
        """Load metrics summary"""
        metrics_file = self.data_dir / 'strategy_metrics_summary.csv'
        if metrics_file.exists():
            return pd.read_csv(metrics_file, index_col=0)
        return pd.DataFrame()
    
    def plot_equity_curves(self, results: Dict[str, pd.DataFrame], 
                          save_path: Optional[str] = None) -> None:
        """Plot equity curves for all strategies"""
        plt.figure(figsize=(14, 8))
        
        # Plot each strategy
        for name, df in results.items():
            # Clean up strategy name for display
            display_name = name.replace('_', ' ').title()
            df['values'].plot(label=display_name, linewidth=2)
        
        plt.title('Strategy Performance Comparison', fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=12, labelpad=10)
        plt.ylabel('Portfolio Value', fontsize=12, labelpad=10)
        plt.legend(fontsize=10, frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved equity curves to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics: pd.DataFrame, 
                              metrics_to_plot: List[str] = None,
                              save_path: Optional[str] = None) -> None:
        """Plot bar charts comparing metrics across strategies"""
        if metrics_to_plot is None:
            metrics_to_plot = [
                'Annualized Return (%)',
                'Annualized Volatility (%)',
                'Max Drawdown (%)',
                'Modified IR',
                'Mean Stocks (75% coverage)'
            ]
        
        metrics = metrics[metrics_to_plot]
        
        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            metrics[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(metric, fontsize=12)
            ax.set_xticklabels(metrics.index, rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add value labels on top of bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10),
                           textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics comparison to {save_path}")
        
        plt.show()

def main():
    # Initialize plotter
    plotter = ResultsPlotter()
    
    # Load results
    print("Loading results...")
    results = plotter.load_results()
    metrics = plotter.load_metrics()
    
    if not results:
        print("No results found. Please run the backtest first.")
        return
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot equity curves
    plotter.plot_equity_curves(
        results,
        save_path='plots/equity_curves.png'
    )
    
    # Plot metrics comparison
    if not metrics.empty:
        plotter.plot_metrics_comparison(
            metrics,
            save_path='plots/metrics_comparison.png'
        )
    
    print("\nAll plots have been generated in the 'plots' directory.")

if __name__ == "__main__":
    main()
