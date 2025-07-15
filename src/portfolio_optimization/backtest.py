import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from portfolio_optimization.settings import (
    START_DATE, END_DATE, REBALANCE_FREQ, ESTIMATION_WINDOW,
    NIFTY50_TICKERS, DATA_DIR, PLOTS_DIR, TRANSACTION_COSTS
)
from portfolio_optimization.forecasting.models import ARIMAGARCH, XGBoost
from portfolio_optimization.optimizer import PortfolioOptimizer

warnings.filterwarnings('ignore')

class BacktestEngine:
    def __init__(self, use_cached_data: bool = True):
        self.use_cached_data = use_cached_data
        self.optimizer = PortfolioOptimizer()
        self._ensure_directories()

    def run(self):
        """Run the complete backtest process."""
        # Prepare data
        prices, log_returns, rebalance_dates = self.prepare_data()
        
        # Run backtests and collect results
        results = {}
        for model_type in ['arima', 'xgboost']:
            for objective in ['gmir', 'gmv']:
                strategy = f"{model_type}_{objective}"
                print(f"\n{'='*100}\nStrategy: {strategy.upper()}\n{'='*100}")
                print(f"Rebalance Period: {rebalance_dates[0]} to {rebalance_dates[-1]}")
                print(f"Number of Rebalance Dates: {len(rebalance_dates)}")
                print(f"Number of Assets: {len(log_returns.columns)}")
                print(f"{'='*100}")
                
                results[strategy] = self.run_backtest(log_returns, rebalance_dates, model_type, objective)
                
                # Calculate strategy summary metrics
                self._calculate_strategy_summary(results[strategy], strategy)
                print(f"\n{'='*100}\nSummary Performance ({rebalance_dates[0]} to {rebalance_dates[-1]}):")
                print("- Cumulative Return: {:.2%}".format(results[strategy]['summary']['cumulative_return']))
                print("- Annualized Return: {:.2%}".format(results[strategy]['summary']['annualized_return']))
                print("- Annualized Volatility: {:.2%}".format(results[strategy]['summary']['annualized_volatility']))
                print("- Sharpe Ratio: {:.2f}".format(results[strategy]['summary']['sharpe_ratio']))
                print("- Max Drawdown: {:.2%}".format(results[strategy]['summary']['max_drawdown']))
                print(f"{'='*100}")
                
                # Print final summary and performance metrics
                self._print_summary_performance(results[strategy])
                self._print_monthly_rebalances(results[strategy])
                self._print_model_diagnostics(results[strategy], strategy.split('_')[0])
                self._print_weight_distribution(results[strategy])
                
                print(f"{'='*100}\n")
                
        # Print comparison table
        self._print_comparison_table(results)
        
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
    def _calculate_strategy_summary(self, results, strategy):
        """Calculate summary metrics for a strategy."""
        returns = results['daily_returns'].dropna()
        
        # Calculate cumulative return
        cum_return = (1 + returns).prod() - 1
        
        # Calculate annualized return
        n_days = len(returns)
        annualized_return = (1 + cum_return) ** (252/n_days) - 1
        
        # Calculate annualized volatility
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = annualized_return / annualized_vol
        
        # Calculate maximum drawdown
        portfolio_value = results['portfolio_value'].dropna()
        rolling_max = portfolio_value.rolling(window=len(portfolio_value), min_periods=1).max()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Store summary metrics
        results['summary'] = {
            'cumulative_return': cum_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
    def _print_monthly_rebalances(self, results):
        """Print monthly rebalances in a formatted table."""
        print("\nMonthly Rebalances:")
        print("=" * 120)
        print("| Month | Portfolio Value | Return | Volatility | Assets | Max Weight | Mean Weight |")
        print("|-------|-----------------|--------|------------|--------|------------|------------|")
        
        # Get all rebalance dates
        rebalance_dates = results['weights'].index[results['weights'].notna().any(axis=1)]
        
        # Group by month and calculate monthly returns
        monthly_data = []
        current_month = None
        current_month_dates = []
        monthly_value = []
        monthly_returns = []
        
        # Sort rebalance dates to ensure proper monthly grouping
        rebalance_dates = sorted(rebalance_dates)
        
        for date in rebalance_dates:
            if date.month != current_month:
                if current_month is not None:
                    # Calculate monthly return and volatility
                    monthly_return = (monthly_value[-1] / monthly_value[0] - 1) if len(monthly_value) > 0 else 0
                    monthly_vol = np.std(monthly_returns) * np.sqrt(252) if len(monthly_returns) > 1 else 0
                    
                    # Calculate weight distribution metrics using the first date of the month
                    weights = results['weights'].loc[current_month_dates[0]]
                    non_zero_weights = weights[weights > 0]
                    max_weight_assets = sum(weights == self.optimizer.max_weight)
                    mean_weight = weights[non_zero_weights.index].mean()
                    
                    monthly_data.append({
                        'month': current_month_dates[0].strftime('%Y-%m'),
                        'value': monthly_value[-1],
                        'return': monthly_return,
                        'volatility': monthly_vol,
                        'assets': len(non_zero_weights),
                        'max_weight': max_weight_assets,
                        'mean_weight': mean_weight
                    })
                
                current_month = date.month
                current_month_dates = []
                monthly_value = []
                monthly_returns = []
            
            current_month_dates.append(date)
            portfolio_value = results['portfolio_value'].loc[date]
            daily_return = results['daily_returns'].loc[date]
            monthly_value.append(portfolio_value)
            monthly_returns.append(daily_return)
            
        # Add the last month's data
        if len(monthly_value) > 0:
            monthly_return = (monthly_value[-1] / monthly_value[0] - 1)
            monthly_vol = np.std(monthly_returns) * np.sqrt(252) if len(monthly_returns) > 1 else 0
            
            # Calculate weight distribution metrics using the first date of the month
            weights = results['weights'].loc[current_month_dates[0]]
            non_zero_weights = weights[weights > 0]
            max_weight_assets = sum(weights == self.optimizer.max_weight)
            mean_weight = weights[non_zero_weights.index].mean()
            
            monthly_data.append({
                'month': current_month_dates[0].strftime('%Y-%m'),
                'value': monthly_value[-1],
                'return': monthly_return,
                'volatility': monthly_vol,
                'assets': len(non_zero_weights),
                'max_weight': max_weight_assets,
                'mean_weight': mean_weight
            })
        
        # Print the monthly data
        for data in monthly_data:
            print(f"| {data['month']} | {data['value']:14.4f} | {data['return']:6.2%} | {data['volatility']:10.2%} | {data['assets']:6} | {data['max_weight']:10} | {data['mean_weight']:10.2%} |")
        
        print("=" * 120)

    def _print_model_diagnostics(self, results, model_type):
        """Print model-specific diagnostics."""
        print("\nModel Diagnostics:")
        print("-" * 80)
        if model_type == 'xgboost':
            print("XGBoost Feature Importance:")
            print("-" * 80)
            # Add XGBoost specific diagnostics
        else:
            print("ARIMA-GARCH Parameters:")
            print("-" * 80)
            # Add ARIMA-GARCH specific diagnostics

    def _print_weight_distribution(self, results):
        """Print weight distribution metrics."""
        weights = results['weights'].iloc[-1]
        non_zero_weights = weights[weights > 0]
        
        print("\nWeight Distribution:")
        print("-" * 80)
        print(f"- Min: {non_zero_weights.min():.2%}")
        print(f"- Max: {non_zero_weights.max():.2%}")
        print(f"- Mean: {non_zero_weights.mean():.2%}")
        print(f"- Std: {non_zero_weights.std():.2%}")
        print(f"- Assets with Weight > 0: {len(non_zero_weights)}")
        print(f"- Assets at Max Weight (10%): {(weights >= 0.1).sum()}")
        print(f"- 75th Percentile Weight: {non_zero_weights.quantile(0.75):.2%} ({self._get_min_stocks(weights, 0.75)} assets)")
        print(f"- 90th Percentile Weight: {non_zero_weights.quantile(0.90):.2%} ({self._get_min_stocks(weights, 0.90)} assets)")
        print("Top Assets:")
        top_assets = non_zero_weights.nlargest(5)
        for asset, weight in top_assets.items():
            print(f"- {asset}: {weight:.2%}")
        print("-" * 80)

    def _print_summary_performance(self, results):
        """Print summary performance metrics."""
        summary = results['summary']
        print("\nSummary Performance:")
        print("=" * 80)
        print(f"Period: {results['portfolio_value'].index[0]} to {results['portfolio_value'].index[-1]}")
        print(f"- Cumulative Return: {summary['cumulative_return']:.2%}")
        print(f"- Annualized Return: {summary['annualized_return']:.2%}")
        print(f"- Annualized Volatility: {summary['annualized_volatility']:.2%}")
        print(f"- Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"- Max Drawdown: {summary['max_drawdown']:.2%}")
        print("=" * 80)
        
    def _print_comparison_table(self, results):
        """Print comparison table across all strategies."""
        print("\nComparison Across Strategies:")
        print("=" * 120)
        print("| Strategy          | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe Ratio |")
        print("|-------------------|-------------------|-------------------|-----------------------|--------------|")
        
        for strategy in ['xgboost_gmir', 'xgboost_gmv', 'arima_gmir', 'arima_gmv']:
            summary = results[strategy]['summary']
            print(f"| {strategy.upper():<20} | {summary['cumulative_return']:^14.2%} | {summary['annualized_return']:^14.2%} | {summary['annualized_volatility']:^18.2%} | {summary['sharpe_ratio']:^12.2f} |")
        print("=" * 120)
        
    def prepare_data(self):
        prices = None
        log_returns = None
        
        if self.use_cached_data:
            log_returns = self._load_log_returns()
        
        if log_returns is None or log_returns.empty:
            if self.use_cached_data:
                prices = self._load_data()
            if prices is None or prices.empty:
                prices = self._download_prices()
                self._save_data(prices)
            log_returns = self._compute_log_returns(prices)
            self._save_log_returns(log_returns)
        
        rebalance_dates = log_returns.resample(REBALANCE_FREQ).last().index
        return prices, log_returns, rebalance_dates
        
    def _download_prices(self):
        print(f"Downloading price data from {START_DATE} to {END_DATE}...")
        data = yf.download(NIFTY50_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
        data = data.dropna(axis=1, how='any')
        print(f"Successfully downloaded data for {len(data.columns)} stocks")
        return data
        
    def _save_data(self, prices, filename=f'{DATA_DIR}/stock_data.csv'):
        prices.to_csv(filename)
        print(f"Data saved to {filename}")
        
    def _load_data(self, filename=f'{DATA_DIR}/stock_data.csv'):
        try:
            prices = pd.read_csv(filename, index_col=0, parse_dates=True)
            if set(prices.columns) != set(NIFTY50_TICKERS):
                print(f"Warning: Loaded data columns do not match NIFTY50_TICKERS")
                return None
            print(f"Data loaded from {filename}")
            return prices
        except FileNotFoundError:
            print(f"No data file found at {filename}")
            return None
            
    def _compute_log_returns(self, prices):
        return np.log(prices / prices.shift(1)).dropna()
        
    def _save_log_returns(self, log_returns, filename=f'{DATA_DIR}/log_returns.csv'):
        log_returns.to_csv(filename)
        print(f"Log returns saved to {filename}")
        
    def _load_log_returns(self, filename=f'{DATA_DIR}/log_returns.csv'):
        try:
            returns = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Log returns loaded from {filename}")
            return returns
        except FileNotFoundError:
            print(f"No log returns file found at {filename}")
            return None
            
    def run_backtest(self, log_returns, rebalance_dates, model_type='arima', objective='gmir'):
        """Run portfolio backtest."""
        if log_returns.empty or len(rebalance_dates) == 0:
            raise ValueError("log_returns or rebalance_dates are empty")
            
        print(f"\n{'='*80}")
        print(f"Strategy: {model_type.upper()} {objective.upper()}")
        print(f"{'='*80}")
        print(f"Rebalance Period: {rebalance_dates[0]} to {rebalance_dates[-1]}")
        print(f"Number of Rebalance Dates: {len(rebalance_dates)}")
        print(f"Number of Assets: {len(log_returns.columns)}")
        print(f"{'='*80}\n")
        
        # Calculate total days since start
        total_days = (log_returns.index[-1] - log_returns.index[0]).days
        print(f"Total days in backtest period: {total_days}")
        print(f"{'='*80}\n")
        
        # Initialize results with full date range but only store values on rebalance dates
        date_range = pd.date_range(start=log_returns.index[0], end=log_returns.index[-1], freq='B')
        results = {
            'portfolio_value': pd.Series(index=date_range, dtype=float),
            'daily_returns': pd.Series(index=date_range, dtype=float),
            'daily_vol': pd.Series(index=date_range, dtype=float),
            'weights': pd.DataFrame(index=date_range, columns=log_returns.columns, dtype=float),
            'turnover': pd.Series(index=date_range, dtype=float),
            'mn75': pd.Series(index=date_range, dtype=float),
            'mn90': pd.Series(index=date_range, dtype=float),
            'cumulative_returns': pd.Series(index=date_range, dtype=float)
        }
        
        # Initialize current weights
        current_weights = np.ones(len(log_returns.columns)) / len(log_returns.columns)
        
        # Get rebalance dates from log returns index
        rebalance_dates = pd.date_range(start=log_returns.index[0], end=log_returns.index[-1], freq='BM')
        rebalance_dates = rebalance_dates[rebalance_dates.isin(log_returns.index)]
        
        # Pre-compute rebalance date indices
        rebalance_indices = {date: log_returns.index.get_loc(date) for date in rebalance_dates}
        
        # Pre-compute estimation window slices
        estimation_windows = {}
        for date in rebalance_dates:
            # Find the index position in log_returns
            log_returns_idx = log_returns.index.get_loc(date)
            start_idx = max(0, log_returns_idx - ESTIMATION_WINDOW)
            end_idx = log_returns_idx
            estimation_windows[date] = (start_idx, end_idx)
        
        # Initialize forecasting model
        if model_type.lower() == 'arima':
            forecaster = ARIMAGARCH()
        else:
            forecaster = XGBoost()
        
        # Run backtest
        for i, date in enumerate(date_range[ESTIMATION_WINDOW:], start=ESTIMATION_WINDOW):
            if date in log_returns.index:
                simple_returns = np.exp(log_returns.loc[date]) - 1
                daily_return = (current_weights * simple_returns).sum()
                results['daily_returns'].loc[date] = daily_return
                
                if i > ESTIMATION_WINDOW:
                    results['portfolio_value'].loc[date] = results['portfolio_value'].iloc[i-1] * (1 + daily_return)
                else:
                    results['portfolio_value'].loc[date] = 1.0 * (1 + daily_return)
            
            # Only track portfolio value and returns for all dates
            results['portfolio_value'].loc[date] = 1.0 * (1 + daily_return)
            results['daily_returns'].loc[date] = daily_return
            
            # Perform portfolio optimization on rebalance dates
            if date in rebalance_dates:
                # Get historical returns using pre-computed indices
                start_idx, end_idx = estimation_windows[date]
                hist_returns = log_returns.iloc[start_idx:end_idx]
                
                # Ensure we have enough data for the estimation window
                if len(hist_returns) < ESTIMATION_WINDOW:
                    print(f"Warning: Insufficient data for estimation window at {date}")
                    continue
                
                # Calculate mean and covariance
                mu = hist_returns.mean()
                sigma = hist_returns.cov()
                
                # Run optimization
                optimized_weights = self.optimizer.optimize(mu, sigma, objective)
                current_weights = optimized_weights
                results['weights'].loc[date] = current_weights
                
                # Calculate turnover if not first rebalance
                if i > ESTIMATION_WINDOW:
                    prev_weights = results['weights'].iloc[i-1].values
                    results['turnover'].loc[date] = np.sum(np.abs(current_weights - prev_weights))
                
                # Calculate MN75 and MN90
                mn75 = self._get_min_stocks(pd.Series(current_weights), 0.75)
                mn90 = self._get_min_stocks(pd.Series(current_weights), 0.90)
                results['mn75'].loc[date] = mn75
                results['mn90'].loc[date] = mn90
                
                # Forward fill weights until next rebalance
                if i < len(date_range) - 1:
                    results['weights'].iloc[i+1] = current_weights
                
                # Calculate portfolio metrics using forecasts
                portfolio_return = current_weights @ mu
                portfolio_vol = np.sqrt(current_weights @ sigma @ current_weights)
                
                # Calculate daily, monthly, and annualized metrics
                daily_return = portfolio_return
                monthly_return = (1 + daily_return) ** 21 - 1  # Assuming 21 trading days per month
                annualized_return = (1 + daily_return) ** 252 - 1
                annualized_vol = portfolio_vol * np.sqrt(252)
                
                # Store daily volatility
                results['daily_vol'].loc[date] = portfolio_vol
                
                # Update cumulative returns
                results['cumulative_returns'].loc[date] = results['portfolio_value'].loc[date] - 1
                
                # Print metrics with clearer labels
                print(f"\nPortfolio Metrics (as of {date}):")
                print(f"Daily Return: {daily_return:.2%}")
                print(f"Monthly Return: {monthly_return:.2%}")
                print(f"Annualized Return: {annualized_return:.2%}")
                print(f"Daily Volatility: {portfolio_vol:.2%}")
                print(f"Annualized Volatility: {annualized_vol:.2%}")
                
                # Calculate weight distribution metrics
                weights_series = pd.Series(current_weights)
                non_zero_weights = weights_series[weights_series > 0]
                max_weight_assets = sum(weights_series == self.optimizer.max_weight)
                
                # Print optimization results in a structured format
                print(f"\n{'='*50} Portfolio Optimization {'='*50}")
                print(f"Date: {date}")
                print(f"{'-'*50}")
                print(f"Portfolio Value: {results['portfolio_value'].loc[date]:.6f}")
                print(f"Portfolio Return: {daily_return:.4%}")
                print(f"Portfolio Volatility: {portfolio_vol:.4%}")
                print(f"{'-'*50}")
                print(f"--- Optimization Diagnostics ---")
                print(f"Expected returns (μ) stats: mean={mu.mean():.2e}, std={mu.std():.2e}, min={mu.min():.2e}, max={mu.max():.2e}")
                print(f"Covariance matrix diagonal (σ²) stats: mean={np.diag(sigma).mean():.2e}, std={np.diag(sigma).std():.2e}")
                print(f"{'-'*50}")
                print(f"Optimal weights: min={weights_series.min():.2%}, max={weights_series.max():.2%}, mean={weights_series.mean():.2%}")
                print(f"Sum of weights: {weights_series.sum():.6f}")
                print(f"Number of assets with >0 weight: {len(non_zero_weights)}")
                print(f"Number of assets at max weight ({(weights_series >= self.optimizer.max_weight*0.999).sum()}/{len(log_returns.columns)} at {self.optimizer.max_weight:.1%})")
                print(f"{'-'*50}")
                print(f"Weight distribution:")
                print(f"  Min: {weights_series.min():.2%}")
                print(f"  Max: {weights_series.max():.2%}")
                print(f"  Mean: {weights_series.mean():.2%}")
                print(f"  Std: {weights_series.std():.2%}")
                print(f"  # Assets > 0: {len(non_zero_weights)}")
                print(f"  # Assets at max weight: {(weights_series >= self.optimizer.max_weight*0.999).sum()}")
                print(f"  MN75%: {mn75}, MN90%: {mn90}")
                print(f"{'-'*50}")
                print(f"New weights assigned - Sum: {weights_series.sum():.6f}, Min: {weights_series.min():.6f}, Max: {weights_series.max():.6f}")
                print(f"Number of assets with weight > 0: {len(non_zero_weights)}")
                print(f"{'='*50}")
                
                # Daily performance
                if i > ESTIMATION_WINDOW:
                    print(f"Day {i}: Portfolio Value = {results['portfolio_value'].loc[date]:.6f}, Daily Return = {results['daily_returns'].loc[date]:.6f}")
            
            # Calculate cumulative returns
            results['cumulative_returns'].loc[date] = results['portfolio_value'].loc[date] - 1
        
        return results
        
    def calculate_performance_metrics(self, portfolio_values: pd.Series, 
                                    daily_returns: pd.Series, 
                                    weights_history: pd.DataFrame = None,
                                    benchmark_returns: pd.Series = None,
                                    freq: int = 252) -> Dict[str, float]:
        """
        Calculate performance metrics for the portfolio.
        
        Args:
            portfolio_values: Series of portfolio values
            daily_returns: Series of daily returns
            weights_history: DataFrame of portfolio weights history
            benchmark_returns: Series of benchmark returns
            freq: Number of trading days in a year
            
        Returns:
            Dictionary of performance metrics
        """
        if len(portfolio_values) < 2:
            return {}
            
        # Calculate basic metrics
        total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
        annualized_return = (1 + total_return) ** (freq / len(portfolio_values)) - 1
        daily_vol = daily_returns.std()
        annualized_vol = daily_vol * np.sqrt(freq)
        sharpe_ratio = annualized_return / annualized_vol
        
        # Calculate drawdown metrics
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate turnover metrics
        if weights_history is not None:
            turnover = weights_history.diff().abs().sum(axis=1).mean()
        else:
            turnover = 0
            
        # Calculate MN75 and MN90 metrics
        if weights_history is not None:
            mn75 = weights_history.apply(lambda x: self._get_min_stocks(x, 0.75)).mean()
            mn90 = weights_history.apply(lambda x: self._get_min_stocks(x, 0.90)).mean()
        else:
            mn75 = 0
            mn90 = 0
            
        metrics = {
            'Total Return': total_return * 100,
            'Annualized Return': annualized_return * 100,
            'Annualized Volatility': annualized_vol * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown * 100,
            'Turnover': turnover * 100,
            'Mean Stocks (75% coverage)': mn75,
            'Mean Stocks (90% coverage)': mn90
        }
        
        # Calculate benchmark metrics if provided
        if benchmark_returns is not None:
            excess_returns = daily_returns - benchmark_returns
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(freq)
            metrics['Information Ratio'] = information_ratio
            
            benchmark_total_return = benchmark_returns.add(1).prod() - 1
            benchmark_annualized_return = (1 + benchmark_total_return) ** (freq / len(benchmark_returns)) - 1
            benchmark_vol = benchmark_returns.std() * np.sqrt(freq)
            metrics['Benchmark Annualized Return'] = benchmark_annualized_return * 100
            metrics['Benchmark Annualized Volatility'] = benchmark_vol * 100
            
        return metrics
        
    def _get_min_stocks(self, weights: pd.Series, threshold: float) -> int:
        """
        Calculate minimum number of stocks needed to reach a coverage threshold.
        
        Args:
            weights: Series of portfolio weights
            threshold: Coverage threshold (e.g., 0.75 for MN75)
            
        Returns:
            Number of stocks needed to reach the threshold
        """
        if weights.sum() == 0:
            return 0
        sorted_weights = weights.sort_values(ascending=False)
        cumulative_sum = sorted_weights.cumsum()
        n_stocks = (cumulative_sum < threshold).sum()
        if n_stocks == 0:
            return 1
        if cumulative_sum.iloc[n_stocks] >= threshold:
            return n_stocks
            
        # Otherwise, include the stock that crosses the threshold
        return n_stocks + 1
        
    def plot_results(self, results: Dict[str, pd.Series], title: str, strategy_name: str):
        """
        Plot backtest results including equity curve, weights, and performance metrics.
        
        Args:
            results: Dictionary of backtest results
            title: Plot title
            strategy_name: Name of the strategy
        """
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.title(f"{title} - Equity Curve")
        plt.plot(results['portfolio_value'], label='Portfolio Value', color='#2ecc71', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/{strategy_name}_equity_curve.png')
        plt.close()
        
        # Plot weights history
        plt.figure(figsize=(12, 6))
        plt.title(f"{title} - Portfolio Weights")
        results['weights'].plot.area(stacked=True, cmap='viridis', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/{strategy_name}_weights.png')
        plt.close()
        
        # Plot turnover
        plt.figure(figsize=(12, 6))
        plt.title(f"{title} - Turnover")
        plt.plot(results['turnover'] * 100, label='Turnover', color='#e74c3c')
        plt.xlabel('Date')
        plt.ylabel('Turnover %')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/{strategy_name}_turnover.png')
        plt.close()
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'portfolio_value': results['portfolio_value'],
            'daily_returns': results['daily_returns'],
            'turnover': results['turnover'],
            'mn75': results['mn75'],
            'mn90': results['mn90']
        })
        results_df.to_csv(f'{PLOTS_DIR}/{strategy_name}_results.csv')
        
        # Save weights history
        results['weights'].to_csv(f'{PLOTS_DIR}/{strategy_name}_weights_history.csv')
        plt.title(f'Equity Curve: {title}', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12, labelpad=10)
        plt.ylabel('Portfolio Value', fontsize=12, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        equity_path = f'{PLOTS_DIR}/{strategy_name}_equity.png'
        plt.savefig(equity_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot drawdown
        plt.figure(figsize=(12, 4))
        drawdown = (portfolio_values / portfolio_values.cummax() - 1) * 100
        plt.fill_between(drawdown.index, drawdown.values, color='#e74c3c', alpha=0.2)
        plt.plot(drawdown, color='#e74c3c', linewidth=1.5)
        plt.title(f'Drawdown: {title}', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12, labelpad=10)
        plt.ylabel('Drawdown (%)', fontsize=12, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        drawdown_path = f'{PLOTS_DIR}/{strategy_name}_drawdown.png'
        plt.savefig(drawdown_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved equity curve plot to {equity_path}")
        print(f"✓ Saved drawdown plot to {drawdown_path}")
