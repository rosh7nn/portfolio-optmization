import os
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
from typing import Dict, Tuple, List
warnings.filterwarnings('ignore')

# Configuration
START_DATE = '2023-01-01'  # Start date for backtest
END_DATE = '2025-07-01'    # End date for backtest
ESTIMATION_WINDOW = 252  # 12 months of trading days
REBALANCE_FREQ = 'M'     # Monthly rebalancing
FORECAST_HORIZON = 50    # max(21 * f, 50) where f=1 month
TRANSACTION_COSTS = [0.0005, 0.001, 0.002, 0.005]  # 0.05%, 0.1%, 0.2%, 0.5%
MAX_WEIGHT = 0.1         # Maximum weight per stock
BOOTSTRAP_PATHS = 1000   # Number of bootstrap paths for simulation
RISK_FREE_RATE = 0.05    # Annual risk-free rate

# XGBoost parameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'learning_rate': 0.3,
    'reg_lambda': 1,
    'max_depth': 6,
    'subsample': 1.0,
    'seed': 42
}
NIFTY50_TICKERS = [
    'ADANIENT.NS','ADANIPORTS.NS','APOLLOHOSP.NS','ASIANPAINT.NS','AXISBANK.NS',
    'BAJAJ-AUTO.NS','BAJFINANCE.NS','BAJAJFINSV.NS','BEL.NS','BHARTIARTL.NS',
    'CIPLA.NS','COALINDIA.NS','DIVISLAB.NS','DRREDDY.NS','EICHERMOT.NS',
    'GRASIM.NS','HCLTECH.NS','HDFCBANK.NS','HDFCLIFE.NS','HEROMOTOCO.NS',
    'HINDALCO.NS','HINDUNILVR.NS','ICICIBANK.NS','INDUSINDBK.NS','INFY.NS',
    'ITC.NS','JIOFIN.NS','JSWSTEEL.NS','KOTAKBANK.NS','LT.NS',
    'M&M.NS','MARUTI.NS','NESTLEIND.NS','NTPC.NS','ONGC.NS',
    'POWERGRID.NS','RELIANCE.NS','SBILIFE.NS','SBIN.NS','SHRIRAMFIN.NS',
    'SUNPHARMA.NS','TATACONSUM.NS','TATAMOTORS.NS','TATASTEEL.NS','TCS.NS',
    'TECHM.NS','TITAN.NS','TRENT.NS','ULTRACEMCO.NS','WIPRO.NS'
]


def download_prices() -> pd.DataFrame:
    """Download adjusted close prices for NIFTY 50 stocks."""
    print(f"Downloading price data from {START_DATE} to {END_DATE}...")
    data = yf.download(NIFTY50_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
    data = data.dropna(axis=1, how='any')
    print(f"Successfully downloaded data for {len(data.columns)} stocks")
    return data

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price data."""
    return np.log(prices / prices.shift(1)).dropna()

def ensure_data_dir():
    """Ensure data directory exists."""
    os.makedirs('data', exist_ok=True)

def save_data(prices: pd.DataFrame, filename: str = 'data/stock_data.csv'):
    """
    Save price data to CSV.
    
    Args:
        prices: DataFrame containing price data
        filename: Path to save the CSV file
    """
    ensure_data_dir()
    prices.to_csv(filename)
    print(f"Data saved to {filename}")


def save_log_returns(log_returns: pd.DataFrame, filename: str = 'data/log_returns.csv') -> None:
    """
    Save log returns to CSV.
    
    Args:
        log_returns: DataFrame containing log returns
        filename: Path to save the CSV file
    """
    ensure_data_dir()
    log_returns.to_csv(filename)
    print(f"Log returns saved to {filename}")


def load_log_returns(filename: str = 'data/log_returns.csv') -> pd.DataFrame:
    """
    Load log returns from CSV.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        DataFrame containing the log returns or None if file not found
    """
    try:
        log_returns = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Log returns loaded from {filename}")
        return log_returns
    except FileNotFoundError:
        print(f"No log returns file found at {filename}")
        return None

def load_data(filename: str = 'data/stock_data.csv') -> pd.DataFrame:
    """
    Load price data from CSV.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        DataFrame containing the price data or None if file not found
    """
    try:
        prices = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Data loaded from {filename}")
        return prices
    except FileNotFoundError:
        print(f"No data file found at {filename}")
        return None

def prepare_data(use_cached: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """
    Download and prepare the dataset.
    
    Args:
        use_cached: If True, tries to load data from cache first
        
    Returns:
        Tuple of (prices, log_returns, rebalance_dates)
    """
    prices = None
    log_returns = None
    
    # Try to load cached log returns first
    if use_cached:
        log_returns = load_log_returns()
    
    # If log returns not found in cache, try to load prices
    if log_returns is None or log_returns.empty:
        if use_cached:
            prices = load_data()
        
        # If prices not in cache, download them
        if prices is None or prices.empty:
            prices = download_prices()
            save_data(prices)
        
        # Compute and save log returns
        log_returns = compute_log_returns(prices)
        save_log_returns(log_returns)
    
    # Get rebalance dates from log returns index
    rebalance_dates = log_returns.resample(REBALANCE_FREQ).last().index
    
    # Return prices if available, otherwise return None for prices
    return prices if prices is not None else None, log_returns, rebalance_dates


def forecast_arima_garch(returns: pd.Series, n_paths: int = BOOTSTRAP_PATHS, 
                        horizon: int = FORECAST_HORIZON) -> Tuple[float, float]:
    """
    Forecast mean and volatility using ARIMA(2,0,1)-GARCH(1,1) model with bootstrapping.
    
    Args:
        returns: Series of log returns
        n_paths: Number of bootstrap paths
        horizon: Forecast horizon in days
        
    Returns:
        Tuple of (mean_return, volatility) forecasts
    """
    try:
        # Fit ARIMA(2,0,1) model
        model = ARIMA(returns, order=(2,0,1)).fit()
        residuals = model.resid
        
        # Fit GARCH(1,1) model
        garch = arch_model(residuals, vol='Garch', p=1, q=1).fit(disp='off')
        
        # Get model parameters
        const = model.params.get('const', 0)
        phi1, phi2 = model.params.get('ar.L1', 0), model.params.get('ar.L2', 0)
        theta1 = model.params.get('ma.L1', 0)
        
        # Initial values
        r1, r2 = returns.iloc[-1], returns.iloc[-2]
        e1 = residuals.iloc[-1]
        
        # Simulate future paths
        sim_returns = np.zeros((n_paths, horizon))
        sim_resid = np.random.normal(0, np.sqrt(garch.conditional_volatility[-1]), (n_paths, horizon))
        
        for t in range(horizon):
            if t == 0:
                sim_returns[:, t] = const + phi1*r1 + phi2*r2 + theta1*e1 + sim_resid[:, t]
            elif t == 1:
                sim_returns[:, t] = const + phi1*sim_returns[:, t-1] + phi2*r1 + theta1*sim_resid[:, t-1] + sim_resid[:, t]
            else:
                sim_returns[:, t] = const + phi1*sim_returns[:, t-1] + phi2*sim_returns[:, t-2] + theta1*sim_resid[:, t-1] + sim_resid[:, t]
        
        # Calculate mean and volatility of terminal returns
        terminal_returns = np.sum(sim_returns, axis=1)
        mean_return = np.mean(terminal_returns)
        volatility = np.std(terminal_returns)
        
        return mean_return, volatility
        
    except Exception as e:
        print(f"Error in ARIMA-GARCH forecasting: {e}")
        return np.nan, np.nan



def forecast_xgboost(returns: pd.Series, lags: int = 11, 
                    horizon: int = FORECAST_HORIZON) -> Tuple[float, float]:
    """
    Forecast mean and volatility using XGBoost with recursive multi-step forecasting.
    
    Args:
        returns: Series of log returns
        lags: Number of lagged returns to use as features
        horizon: Forecast horizon in days
        
    Returns:
        Tuple of (mean_return, volatility) forecasts
    """
    try:
        # Prepare features and target
        df = pd.DataFrame(returns)
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = returns.shift(i)
        
        df = df.dropna()
        X = df.drop(columns=[returns.name])
        y = df[returns.name]
        
        # Train XGBoost model
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X, y)
        
        # Recursive forecasting
        forecasts = []
        current_features = X.iloc[-1].values.reshape(1, -1)
        
        for _ in range(horizon):
            # Make one-step forecast
            pred = model.predict(current_features)[0]
            forecasts.append(pred)
            
            # Update features for next prediction
            current_features = np.roll(current_features, 1)
            current_features[0, 0] = pred
        
        # Calculate mean and volatility of cumulative returns
        cumulative_returns = np.cumsum(forecasts)
        mean_return = cumulative_returns[-1]  # Total return over horizon
        volatility = np.std(forecasts) * np.sqrt(horizon)  # Annualized volatility
        
        return mean_return, volatility
        
    except Exception as e:
        print(f"Error in XGBoost forecasting: {e}")
        return np.nan, np.nan


def get_forecast_matrices(returns: pd.DataFrame, model_type: str = 'arima') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate forecasted return vector and covariance matrix.
    
    Args:
        returns: DataFrame of log returns (tickers in columns)
        model_type: 'arima' or 'xgb' for forecasting model
        
    Returns:
        Tuple of (expected_returns, covariance_matrix)
    """
    tickers = returns.columns
    n_assets = len(tickers)
    
    # Initialize arrays for forecasts
    mu_forecasts = np.zeros(n_assets)
    return_paths = np.zeros((n_assets, BOOTSTRAP_PATHS))
    
    # Generate forecasts for each asset
    for i, ticker in enumerate(tqdm(tickers, desc=f"Generating {model_type.upper()} forecasts")):
        if model_type.lower() == 'arima':
            mu, sigma = forecast_arima_garch(returns[ticker])
        else:  # xgb
            mu, sigma = forecast_xgboost(returns[ticker])
            
        mu_forecasts[i] = mu
        # Simulate returns using forecasted mean and volatility
        return_paths[i] = np.random.normal(mu, sigma, BOOTSTRAP_PATHS)
    
    # Calculate covariance matrix from simulated returns
    sigma_mat = np.cov(return_paths)
    
    return mu_forecasts, sigma_mat


def optimize_portfolio(mu: np.ndarray, sigma: np.ndarray, 
                      objective: str = 'gmv', 
                      max_weight: float = MAX_WEIGHT) -> np.ndarray:
    """
    Optimize portfolio weights using scipy.optimize.
    
    Args:
        mu: Expected returns vector
        sigma: Covariance matrix
        objective: 'gmv' for Global Minimum Variance, 'gmir' for Global Maximum Information Ratio
        max_weight: Maximum weight per asset
        
    Returns:
        Optimal portfolio weights
    """
    n_assets = len(mu)
    
    # Print diagnostic info
    print("\n--- Optimization Diagnostics ---")
    print(f"Expected returns (μ) stats: mean={mu.mean():.2e}, std={mu.std():.2e}, min={mu.min():.2e}, max={mu.max():.2e}")
    print(f"Covariance matrix diagonal (σ²) stats: mean={np.diag(sigma).mean():.2e}, std={np.diag(sigma).std():.2e}")
    
    # Define objective function
    if objective.lower() == 'gmv':
        def objective_function(weights):
            return weights.T @ sigma @ weights
    elif objective.lower() == 'gmir':
        def objective_function(weights):
            portfolio_return = weights.T @ mu
            portfolio_vol = np.sqrt(weights.T @ sigma @ weights)
            return -portfolio_return / portfolio_vol  # Negative for minimization
    else:
        raise ValueError("Objective must be 'gmv' or 'gmir'")
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum to 1
    ]
    
    # Bounds (0 <= weight <= max_weight)
    bounds = tuple((0, max_weight) for _ in range(n_assets))
    
    # Initial guess (equal weights)
    x0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        objective_function,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")
    
    weights = result.x
    
    # Print optimization results
    print(f"Optimal weights: min={weights.min():.2%}, max={weights.max():.2%}, mean={weights.mean():.2%}")
    print(f"Sum of weights: {weights.sum():.6f}")
    print(f"Number of assets with >0 weight: {(weights > 1e-6).sum()}")
    print(f"Number of assets at max weight ({(weights >= max_weight*0.999).sum()}/{len(weights)} at {max_weight:.1%})")
    
    return weights


def run_backtest(log_returns: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, 
                model_type: str = 'arima', objective: str = 'gmir') -> Dict[str, pd.Series]:
    print(f"\n{'='*80}\nStarting {model_type.upper()} {objective.upper()} backtest")
    print(f"Rebalance dates: {len(rebalance_dates)} points from {rebalance_dates[0].date()} to {rebalance_dates[-1].date()}")
    print(f"Number of assets: {len(log_returns.columns)}")
    print(f"Date range: {log_returns.index[0].date()} to {log_returns.index[-1].date()}")
    print(f"Total trading days: {len(log_returns)}")
    print(f"Estimation window: {ESTIMATION_WINDOW} days")
    print(f"Max weight constraint: {MAX_WEIGHT:.1%}\n{'='*80}")
    """
    Run backtest for a given forecasting model and optimization objective.
    
    Args:
        log_returns: DataFrame of log returns
        rebalance_dates: Dates to rebalance the portfolio
        model_type: 'arima' or 'xgb' for forecasting model
        objective: 'gmv' for Global Minimum Variance, 'gmir' for Global Maximum Information Ratio
        
    Returns:
        Dictionary with backtest results and metrics including:
        - portfolio_value: Portfolio values over time (without transaction costs)
        - daily_returns: Daily returns (without transaction costs)
        - turnover: Turnover at each rebalance date
        - weights: Portfolio weights at each time period
    """
    # Initialize results dictionary
    results = {
        'portfolio_value': pd.Series(1.0, index=log_returns.index),
        'daily_returns': pd.Series(0.0, index=log_returns.index),
        'turnover': pd.Series(0.0, index=log_returns.index),
        'weights': pd.DataFrame(0.0, index=log_returns.index, columns=log_returns.columns)
    }
    
    # Initialize portfolio weights (equal weight)
    current_weights = pd.Series(1.0/len(log_returns.columns), index=log_returns.columns)
    
    # Main backtest loop
    for i, date in enumerate(log_returns.index[ESTIMATION_WINDOW:], start=ESTIMATION_WINDOW):
        if date in rebalance_dates:
            print(f"\nRebalancing on {date.date()}")
        
        # Show progress every 30 days
        if i % 30 == 0 or date in rebalance_dates:
            print(f"Processing {date.date()} (day {i+1}/{len(log_returns)})", end='\r')
        
        # Get training data (expanding window)
        train_data = log_returns.iloc[:i]
        
        # Check if it's a rebalance date
        if date in rebalance_dates:
            # Get forecasted return and covariance matrices
            mu, sigma = get_forecast_matrices(train_data, model_type)
            
            # Optimize portfolio weights
            new_weights = optimize_portfolio(mu, sigma, objective=objective)
            
            # Calculate turnover (L1 norm of weight changes)
            turnover = np.abs(new_weights - current_weights).sum()
            results['turnover'].loc[date] = turnover
            
            # Print weight distribution stats
            print("\nWeight distribution:")
            print(f"  Min: {new_weights.min():.2%}")
            print(f"  Max: {new_weights.max():.2%}")
            print(f"  Mean: {new_weights.mean():.2%}")
            print(f"  Std: {new_weights.std():.2%}")
            print(f"  # Assets > 0: {(new_weights > 1e-6).sum()}")
            print(f"  # Assets at max weight: {(new_weights >= MAX_WEIGHT*0.999).sum()}")
            
            # Calculate concentration metrics
            sorted_weights = np.sort(new_weights)[::-1]
            cum_weights = np.cumsum(sorted_weights)
            n_75 = np.argmax(cum_weights >= 0.75) + 1
            n_90 = np.argmax(cum_weights >= 0.90) + 1
            print(f"  MN75%: {n_75}, MN90%: {n_90}")
            
            # Update weights
            current_weights = new_weights
        
        # Store current weights
        results['weights'].iloc[i] = current_weights
        
        # Calculate daily return
        daily_return = (current_weights * np.exp(log_returns.iloc[i])).sum() - 1
        results['daily_returns'].iloc[i] = daily_return
        
        # Update portfolio value
        results['portfolio_value'].iloc[i] = results['portfolio_value'].iloc[i-1] * (1 + daily_return)
    
    return results


def calculate_performance_metrics(portfolio_values: pd.Series, 
                               daily_returns: pd.Series, 
                               weights_history: pd.DataFrame = None,
                               freq: int = 252) -> Dict[str, float]:
    """
    Calculate performance metrics for the backtested portfolio.
    
    Args:
        portfolio_values: Series of portfolio values over time
        daily_returns: Series of daily returns
        weights_history: DataFrame containing portfolio weights over time (for MN75% and MN90%)
        freq: Number of trading days in a year
        
    Returns:
        Dictionary of performance metrics
    """
    if len(portfolio_values) < 2:
        return {}
        
    # Calculate returns
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    annualized_return = (1 + total_return) ** (freq / len(portfolio_values)) - 1
    
    # Calculate volatility
    if isinstance(daily_returns, pd.DataFrame):
        daily_vol = daily_returns.std().mean()
    else:
        daily_vol = daily_returns.std()
    annualized_vol = daily_vol * np.sqrt(freq)
    
    # Calculate drawdowns
    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.cummax()
    drawdowns = (cum_returns / peak - 1) * 100
    max_drawdown = drawdowns.min()
    
    # Calculate risk-adjusted returns
    excess_returns = daily_returns - RISK_FREE_RATE / freq
    sharpe_ratio = np.sqrt(freq) * excess_returns.mean() / daily_returns.std()
    sortino_ratio = np.sqrt(freq) * excess_returns.mean() / daily_returns[daily_returns < 0].std()
    
    # Calculate information ratio (vs. equal-weighted portfolio)
    if isinstance(daily_returns, pd.DataFrame):
        equal_weight_returns = daily_returns.mean(axis=1)
        tracking_error = (daily_returns.sub(equal_weight_returns, axis=0)).std().mean() * np.sqrt(freq)
        information_ratio = (annualized_return - equal_weight_returns.mean() * freq) / tracking_error if tracking_error != 0 else float('nan')
    else:
        information_ratio = float('nan')
    
    # Calculate Modified Information Ratio (MIR)
    arc = annualized_return * 100  # Convert to percentage for MIR calculation
    mdd = max_drawdown
    asd = annualized_vol * 100  # Convert to percentage for MIR calculation
    mir = (arc * abs(arc)) / (asd * abs(mdd)) if (asd != 0 and mdd != 0) else float('nan')
    
    # Calculate mean number of stocks for 75% and 90% portfolio coverage
    mn75, mn90 = float('nan'), float('nan')
    if weights_history is not None and not weights_history.empty:
        def get_min_stocks(weights, threshold):
            sorted_weights = weights.sort_values(ascending=False)
            cum_weight = 0
            for i, weight in enumerate(sorted_weights, 1):
                cum_weight += weight
                if cum_weight >= threshold:
                    return i
            return len(weights)
        
        mn75 = weights_history.apply(lambda x: get_min_stocks(x, 0.75), axis=1).mean()
        mn90 = weights_history.apply(lambda x: get_min_stocks(x, 0.90), axis=1).mean()
    
    metrics = {
        'Absolute Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_vol * 100,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Information Ratio': information_ratio,
        'Modified IR': mir,
        'Mean Stocks (75% coverage)': mn75,
        'Mean Stocks (90% coverage)': mn90
    }
    
    return metrics


def save_strategy_returns(results: Dict[str, pd.Series], strategy_name: str):
    """
    Save strategy returns to a CSV file.
    
    Args:
        results: Dictionary containing backtest results
        strategy_name: Name of the strategy (used for filename)
    """
    ensure_data_dir()
    filename = f'data/{strategy_name.lower().replace(" ", "_")}_returns.csv'
    
    # Create a DataFrame with all return series
    returns_data = {}
    
    # Add main portfolio returns
    if 'portfolio_value' in results:
        returns_data['strategy'] = results['portfolio_value'].pct_change().dropna()
    
    # Add transaction cost variants
    for key in results:
        if key.startswith('portfolio_value_after_tc_'):
            tc_value = key.split('_')[-1].replace('bps', '')
            returns_data[f'strategy_tc_{tc_value}bps'] = results[key].pct_change().dropna()
    
    # Save to CSV if we have any data
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        returns_df.to_csv(filename)
        print(f"Saved {strategy_name} returns to {filename}")


def plot_backtest_results(results: Dict[str, pd.Series], title: str = 'Portfolio Performance', 
                         save_returns: bool = True, strategy_name: str = None):
    """
    Plot the results of the backtest and optionally save returns to CSV.
    
    Args:
        results: Dictionary containing backtest results
        title: Plot title
        save_returns: Whether to save returns to CSV
        strategy_name: Name of the strategy (used for filename if save_returns is True)
    """
    plt.figure(figsize=(14, 8))
    
    # Plot main portfolio value
    plt.plot(results['portfolio_value'], label='Strategy', linewidth=2)
    
    # Plot portfolio values after transaction costs
    for key in results:
        if key.startswith('portfolio_value_after_tc_'):
            # Extract the numerical part before 'bps'
            tc_value = key.split('_')[-1].replace('bps', '')
            tc_bps = int(tc_value)
            plt.plot(results[key], '--', 
                   label=f'After {tc_bps/100:.2f}% TC', 
                   alpha=0.7)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Plot drawdowns if available
    if 'daily_returns' in results:
        cum_returns = (1 + results['daily_returns']).cumprod()
        drawdown = (cum_returns / cum_returns.cummax() - 1) * 100
        
        plt.figure(figsize=(14, 4))
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown, color='darkred', linewidth=1)
        plt.title('Drawdown', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    # Save returns to CSV if requested
    if save_returns and strategy_name:
        save_strategy_returns(results, strategy_name)


def ensure_data_dir():
    """Ensure data directory exists"""
    os.makedirs('data', exist_ok=True)

def apply_transaction_costs(portfolio_values: pd.Series, turnover_series: pd.Series, 
                          rebalance_dates: pd.DatetimeIndex, tc_rate: float) -> pd.Series:
    """
    Apply transaction costs to portfolio values based on turnover at rebalance dates.
    
    Args:
        portfolio_values: Raw portfolio values without transaction costs
        turnover_series: Series containing turnover at each rebalance date
        rebalance_dates: Dates when rebalancing occurred
        tc_rate: Transaction cost rate (e.g., 0.001 for 10bps)
        
    Returns:
        Series of portfolio values after applying transaction costs
    """
    adjusted_values = portfolio_values.copy()
    for date in rebalance_dates:
        if date in adjusted_values.index and date in turnover_series.index:
            # Apply transaction cost as a percentage of portfolio value
            adjusted_values.loc[date:] *= (1 - tc_rate * turnover_series[date])
    return adjusted_values


def save_backtest_results(results: dict, strategy_name: str):
    """Save backtest results to CSV files"""
    ensure_data_dir()
    
    # Save portfolio values
    portfolio_values = pd.DataFrame({
        'portfolio_value': results['portfolio_value'],
        'daily_returns': results['daily_returns']
    })
    
    # Add transaction cost variants
    for tc in TRANSACTION_COSTS:
        tc_portfolio = apply_transaction_costs(
            results['portfolio_value'],
            results['turnover'],
            results['turnover'][results['turnover'] > 0].index,  # Only rebalance dates
            tc
        )
        portfolio_values[f'portfolio_value_after_tc_{int(tc*10000)}bps'] = tc_portfolio
    
    portfolio_values.to_csv(f'data/{strategy_name}_values.csv')
    
    # Save weights if available
    if 'weights' in results and not results['weights'].empty:
        results['weights'].to_csv(f'data/{strategy_name}_weights.csv')
    
    # Save turnover if available
    if 'turnover' in results and not results['turnover'].empty:
        results['turnover'].to_csv(f'data/{strategy_name}_turnover.csv')
    
    # Save weights if available
    if 'weights_history' in results and results['weights_history'] is not None:
        results['weights_history'].to_csv(f'data/{strategy_name}_weights.csv')


def main(use_cached_data: bool = True):
    # Ensure data directory exists
    ensure_data_dir()
    
    print("Preparing data...")
    prices, log_returns, rebalance_dates = prepare_data(use_cached=use_cached_data)
    
    # Define strategies to test
    strategies = [
        ('arima_garch', 'gmir', 'ARIMA-GARCH GMIR'),
        ('arima_garch', 'gmv', 'ARIMA-GARCH GMV'),
        ('xgboost', 'gmir', 'XGBoost GMIR'),
        ('xgboost', 'gmv', 'XGBoost GMV'),
    ]
    
    all_metrics = {}
    all_results = {}
    
    for model_type, objective, strategy_name in strategies:
        print(f"\nRunning {strategy_name} strategy...")
        results = run_backtest(
            log_returns=log_returns,
            rebalance_dates=rebalance_dates,
            model_type=model_type,
            objective=objective
        )
        
        # Save results for final comparison plot
        all_results[strategy_name] = results
        
        # Save detailed results to CSV
        save_backtest_results(results, strategy_name)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(
            portfolio_values=results['portfolio_value'],
            daily_returns=results['daily_returns'],
            weights_history=results['weights']
        )
        all_metrics[strategy_name] = metrics
        
        # Print metrics
        print(f"\n{strategy_name} Performance:")
        print("-" * 50)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
            
        # Plot and save results
        plot_backtest_results(
            results, 
            title=f'{strategy_name} Performance',
            strategy_name=strategy_name
        )
        
        # Save metrics to master CSV after each strategy
        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
        metrics_df.to_csv('data/strategy_metrics_summary.csv')
        print(f"\nSaved detailed results for {strategy_name} to data/ directory")
    
    # Save all metrics to CSV
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df.to_csv('data/strategy_metrics_summary.csv')
    print("\nSaved strategy metrics to data/strategy_metrics_summary.csv")
    
    # Print metrics summary
    print("\nStrategy Comparison:")
    print("-" * 50)
    print(metrics_df.round(4))
    
    # Plot strategy comparison
    plt.figure(figsize=(14, 8))
    for strategy_name, results in all_results.items():
        plt.plot(results['portfolio_value'], label=strategy_name)
    
    plt.title('Strategy Comparison', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_path = 'data/strategy_comparison.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved strategy comparison plot to {comparison_plot_path}")
    
    # Show all plots at the end
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run portfolio optimization with optional data caching.')
    parser.add_argument('--no-cache', action='store_true', help='Force download fresh data')
    args = parser.parse_args()
    
    main(use_cached_data=not args.no_cache)
