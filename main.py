import os
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

def setup_plot_style():
    
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'figure.figsize': (14, 7),
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

setup_plot_style()

def ensure_data_dir():
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

def save_dataframe(df: pd.DataFrame, filename: str, index: bool = True):
    
    ensure_data_dir()
    df.to_csv(filename, index=index)
    print(f"✓ Saved: {filename}")

def plot_equity_and_drawdown(results: dict, title: str, safe_name: str, first_rebalance_date: pd.Timestamp = None):

    os.makedirs('plots', exist_ok=True)

    portfolio_values = results['portfolio_value']
    if first_rebalance_date is not None:
        portfolio_values = portfolio_values[portfolio_values.index >= first_rebalance_date]

    plt.figure(figsize=(12, 6))
    portfolio_values.plot(label='Strategy', color='#2ecc71', linewidth=2)
    plt.title(f'Equity Curve: {title}', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Portfolio Value', fontsize=12, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    equity_path = f'plots/{safe_name}_equity.png'
    plt.savefig(equity_path, dpi=300, bbox_inches='tight')
    plt.close()

    daily_returns = results['daily_returns']
    if first_rebalance_date is not None:
        daily_returns = daily_returns[daily_returns.index >= first_rebalance_date]
    
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / (1 + running_max)
    
    plt.figure(figsize=(12, 4))
    drawdown.plot(color='#e74c3c', linewidth=1.5)
    plt.fill_between(drawdown.index, drawdown.values, color='#e74c3c', alpha=0.2)
    plt.title(f'Drawdown: {title}', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Drawdown', fontsize=12, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    drawdown_path = f'plots/{safe_name}_drawdown.png'
    plt.savefig(drawdown_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved drawdown plot to {drawdown_path}")

START_DATE = '2023-01-01'  
END_DATE = '2025-07-01'    
ESTIMATION_WINDOW = 252  
REBALANCE_FREQ = 'M'     
FORECAST_HORIZON = 50    
TRANSACTION_COSTS = [0.0005, 0.001, 0.002, 0.005]  
MAX_WEIGHT = 0.1         
BOOTSTRAP_PATHS = 1000   
RISK_FREE_RATE = 0.05    

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
    
    print(f"Downloading price data from {START_DATE} to {END_DATE}...")
    data = yf.download(NIFTY50_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
    data = data.dropna(axis=1, how='any')
    print(f"Successfully downloaded data for {len(data.columns)} stocks")
    return data

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    
    return np.log(prices / prices.shift(1)).dropna()

def ensure_data_dir():
    
    os.makedirs('data', exist_ok=True)

def save_data(prices: pd.DataFrame, filename: str = 'data/stock_data.csv'):
    
    ensure_data_dir()
    prices.to_csv(filename)
    print(f"Data saved to {filename}")

def save_log_returns(log_returns: pd.DataFrame, filename: str = 'data/log_returns.csv') -> None:
    
    ensure_data_dir()
    log_returns.to_csv(filename)
    print(f"Log returns saved to {filename}")

def load_log_returns(filename: str = 'data/log_returns.csv') -> pd.DataFrame:
    
    try:
        log_returns = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Log returns loaded from {filename}")
        return log_returns
    except FileNotFoundError:
        print(f"No log returns file found at {filename}")
        return None

def load_data(filename: str = 'data/stock_data.csv') -> pd.DataFrame:
    
    try:
        prices = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Data loaded from {filename}")
        return prices
    except FileNotFoundError:
        print(f"No data file found at {filename}")
        return None

def prepare_data(use_cached: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    
    prices = None
    log_returns = None

    if use_cached:
        log_returns = load_log_returns()

    if log_returns is None or log_returns.empty:
        if use_cached:
            prices = load_data()

        if prices is None or prices.empty:
            prices = download_prices()
            save_data(prices)

        log_returns = compute_log_returns(prices)
        save_log_returns(log_returns)

    rebalance_dates = log_returns.resample(REBALANCE_FREQ).last().index

    return prices if prices is not None else None, log_returns, rebalance_dates

def forecast_arima_garch(returns: pd.Series, n_paths: int = BOOTSTRAP_PATHS, 
                        horizon: int = FORECAST_HORIZON) -> Tuple[float, float]:
    
    try:

        model = ARIMA(returns, order=(2,0,1)).fit()
        residuals = model.resid

        garch = arch_model(residuals, vol='Garch', p=1, q=1).fit(disp='off')

        const = model.params.get('const', 0)
        phi1, phi2 = model.params.get('ar.L1', 0), model.params.get('ar.L2', 0)
        theta1 = model.params.get('ma.L1', 0)

        r1, r2 = returns.iloc[-1], returns.iloc[-2]
        e1 = residuals.iloc[-1]

        sim_returns = np.zeros((n_paths, horizon))
        sim_resid = np.random.normal(0, np.sqrt(garch.conditional_volatility[-1]), (n_paths, horizon))
        
        for t in range(horizon):
            if t == 0:
                sim_returns[:, t] = const + phi1*r1 + phi2*r2 + theta1*e1 + sim_resid[:, t]
            elif t == 1:
                sim_returns[:, t] = const + phi1*sim_returns[:, t-1] + phi2*r1 + theta1*sim_resid[:, t-1] + sim_resid[:, t]
            else:
                sim_returns[:, t] = const + phi1*sim_returns[:, t-1] + phi2*sim_returns[:, t-2] + theta1*sim_resid[:, t-1] + sim_resid[:, t]

        terminal_returns = np.sum(sim_returns, axis=1)
        mean_return = np.mean(terminal_returns)
        volatility = np.std(terminal_returns)
        
        return mean_return, volatility
        
    except Exception as e:
        print(f"Error in ARIMA-GARCH forecasting: {e}")
        return np.nan, np.nan

def forecast_xgboost(returns: pd.Series, lags: int = 11, 
                    horizon: int = FORECAST_HORIZON) -> Tuple[float, float]:
    
    try:

        df = pd.DataFrame(returns)
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = returns.shift(i)
        
        df = df.dropna()
        X = df.drop(columns=[returns.name])
        y = df[returns.name]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X, y)

        forecasts = []
        current_features = X.iloc[-1].values.reshape(1, -1)
        
        for _ in range(horizon):

            pred = model.predict(current_features)[0]
            forecasts.append(pred)

            current_features = np.roll(current_features, 1)
            current_features[0, 0] = pred

        cumulative_returns = np.cumsum(forecasts)
        mean_return = cumulative_returns[-1]  
        volatility = np.std(forecasts) * np.sqrt(horizon)  
        
        return mean_return, volatility
        
    except Exception as e:
        print(f"Error in XGBoost forecasting: {e}")
        return np.nan, np.nan

def get_forecast_matrices(returns: pd.DataFrame, model_type: str = 'arima') -> Tuple[np.ndarray, np.ndarray]:
    
    tickers = returns.columns
    n_assets = len(tickers)

    mu_forecasts = np.zeros(n_assets)
    return_paths = np.zeros((n_assets, BOOTSTRAP_PATHS))

    for i, ticker in enumerate(tqdm(tickers, desc=f"Generating {model_type.upper()} forecasts")):
        if model_type.lower() == 'arima':
            mu, sigma = forecast_arima_garch(returns[ticker])
        else:  # xgb
            mu, sigma = forecast_xgboost(returns[ticker])
            
        mu_forecasts[i] = mu

        return_paths[i] = np.random.normal(mu, sigma, BOOTSTRAP_PATHS)

    sigma_mat = np.cov(return_paths)
    
    return mu_forecasts, sigma_mat

def optimize_portfolio(mu: np.ndarray, sigma: np.ndarray, 
                      objective: str = 'gmv', 
                      max_weight: float = MAX_WEIGHT) -> np.ndarray:
    
    n_assets = len(mu)

    print("\n--- Optimization Diagnostics ---")
    print(f"Expected returns (μ) stats: mean={mu.mean():.2e}, std={mu.std():.2e}, min={mu.min():.2e}, max={mu.max():.2e}")
    print(f"Covariance matrix diagonal (σ²) stats: mean={np.diag(sigma).mean():.2e}, std={np.diag(sigma).std():.2e}")

    if objective.lower() == 'gmv':
        def objective_function(weights):
            return weights.T @ sigma @ weights
    elif objective.lower() == 'gmir':
        def objective_function(weights):
            portfolio_return = weights.T @ mu
            portfolio_vol = np.sqrt(weights.T @ sigma @ weights)
            return -portfolio_return / portfolio_vol  
    else:
        raise ValueError("Objective must be 'gmv' or 'gmir'")

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  
    ]

    bounds = tuple((0, max_weight) for _ in range(n_assets))

    x0 = np.ones(n_assets) / n_assets

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

    results = {
        'portfolio_value': pd.Series(1.0, index=log_returns.index),
        'daily_returns': pd.Series(0.0, index=log_returns.index),
        'turnover': pd.Series(0.0, index=log_returns.index),
        'weights': pd.DataFrame(0.0, index=log_returns.index, columns=log_returns.columns),
        'mn75': pd.Series(0.0, index=log_returns.index, dtype=float),  
        'mn90': pd.Series(0.0, index=log_returns.index, dtype=float)   
    }

    current_weights = pd.Series(0.0, index=log_returns.columns)
    current_weights[:] = 1.0 / len(log_returns.columns)  

    print("\nInitial portfolio weights:")
    print(f"Sum: {current_weights.sum():.6f}")
    print(f"Min: {current_weights.min():.6f}, Max: {current_weights.max():.6f}")
    print(f"Number of assets: {len(current_weights)}")

    for i, date in enumerate(log_returns.index[ESTIMATION_WINDOW:], start=ESTIMATION_WINDOW):
        if date in rebalance_dates:
            print(f"\nRebalancing on {date.date()}")

        if i % 30 == 0 or date in rebalance_dates:
            print(f"Processing {date.date()} (day {i+1}/{len(log_returns)})", end='\r')

        train_data = log_returns.iloc[:i]

        if date in rebalance_dates:

            mu, sigma = get_forecast_matrices(train_data, model_type)

            new_weights = optimize_portfolio(mu, sigma, objective=objective)

            turnover = np.abs(new_weights - current_weights).sum()
            results['turnover'].loc[date] = turnover

            print("\nWeight distribution:")
            print(f"  Min: {new_weights.min():.2%}")
            print(f"  Max: {new_weights.max():.2%}")
            print(f"  Mean: {new_weights.mean():.2%}")
            print(f"  Std: {new_weights.std():.2%}")
            print(f"  # Assets > 0: {(new_weights > 1e-6).sum()}")
            print(f"  # Assets at max weight: {(new_weights >= MAX_WEIGHT*0.999).sum()}")

            sorted_weights = np.sort(new_weights)[::-1]
            cum_weights = np.cumsum(sorted_weights)
            n_75 = np.argmax(cum_weights >= 0.75) + 1
            n_90 = np.argmax(cum_weights >= 0.90) + 1
            results['mn75'].loc[date] = n_75
            results['mn90'].loc[date] = n_90
            print(f"  MN75%: {n_75}, MN90%: {n_90}")

            new_weights = new_weights / new_weights.sum()  
            print(f"\nNew weights assigned - Sum: {new_weights.sum():.6f}, "
                  f"Min: {new_weights.min():.6f}, Max: {new_weights.max():.6f}")
            print(f"Number of assets with weight > 0: {(new_weights > 1e-6).sum()}")

            results['weights'].loc[date, :] = new_weights

            current_weights = new_weights.copy()

        results['weights'].loc[date, :] = current_weights

        simple_returns = np.exp(log_returns.loc[date]) - 1
        daily_return = (current_weights * simple_returns).sum()
        results['daily_returns'].loc[date] = daily_return

        if i > ESTIMATION_WINDOW:  
            results['portfolio_value'].loc[date] = results['portfolio_value'].iloc[i-1] * (1 + daily_return)
        else:
            results['portfolio_value'].loc[date] = 1.0 * (1 + daily_return)

        if i < ESTIMATION_WINDOW + 5 or date in rebalance_dates:
            print(f"Day {i}: Portfolio Value = {results['portfolio_value'].iloc[i]:.6f}, "
                  f"Daily Return = {daily_return:.6f}")
    
    return results

def calculate_performance_metrics(portfolio_values: pd.Series, 
                               daily_returns: pd.Series, 
                               weights_history: pd.DataFrame = None,
                               benchmark_returns: pd.Series = None,
                               freq: int = 252) -> Dict[str, float]:
    
    if len(portfolio_values) < 2:
        return {}

    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    annualized_return = (1 + total_return) ** (freq / len(portfolio_values)) - 1

    if isinstance(daily_returns, pd.DataFrame):
        daily_vol = daily_returns.std().mean()
    else:
        daily_vol = daily_returns.std()
    annualized_vol = daily_vol * np.sqrt(freq)

    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.cummax()
    drawdowns = (cum_returns / peak - 1) * 100
    max_drawdown = drawdowns.min()

    excess_returns = daily_returns - (RISK_FREE_RATE / freq)

    annualized_excess_return = (1 + excess_returns).prod() ** (freq/len(daily_returns)) - 1
    annualized_vol = daily_returns.std() * np.sqrt(freq)

    sharpe_ratio = (annualized_excess_return / annualized_vol) if annualized_vol > 0 else 0

    information_ratio = float('nan')
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:

            aligned_benchmark = benchmark_returns.reindex(daily_returns.index, fill_value=0)

            if len(aligned_benchmark) > 10: 

                active_returns = daily_returns - aligned_benchmark

                tracking_error = active_returns.std() * np.sqrt(freq)

                portfolio_ann_return = (1 + daily_returns).prod() ** (freq/len(daily_returns)) - 1
                benchmark_ann_return = (1 + aligned_benchmark).prod() ** (freq/len(aligned_benchmark)) - 1

                if tracking_error > 1e-10:  
                    information_ratio = (portfolio_ann_return - benchmark_ann_return) / tracking_error

                print(f"\nInformation Ratio Calculation:")
                print(f"- Portfolio Ann. Return: {portfolio_ann_return:.4f}")
                print(f"- Benchmark Ann. Return: {benchmark_ann_return:.4f}")
                print(f"- Tracking Error: {tracking_error:.6f}")
                print(f"- Information Ratio: {information_ratio:.4f}")
                
        except Exception as e:
            print(f"Error calculating Information Ratio: {str(e)}")
            information_ratio = float('nan')

    arc = annualized_return * 100  
    mdd = max_drawdown
    asd = annualized_vol * 100  
    mir = (arc * abs(arc)) / (asd * abs(mdd)) if (asd != 0 and mdd != 0) else float('nan')

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
        'Total Return': total_return * 100,
        'Annualized Return': annualized_return * 100,
        'Annualized Volatility': annualized_vol * 100,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Information Ratio': information_ratio,
        'Modified IR': mir,
        'Mean Stocks (75% coverage)': mn75,
        'Mean Stocks (90% coverage)': mn90
    }
    
    return metrics

def save_strategy_returns(results: Dict[str, pd.Series], strategy_name: str):
    
    ensure_data_dir()
    filename = f'data/{strategy_name.lower().replace(" ", "_")}_returns.csv'

    returns_data = {}

    if 'portfolio_value' in results:
        returns_data['strategy'] = results['portfolio_value'].pct_change().dropna()

    for key in results:
        if key.startswith('portfolio_value_after_tc_'):
            tc_value = key.split('_')[-1].replace('bps', '')
            returns_data[f'strategy_tc_{tc_value}bps'] = results[key].pct_change().dropna()

    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        returns_df.to_csv(filename)
        print(f"Saved {strategy_name} returns to {filename}")

def plot_backtest_results(results: Dict[str, pd.Series], title: str = 'Portfolio Performance', 
                         save_returns: bool = True, strategy_name: str = None):
    
    plt.figure(figsize=(14, 8))

    plt.plot(results['portfolio_value'], label='Strategy', linewidth=2)

    for key in results:
        if key.startswith('portfolio_value_after_tc_'):

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

    if save_returns and strategy_name:
        save_strategy_returns(results, strategy_name)

def apply_transaction_costs(portfolio_values: pd.Series, turnover_series: pd.Series, 
                          rebalance_dates: pd.DatetimeIndex, tc_rate: float) -> pd.Series:
    
    adjusted_values = portfolio_values.copy()
    for date in rebalance_dates:
        if date in adjusted_values.index and date in turnover_series.index:

            adjusted_values.loc[date:] *= (1 - tc_rate * turnover_series[date])
    return adjusted_values

def save_backtest_results(results: dict, strategy_name: str, first_rebalance_date: pd.Timestamp = None):

    safe_name = strategy_name.lower().replace('-', '_').replace(' ', '_')

    portfolio_values = results['portfolio_value']
    daily_returns = results['daily_returns']
    
    if first_rebalance_date is not None:
        portfolio_values = portfolio_values[portfolio_values.index >= first_rebalance_date]
        daily_returns = daily_returns[daily_returns.index >= first_rebalance_date]

    portfolio_df = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'daily_returns': daily_returns
    })
    save_dataframe(portfolio_df, f'data/portfolio_{safe_name}.csv')

    if 'weights' in results and results['weights'] is not None:
        weights = results['weights']
        if first_rebalance_date is not None:
            weights = weights[weights.index >= first_rebalance_date]
        save_dataframe(weights, f'data/weights_{safe_name}.csv')

    if 'metrics' in results:
        metrics_df = pd.DataFrame([results['metrics']])
        save_dataframe(metrics_df, f'data/metrics_{safe_name}.csv')

    plot_equity_and_drawdown(results, strategy_name, safe_name, first_rebalance_date)
    
    print(f"✓ Saved complete backtest results for {strategy_name} to data/ directory")
    
    print(f"✓ Saved complete backtest results for {strategy_name} to data/ directory")

def main(use_cached_data: bool = True):

    all_results = {}
    all_metrics_list = []
    
    print("Preparing data...")
    _, log_returns, rebalance_dates = prepare_data(use_cached=use_cached_data)

    strategies = [
        ('arima_garch', 'gmir', 'ARIMA-GARCH GMIR'),
        ('arima_garch', 'gmv', 'ARIMA-GARCH GMV'),
        ('xgboost', 'gmir', 'XGBoost GMIR'),
        ('xgboost', 'gmv', 'XGBoost GMV'),
    ]

    first_rebalance_date = log_returns.index[ESTIMATION_WINDOW]

    rebalance_dates = pd.date_range(start=first_rebalance_date, end=END_DATE, freq=REBALANCE_FREQ)

    benchmark_returns = log_returns.mean(axis=1)
    benchmark_returns = benchmark_returns[first_rebalance_date:]  # Align with backtest period
    print(f"\nCalculated benchmark returns from {benchmark_returns.index[0]} to {benchmark_returns.index[-1]}")
    print(f"Benchmark mean return: {benchmark_returns.mean():.6f}, std: {benchmark_returns.std():.6f}")

    for model_type, objective, strategy_name in strategies:
        print(f"\n{'='*50}")
        print(f"Running backtest for {strategy_name}...")
        print(f"Model: {model_type}, Objective: {objective.upper()}")
        print(f"{'='*50}")

        results = run_backtest(
            log_returns=log_returns,
            rebalance_dates=rebalance_dates,
            model_type=model_type,
            objective=objective
        )

        print(f"\nCalculating performance metrics for {strategy_name}...")
        metrics = calculate_performance_metrics(
            portfolio_values=results['portfolio_value'],
            daily_returns=results['daily_returns'],
            benchmark_returns=benchmark_returns,  # Use the pre-calculated benchmark
            weights_history=results.get('weights')
        )

        metrics['strategy'] = strategy_name
        metrics['model'] = model_type
        metrics['objective'] = objective

        all_results[strategy_name] = results
        all_metrics_list.append(metrics)

        results['metrics'] = metrics

        save_backtest_results(results, strategy_name, first_rebalance_date)

        print(f"\n{strategy_name} Performance:")
        print("-" * 50)
        for metric, value in metrics.items():
            if metric not in ['strategy', 'model', 'objective']:  # Skip metadata in printout
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")

    if all_metrics_list:

        metrics_df = pd.DataFrame(all_metrics_list)

        metrics_df.to_csv('data/all_metrics.csv', index=False)
        print("\n✓ Saved combined metrics to data/all_metrics.csv")

        plt.figure(figsize=(14, 10))
        metrics_to_plot = ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown']
        
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 2, i)
            sns.barplot(x='strategy', y=metric, data=metrics_df, hue='objective')
            plt.title(metric)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
        comparison_plot_path = 'plots/strategy_comparison.png'
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved strategy comparison plot to {comparison_plot_path}")

        plt.figure(figsize=(14, 8))
        for strategy_name, results in all_results.items():
            equity_curve = results['portfolio_value']
            equity_curve = equity_curve[equity_curve.index >= first_rebalance_date]
            plt.plot(equity_curve, label=strategy_name, linewidth=2)
        
        plt.title('Strategy Comparison - Equity Curves', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12, labelpad=10)
        plt.ylabel('Portfolio Value', fontsize=12, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        equity_plot_path = 'plots/strategy_comparison_equity.png'
        plt.savefig(equity_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved equity curve comparison plot to {equity_plot_path}")

        print("\nStrategy Comparison:")
        print("-" * 50)
        print(metrics_df[['strategy', 'objective', 'Total Return', 'Annualized Return', 
                         'Sharpe Ratio', 'Max Drawdown', 'Information Ratio']].to_string(index=False))
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run portfolio optimization with optional data caching.')
    parser.add_argument('--no-cache', action='store_true', help='Force download fresh data')
    args = parser.parse_args()
    
    main(use_cached_data=not args.no_cache)