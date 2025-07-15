import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Configurations
ESTIMATION_WINDOW = 252
MAX_WEIGHT = 0.1
RISK_FREE_RATE = 0.05
RESULTS_DIR = 'data'
os.makedirs(RESULTS_DIR, exist_ok=True)

def optimize_mv_portfolio(mu, sigma, objective='gmv'):
    if mu is None or sigma is None or len(mu) == 0 or len(sigma) == 0:
        raise ValueError("Invalid mu or sigma inputs")
    n_assets = len(mu)
    if objective == 'gmv':
        def objective_func(weights):
            return weights.T @ sigma @ weights
    elif objective == 'gmir':
        def objective_func(weights):
            port_return = weights.T @ mu
            port_vol = np.sqrt(weights.T @ sigma @ weights)
            return -port_return / port_vol if port_vol > 0 else np.inf
    else:
        raise ValueError("Objective must be 'gmv' or 'gmir'")
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, MAX_WEIGHT) for _ in range(n_assets))
    init_weights = np.ones(n_assets) / n_assets
    result = minimize(objective_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else init_weights

def calculate_mv_metrics(portfolio_values, daily_returns, freq=252):
    if len(portfolio_values) < 2 or daily_returns.empty:
        return {}
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    annualized_return = (1 + total_return) ** (freq / len(portfolio_values)) - 1
    annualized_vol = daily_returns.std() * np.sqrt(freq)
    sharpe = (annualized_return - RISK_FREE_RATE) / annualized_vol if annualized_vol > 0 else np.nan
    cum_returns = (1 + daily_returns).cumprod()
    max_dd = ((cum_returns / cum_returns.cummax()) - 1).min() * 100
    metrics = {
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_vol * 100,
        'Max Drawdown (%)': max_dd,
        'Sharpe Ratio': sharpe
    }
    return metrics

def run_mv_backtest(log_returns, rebalance_dates, objective='gmv'):
    if log_returns.empty or len(rebalance_dates) == 0:
        raise ValueError("log_returns or rebalance_dates are empty")
    print(f"\nStarting MV {objective.upper()} backtest")
    n_assets = log_returns.shape[1]
    portfolio_value = pd.Series(1.0, index=log_returns.index)
    daily_returns = pd.Series(0.0, index=log_returns.index)
    weights = np.ones(n_assets) / n_assets
    for i, date in enumerate(log_returns.index[ESTIMATION_WINDOW:], start=ESTIMATION_WINDOW):
        if date in rebalance_dates:
            train_data = log_returns.iloc[i-ESTIMATION_WINDOW:i]
            if train_data.empty or len(train_data) < ESTIMATION_WINDOW:
                print(f"Warning: Insufficient data for {date}, skipping rebalance")
                continue
            mu = train_data.mean().values
            sigma = train_data.cov().values
            weights = optimize_mv_portfolio(mu, sigma, objective)
        simple_returns = np.exp(log_returns.iloc[i]) - 1
        daily_ret = np.dot(weights, simple_returns)
        daily_returns.iloc[i] = daily_ret
        portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + daily_ret)
    results = {
        'portfolio_value': portfolio_value,
        'daily_returns': daily_returns
    }
    return results

def export_mv_metrics(results, strategy_name):
    metrics = calculate_mv_metrics(results['portfolio_value'], results['daily_returns'])
    metrics['Strategy'] = strategy_name
    df = pd.DataFrame([metrics])
    filename = os.path.join(RESULTS_DIR, 'mv_metrics.csv')
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_combined = df
    df_combined.to_csv(filename, index=False)
    print(f"✓ Metrics for {strategy_name} saved to {filename}")