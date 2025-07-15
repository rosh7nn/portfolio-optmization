import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, List
from portfolio_optimization.settings import MAX_WEIGHT, RISK_FREE_RATE


class PortfolioOptimizer:
    def __init__(self, max_weight=MAX_WEIGHT, risk_free_rate=RISK_FREE_RATE):
        self.max_weight = max_weight
        self.risk_free_rate = risk_free_rate
        
    def optimize(self, mu, sigma, objective='gmv'):
        n_assets = len(mu)
        
        # Validate inputs
        if len(sigma) != n_assets:
            raise ValueError(f"Expected returns and covariance matrix dimensions don't match: {n_assets} vs {len(sigma)}")
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
            {'type': 'ineq', 'fun': lambda x: x - 0},  # Non-negative weights
            {'type': 'ineq', 'fun': lambda x: self.max_weight - x},  # Max weight constraint
            {'type': 'ineq', 'fun': lambda x: 1e-6 - np.min(x)},  # Minimum weight constraint
        ]
        
        # Define bounds
        bounds = [(0, self.max_weight) for _ in range(n_assets)]
        
        # Set initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Define objective function
        if objective.lower() == 'gmir':
            obj_func = lambda w: -(w.T @ mu / np.sqrt(w.T @ sigma @ w + 1e-8))
        elif objective.lower() == 'gmv':
            obj_func = lambda w: w.T @ sigma @ w
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Run optimization with retries
        max_retries = 3
        for attempt in range(max_retries):
            result = minimize(
                obj_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            weights = result.x
            
            if result.success and abs(weights.sum() - 1) < 1e-6:
                break
                
            if attempt == 0:
                x0 = np.random.dirichlet(np.ones(n_assets))
            else:
                x0 = weights / weights.sum()
        
        if not result.success:
            raise ValueError(f"Optimization failed after {max_retries} attempts: {result.message}")
            
        # Post-processing
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / weights.sum()
        
        return weights
        n_assets = len(mu)
        
        # Validate inputs
        if len(sigma) != n_assets:
            raise ValueError(f"Expected returns and covariance matrix dimensions don't match: {n_assets} vs {len(sigma)}")
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
            {'type': 'ineq', 'fun': lambda x: x - 0},  # Non-negative weights
            {'type': 'ineq', 'fun': lambda x: self.max_weight - x},  # Max weight constraint
            {'type': 'ineq', 'fun': lambda x: 1e-6 - np.min(x)},  # Minimum weight constraint
        ]
        
        # Define bounds
        bounds = [(0, self.max_weight) for _ in range(n_assets)]
        
        # Set initial guess - equal weights as starting point
        x0 = np.ones(n_assets) / n_assets
        
        # Define optimization function based on objective
        if objective.lower() == 'gmir':
            def objective_func(weights):
                """Maximize Information Ratio"""
                portfolio_return = weights @ mu
                portfolio_vol = np.sqrt(weights @ sigma @ weights)
                # Add small constant to avoid division by zero
                return -(portfolio_return / (portfolio_vol + 1e-8))
        elif objective.lower() == 'gmv':
            def objective_func(weights):
                """Minimize portfolio variance"""
                portfolio_var = weights @ sigma @ weights
                return portfolio_var
        else:
            raise ValueError(f"Unknown objective: {objective}. Must be 'gmv' or 'gmir'")
                
        # Run optimization with retries
        max_retries = 3
        for attempt in range(max_retries):
            result = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            weights = result.x
            
            # Validate results
            if result.success and abs(weights.sum() - 1) < 1e-6:
                break
                
            # If first attempt failed, try with different initial guess
            if attempt == 0:
                x0 = np.random.dirichlet(np.ones(n_assets))
            else:
                # If second attempt failed, try with scaled initial guess
                x0 = weights / weights.sum()
        
        # If optimization still failed after retries
        if not result.success:
            raise ValueError(f"Optimization failed after {max_retries} attempts: {result.message}")
            
        # Post-processing
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / weights.sum()
        
        # Debug print
        print(f"Optimization successful: {result.success}")
        print(f"Optimal weights sum: {weights.sum():.6f}")
        print(f"Top 5 weights: {pd.Series(weights).nlargest(5)}")
        
        
        return weights
        
        # Define objective function
        if objective.lower() == 'gmv':
            obj_func = lambda w: w.T @ sigma @ w
        elif objective.lower() == 'gmir':
            obj_func = lambda w: -w.T @ mu / np.sqrt(w.T @ sigma @ w)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Run optimization
        result = minimize(
            fun=obj_func,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
        
        weights = result.x
        
        # Print optimization statistics
        print(f"Optimal weights: min={weights.min():.2%}, max={weights.max():.2%}")
        print(f"Sum of weights: {weights.sum():.6f}")
        print(f"Number of assets with >0 weight: {(weights > 1e-6).sum()}")
        print(f"Number of assets at max weight ({(weights >= self.max_weight*0.999).sum()}/{n_assets} at {self.max_weight:.1%})")
        
        return weights
