import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from typing import Tuple, Dict
from portfolio_optimization.settings import BOOTSTRAP_PATHS, FORECAST_HORIZON, ARIMA_ORDER, XGB_PARAMS, XGB_LAGS


class ARIMAGARCH:
    def __init__(self, order: tuple = ARIMA_ORDER):
        self.order = order
        
    def forecast(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        try:
            # Initialize results
            n_assets = len(returns.columns)
            forecasts = pd.Series(index=returns.columns)
            
            # Process each asset separately
            for asset in returns.columns:
                asset_returns = returns[asset]
                
                # Fit ARIMA model
                arima_model = ARIMA(asset_returns, order=self.order).fit()
                
                # Get residuals for GARCH
                residuals = arima_model.resid
                
                # Fit GARCH model
                garch_model = arch_model(residuals, vol='Garch', p=1, q=1).fit(disp='off')
                
                # Extract parameters
                const = arima_model.params.get('const', 0)
                phi1, phi2 = arima_model.params.get('ar.L1', 0), arima_model.params.get('ar.L2', 0)
                theta1 = arima_model.params.get('ma.L1', 0)
                
                # Generate bootstrap paths
                r1, r2 = asset_returns.iloc[-1], asset_returns.iloc[-2]
                e1 = residuals.iloc[-1]
                
                sim_returns = np.zeros((BOOTSTRAP_PATHS, FORECAST_HORIZON))
                sim_resid = np.random.normal(0, np.sqrt(garch_model.params['omega']), 
                                           (BOOTSTRAP_PATHS, FORECAST_HORIZON))
                
                for t in range(FORECAST_HORIZON):
                    if t == 0:
                        sim_returns[:, t] = const + phi1*r1 + phi2*r2 + theta1*e1 + sim_resid[:, t]
                    elif t == 1:
                        sim_returns[:, t] = const + phi1*sim_returns[:, t-1] + phi2*r1 + theta1*sim_resid[:, t-1] + sim_resid[:, t]
                    else:
                        sim_returns[:, t] = const + phi1*sim_returns[:, t-1] + phi2*sim_returns[:, t-2] + theta1*sim_resid[:, t-1] + sim_resid[:, t]
                
                terminal_returns = np.sum(sim_returns, axis=1)
                mean_return = np.mean(terminal_returns)
                forecasts[asset] = mean_return
                
            # Calculate covariance matrix using historical returns
            cov_matrix = returns.cov()
            
            return forecasts, cov_matrix
            
        except Exception as e:
            print(f"Error in ARIMA-GARCH forecasting: {e}")
            return pd.Series(np.nan, index=returns.columns), returns.cov()

    def generate_forecasts(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate forecasts for portfolio optimization.
        
        Args:
            returns: DataFrame of historical returns for multiple assets
            
        Returns:
            Dictionary containing:
                'returns': Series of expected returns
                'covariance': DataFrame of covariance matrix
        """
        try:
            # Get forecasts and covariance
            returns_forecast, cov_matrix = self.forecast(returns)
            
            return {
                'returns': returns_forecast,
                'covariance': cov_matrix
            }
        except Exception as e:
            print(f"Error in generate_forecasts: {e}")
            return {
                'returns': pd.Series(np.nan, index=returns.columns),
                'covariance': returns.cov()
            }

class XGBoost:
    def __init__(self, params: dict = XGB_PARAMS, lags: int = XGB_LAGS):
        self.params = params
        self.lags = lags
        
    def forecast(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        try:
            # Initialize results
            n_assets = len(returns.columns)
            forecasts = pd.Series(index=returns.columns)
            
            # Process each asset separately
            for asset in returns.columns:
                asset_returns = returns[asset]
                
                # Create lagged features
                df = pd.DataFrame(asset_returns)
                for i in range(1, self.lags + 1):
                    df[f'lag_{i}'] = asset_returns.shift(i)
                df.dropna(inplace=True)
                
                # Prepare features and target
                X = df.iloc[:, 1:]
                y = df.iloc[:, 0]
                
                # Train XGBoost model
                model = xgb.XGBRegressor(**self.params)
                model.fit(X, y)
                
                # Make predictions
                latest_features = asset_returns.tail(self.lags).values.reshape(1, -1)
                prediction = model.predict(latest_features)[0]
                
                # Store forecast
                forecasts[asset] = prediction
                
            # Calculate covariance matrix using historical returns
            cov_matrix = returns.cov()
            
            return forecasts, cov_matrix
            
        except Exception as e:
            print(f"Error in XGBoost forecasting: {e}")
            return pd.Series(np.nan, index=returns.columns), returns.cov()

    def generate_forecasts(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate forecasts for portfolio optimization.
        
        Args:
            returns: DataFrame of historical returns for multiple assets
            
        Returns:
            Dictionary containing:
                'returns': Series of expected returns
                'covariance': DataFrame of covariance matrix
        """
        try:
            # Get forecasts and covariance
            returns_forecast, cov_matrix = self.forecast(returns)
            
            return {
                'returns': returns_forecast,
                'covariance': cov_matrix
            }
        except Exception as e:
            print(f"Error in generate_forecasts: {e}")
            return {
                'returns': pd.Series(np.nan, index=returns.columns),
                'covariance': returns.cov()
            }
            return np.nan, np.nan
