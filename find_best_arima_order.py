import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from tqdm import tqdm

def select_best_arima_order(series: pd.Series, p_range=(0, 3), d_range=(0, 1), q_range=(0, 3)) -> tuple:
    """
    Select the best (p, d, q) ARIMA order based on AIC.
    
    Args:
        series: Time series of returns
        p_range, d_range, q_range: ranges for ARIMA parameters to try
        
    Returns:
        Best (p, d, q) order tuple
    """
    best_aic = np.inf
    best_order = None
    
    for p, d, q in product(range(p_range[0], p_range[1]+1),
                           range(d_range[0], d_range[1]+1),
                           range(q_range[0], q_range[1]+1)):
        try:
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = (p, d, q)
        except:
            continue
    
    return best_order

def select_orders_for_all_assets(log_returns: pd.DataFrame, p_range=(0, 3), d_range=(0, 1), q_range=(0, 3)) -> dict:
    """
    Find the best ARIMA orders for all assets in the dataframe.
    
    Args:
        log_returns: DataFrame of asset log returns
        p_range, d_range, q_range: ranges for ARIMA parameters to try
        
    Returns:
        Dictionary {ticker: (p, d, q)}
    """
    best_orders = {}
    
    for ticker in tqdm(log_returns.columns, desc="Finding best ARIMA orders"):
        order = select_best_arima_order(log_returns[ticker], p_range, d_range, q_range)
        best_orders[ticker] = order
    
    return best_orders
