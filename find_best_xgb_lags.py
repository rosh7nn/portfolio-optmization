import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

def create_lagged_features(series, n_lags):
    """Create a DataFrame with lagged features."""
    df = pd.DataFrame(series)
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = series.shift(i)
    df = df.dropna()
    return df

def evaluate_xgb_model(X, y, params, n_splits=5):
    """Evaluate model using TimeSeriesSplit and return average MSE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
    
    return np.mean(mse_scores)

def select_best_xgb_lags(series: pd.Series, lag_range=(5, 25), params=None, n_splits=5):
    """Select the best number of lags for a series using cross-validated MSE."""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.3,
            'reg_lambda': 1,
            'max_depth': 6,
            'subsample': 1.0,
            'seed': 42
        }

    best_lag = None
    best_score = np.inf

    for lags in range(lag_range[0], lag_range[1]+1):
        df = create_lagged_features(series, lags)
        if df.empty:
            continue

        X = df.drop(columns=[series.name])
        y = df[series.name]

        mse = evaluate_xgb_model(X, y, params, n_splits)

        if mse < best_score:
            best_score = mse
            best_lag = lags

    return best_lag, best_score

def select_lags_for_all_assets(log_returns: pd.DataFrame, lag_range=(5, 25), params=None, n_splits=5):
    """Find the best lag count for each asset's XGBoost forecast."""
    best_lags = {}
    for ticker in tqdm(log_returns.columns, desc="Finding best XGBoost lags"):
        best_lag, _ = select_best_xgb_lags(log_returns[ticker], lag_range, params, n_splits)
        best_lags[ticker] = best_lag
    return best_lags
