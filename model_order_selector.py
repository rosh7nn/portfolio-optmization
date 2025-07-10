import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def select_best_arima_order(series: pd.Series, p_range=(0, 3), d_range=(0, 1), q_range=(0, 3)) -> tuple:
    best_aic = np.inf
    best_order = (1, 0, 0)
    for p in range(p_range[0], p_range[1] + 1):
        for d in range(d_range[0], d_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                try:
                    model = ARIMA(series, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p, d, q)
                except:
                    continue
    return best_order

def select_best_xgb_lag(series: pd.Series, max_lag: int = 25) -> int:
    tscv = TimeSeriesSplit(n_splits=5)
    best_lag = 1
    best_score = np.inf

    for lag in range(2, max_lag + 1):
        df = pd.DataFrame(series)
        for i in range(1, lag + 1):
            df[f'lag_{i}'] = series.shift(i)
        df.dropna(inplace=True)

        X = df.drop(columns=[series.name])
        y = df[series.name].copy()

        fold_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.3)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_scores.append(mean_squared_error(y_test, preds))

        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_lag = lag

    return best_lag
