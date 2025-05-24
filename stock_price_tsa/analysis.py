import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def moving_average(data: pd.Series, window_size: int) -> pd.Series:
    """
    Calculate the rolling moving average.

    Args:
        data (pd.Series): Time series data (e.g. closing prices).
        window_size (int): Number of periods for the moving average.

    Returns:
        pd.Series: Moving average values with NaNs for initial periods.
    """
    return data.rolling(window=window_size).mean()

def volatility(data: pd.Series, window_size: int) -> pd.Series:
    """
    Calculate the rolling volatility (standard deviation).

    Args:
        data (pd.Series): Time series data.
        window_size (int): Window size for rolling std dev.

    Returns:
        pd.Series: Rolling volatility with NaNs for initial periods.
    """
    return data.rolling(window=window_size).std()

def linear_regression(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute rolling linear regression trend.

    Args:
        data (np.ndarray): 1D array of values.
        window_size (int): Number of points in the rolling window.

    Returns:
        np.ndarray: Array with trend predictions, NaN for initial window.
    """
    x = np.arange(window_size)
    lr = LinearRegression()
    trend = [np.nan]*window_size  # First window_size values have no trend
    
    for i in range(window_size, len(data)):
        y = data[i-window_size:i]
        lr.fit(x.reshape(-1, 1), y)
        trend.append(lr.predict([[window_size]])[0])  # Predict next point
    
    return np.array(trend)
