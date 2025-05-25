import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Union, Tuple
import sys
import os

# Add the root folder (one level up) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import load_config

config = load_config("stock_analysis_config.yaml")

def moving_average(data: pd.Series, window_size: int) -> pd.Series:
    """
    Calculate the rolling moving average.

    Args:
        data (pd.Series): Time series data (e.g. closing prices).
        window_size (int): Number of periods for the moving average.

    Returns:
        pd.Series: Moving average values with NaNs for initial periods.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    return data.rolling(window=window_size, min_periods=1).mean()

def volatility(data: pd.Series, window_size: int) -> pd.Series:
    """
    Calculate the rolling volatility (standard deviation).

    Args:
        data (pd.Series): Time series data.
        window_size (int): Window size for rolling std dev.

    Returns:
        pd.Series: Rolling volatility with NaNs for initial periods.
    """
    if window_size < 2:
        raise ValueError("Window size must be at least 2 for volatility calculation")
    return data.rolling(window=window_size, min_periods=2).std()

def linear_regression_trend(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling linear regression trend and slope.

    Args:
        data (np.ndarray): 1D array of values.
        window_size (int): Number of points in the rolling window.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays with trend values and slopes.
    """
    if window_size < 2:
        raise ValueError("Window size must be at least 2 for linear regression")
    
    trend = np.full(len(data), np.nan)  # To store predicted last point of each rolling regression
    slopes = np.full(len(data), np.nan) # To store slope of each regression
    
    for i in range(window_size - 1, len(data)):
        start_idx = i - window_size + 1
        y = data[start_idx:i+1]  # Current window data points
        x = np.arange(len(y))    # Independent variable, 0 to window_size-1
        
        # Calculate means
        x_mean = x.mean()
        y_mean = y.mean()
        
        # Calculate slope = covariance(x,y) / variance(x)
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        
        # Intercept from mean values
        intercept = y_mean - slope * x_mean
        
        # Trend = predicted value at last point in window
        trend[i] = slope * (len(y) - 1) + intercept
        slopes[i] = slope
    
    return trend, slopes


def bollinger_bands(data: pd.Series, window_size: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        data (pd.Series): Time series data.
        window_size (int): Window size for moving average and std dev.
        num_std (float): Number of standard deviations for bands.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: Moving average, upper band, lower band.
    """
    ma = moving_average(data, window_size)
    std = volatility(data, window_size)
    
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    
    return ma, upper_band, lower_band

def rsi(data: pd.Series, window_size: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data (pd.Series): Time series data (closing prices).
        window_size (int): Period for RSI calculation.

    Returns:
        pd.Series: RSI values (0-100).
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window_size, min_periods=1).mean()
    avg_loss = loss.rolling(window=window_size, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def exponential_moving_average(data: pd.Series, span: int) -> pd.Series:
    """
    Calculate exponential moving average.

    Args:
        data (pd.Series): Time series data.
        span (int): Span for exponential smoothing.

    Returns:
        pd.Series: Exponential moving average.
    """
    return data.ewm(span=span, adjust=False).mean()

def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in time series data.

    Args:
        data (pd.Series): Time series data.
        method (str): Method for outlier detection ('iqr' or 'zscore').
        threshold (float): Threshold for outlier detection.

    Returns:
        pd.Series: Boolean series indicating outliers.
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")