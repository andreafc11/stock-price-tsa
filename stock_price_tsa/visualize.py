# stock_price_tsa/visualize.py
import os
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_price_tsa.analysis import moving_average, volatility, linear_regression

def visualise_stock_data(stock_file: str, window_size: int = 30, save_path: Optional[str] = None) -> None:
    """
    Load stock data from CSV, compute moving average, volatility, linear regression trend,
    and plot the results.

    Args:
        stock_file (str): Path to the CSV file with stock data.
        window_size (int, optional): Window size for rolling calculations. Defaults to 30.
        save_path (str, optional): If given, save the plot to this path instead of showing it.

    Raises:
        FileNotFoundError: If stock_file does not exist.
        ValueError: If required columns are missing or window_size is invalid.
    """
    # Validate file path
    if not os.path.isfile(stock_file):
        raise FileNotFoundError(f"File not found: {stock_file}")

    # Read CSV
    stock_data = pd.read_csv(stock_file)

    # Validate required columns
    required_columns = {'Date', 'Close'}
    if not required_columns.issubset(stock_data.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    # Validate window size
    if window_size < 1 or window_size > len(stock_data):
        raise ValueError(f"window_size must be between 1 and length of data ({len(stock_data)})")

    # Convert 'Date' column to datetime and set as index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    closing_prices = stock_data['Close']

    # Compute analytics
    ma = moving_average(closing_prices, window_size)
    vol = volatility(closing_prices, window_size)
    trend = linear_regression(closing_prices.values, window_size)

    # Plot setup
    plt.figure(figsize=(14, 8))
    plt.plot(closing_prices.index, closing_prices, label='Closing Price', color='blue')
    plt.plot(closing_prices.index, ma, label=f'{window_size}-Day Moving Average', color='orange')
    plt.plot(closing_prices.index, vol, label=f'{window_size}-Day Volatility', color='red')
    plt.plot(closing_prices.index, trend, label='Linear Regression Trend', color='green')

    plt.title(f'Stock Price Analysis for {os.path.basename(stock_file).split("_")[0]}')
    plt.xlabel('Date')
    plt.ylabel('Price / Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()
