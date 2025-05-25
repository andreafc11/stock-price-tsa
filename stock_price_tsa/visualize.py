import os
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from stock_price_tsa.analysis import (
    moving_average, volatility, linear_regression_trend, 
    bollinger_bands, rsi, exponential_moving_average, detect_outliers
)

def setup_plot_style():
    """Set up a professional plot style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.figsize': (15, 10),
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

def visualise_stock_data(
    stock_file: str, 
    window_size: int = 30, 
    save_path: Optional[str] = None,
    include_volume: bool = True,
    include_bollinger: bool = False,
    include_rsi: bool = False
) -> None:
    """
    Enhanced stock data visualization with multiple analysis options.

    Args:
        stock_file (str): Path to the CSV file with stock data.
        window_size (int): Window size for rolling calculations.
        save_path (str, optional): If given, save the plot to this path.
        include_volume (bool): Whether to include volume subplot.
        include_bollinger (bool): Whether to include Bollinger Bands.
        include_rsi (bool): Whether to include RSI subplot.
    """
    # Validate file path
    if not os.path.isfile(stock_file):
        raise FileNotFoundError(f"File not found: {stock_file}")

    # Read and validate data
    stock_data = pd.read_csv(stock_file)
    required_columns = {'Date', 'Close'}
    if not required_columns.issubset(stock_data.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    if window_size < 1 or window_size > len(stock_data):
        raise ValueError(f"window_size must be between 1 and length of data ({len(stock_data)})")

    # Prepare data
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    stock_data.sort_index(inplace=True)  # Ensure chronological order
    
    closing_prices = stock_data['Close']
    has_volume = 'Volume' in stock_data.columns
    has_ohlc = all(col in stock_data.columns for col in ['Open', 'High', 'Low'])

    # Calculate analytics
    ma = moving_average(closing_prices, window_size)
    ema = exponential_moving_average(closing_prices, window_size)
    vol = volatility(closing_prices, window_size)
    trend, slopes = linear_regression_trend(closing_prices.values, window_size)
    outliers = detect_outliers(closing_prices)
    
    # Optional analytics
    bb_ma, bb_upper, bb_lower = None, None, None
    rsi_values = None
    
    if include_bollinger:
        bb_ma, bb_upper, bb_lower = bollinger_bands(closing_prices, window_size)
    
    if include_rsi:
        rsi_values = rsi(closing_prices, 14)

    # Setup plot
    setup_plot_style()
    
    # Determine subplot layout
    n_subplots = 2  # Price and volatility
    if include_volume and has_volume:
        n_subplots += 1
    if include_rsi:
        n_subplots += 1
    
    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 4 * n_subplots))
    if n_subplots == 1:
        axes = [axes]
    
    subplot_idx = 0
    
    # Main price plot
    ax_price = axes[subplot_idx]
    ax_price.plot(closing_prices.index, closing_prices, label='Close Price', color='#2E86AB', linewidth=1.2)
    ax_price.plot(closing_prices.index, ma, label=f'{window_size}-Day MA', color='#A23B72', alpha=0.8)
    ax_price.plot(closing_prices.index, ema, label=f'{window_size}-Day EMA', color='#F18F01', alpha=0.8)
    
    if include_bollinger and bb_ma is not None:
        ax_price.plot(closing_prices.index, bb_upper, label='Bollinger Upper', color='red', alpha=0.5, linestyle='--')
        ax_price.plot(closing_prices.index, bb_lower, label='Bollinger Lower', color='red', alpha=0.5, linestyle='--')
        ax_price.fill_between(closing_prices.index, bb_upper, bb_lower, alpha=0.1, color='red')
    
    # Highlight outliers
    outlier_data = closing_prices[outliers]
    if not outlier_data.empty:
        ax_price.scatter(outlier_data.index, outlier_data.values, color='red', s=30, alpha=0.7, label='Outliers')
    
    # Add trend line
    trend_series = pd.Series(trend, index=closing_prices.index)
    valid_trend = trend_series.dropna()
    if not valid_trend.empty:
        ax_price.plot(valid_trend.index, valid_trend.values, label='Linear Trend', color='#C73E1D', alpha=0.8, linewidth=2)
    
    stock_symbol = os.path.basename(stock_file).split("_")[0].upper()
    ax_price.set_title(f'{stock_symbol} - Stock Price Analysis', fontsize=16, fontweight='bold')
    ax_price.set_ylabel('Price ($)', fontsize=12)
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)
    
    subplot_idx += 1
    
    # Volatility plot
    ax_vol = axes[subplot_idx]
    ax_vol.plot(closing_prices.index, vol, label=f'{window_size}-Day Volatility', color='#F18F01', linewidth=1.2)
    ax_vol.fill_between(closing_prices.index, vol, alpha=0.3, color='#F18F01')
    ax_vol.set_title('Price Volatility', fontsize=14)
    ax_vol.set_ylabel('Volatility ($)', fontsize=12)
    ax_vol.legend()
    ax_vol.grid(True, alpha=0.3)
    
    subplot_idx += 1
    
    # Volume plot (if available and requested)
    if include_volume and has_volume:
        ax_volume = axes[subplot_idx]
        volume_data = stock_data['Volume']
        ax_volume.bar(volume_data.index, volume_data.values, alpha=0.6, color='#2E86AB', width=1)
        ax_volume.set_title('Trading Volume', fontsize=14)
        ax_volume.set_ylabel('Volume', fontsize=12)
        ax_volume.grid(True, alpha=0.3)
        subplot_idx += 1
    
    # RSI plot (if requested)
    if include_rsi and rsi_values is not None:
        ax_rsi = axes[subplot_idx]
        ax_rsi.plot(rsi_values.index, rsi_values.values, label='RSI (14)', color='#A23B72', linewidth=1.2)
        ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax_rsi.fill_between(rsi_values.index, 30, 70, alpha=0.1, color='gray')
        ax_rsi.set_title('Relative Strength Index (RSI)', fontsize=14)
        ax_rsi.set_ylabel('RSI', fontsize=12)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend()
        ax_rsi.grid(True, alpha=0.3)
    
    # Format x-axis for all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    axes[-1].set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()

def generate_summary_report(stock_file: str, window_size: int = 30) -> Dict[str, Any]:
    """
    Generate a summary statistics report for the stock data.
    
    Args:
        stock_file (str): Path to the CSV file with stock data.
        window_size (int): Window size for calculations.
    
    Returns:
        Dict[str, Any]: Summary statistics and analysis results.
    """
    stock_data = pd.read_csv(stock_file)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    stock_data.sort_index(inplace=True)
    
    closing_prices = stock_data['Close']
    
    # Basic statistics
    report = {
        'symbol': os.path.basename(stock_file).split("_")[0].upper(),
        'period': {
            'start': closing_prices.index.min().strftime('%Y-%m-%d'),
            'end': closing_prices.index.max().strftime('%Y-%m-%d'),
            'days': len(closing_prices)
        },
        'price_stats': {
            'min': closing_prices.min(),
            'max': closing_prices.max(),
            'mean': closing_prices.mean(),
            'median': closing_prices.median(),
            'std': closing_prices.std(),
            'current': closing_prices.iloc[-1],
            'change_pct': ((closing_prices.iloc[-1] - closing_prices.iloc[0]) / closing_prices.iloc[0]) * 100
        },
        'volatility_stats': {
            'avg_volatility': volatility(closing_prices, window_size).mean(),
            'max_volatility': volatility(closing_prices, window_size).max(),
        },
        'trend_analysis': {
            'overall_trend': 'Upward' if closing_prices.iloc[-1] > closing_prices.iloc[0] else 'Downward',
            'recent_trend_slope': linear_regression_trend(closing_prices.tail(window_size).values, min(window_size, len(closing_prices)))[1][-1] if len(closing_prices) >= window_size else None
        }
    }
    
    return report