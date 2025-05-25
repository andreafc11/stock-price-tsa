import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from stock_price_tsa.analysis import (
    moving_average, volatility, linear_regression_trend,
    bollinger_bands, rsi, exponential_moving_average, detect_outliers
)
from stock_price_tsa.visualize import generate_summary_report

class TestAnalysisFunctions:
    """Test cases for analysis functions."""
    
    def setup_method(self):
        """Setup test data."""
        self.sample_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.price_data = pd.Series([100, 102, 98, 105, 103, 107, 104, 108, 106, 110])
    
    def test_moving_average_basic(self):
        """Test basic moving average calculation."""
        result = moving_average(self.sample_data, window_size=3)
        expected = pd.Series([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_moving_average_window_size_one(self):
        """Test that the moving_average function returns the original series when the window size is set to 1."""
        result = moving_average(self.sample_data, window_size=1)
        # Compare values, ignore dtype
        np.testing.assert_allclose(result.values, self.sample_data.values)
    
    def test_moving_average_invalid_window(self):
        """Test moving average with invalid window size."""
        with pytest.raises(ValueError):
            moving_average(self.sample_data, window_size=0)
    
    def test_volatility_basic(self):
        """Test basic volatility calculation."""
        result = volatility(self.price_data, window_size=3)
        assert not result.isna().all()
        assert result.iloc[-1] > 0  # Should have some volatility
    
    def test_volatility_invalid_window(self):
        """Test volatility with invalid window size."""
        with pytest.raises(ValueError):
            volatility(self.price_data, window_size=1)
    
    def test_linear_regression_trend(self):
        """Test linear regression trend calculation."""
        trend, slopes = linear_regression_trend(self.sample_data.values, window_size=3)
        
        # Should have NaN values at the beginning
        assert np.isnan(trend[:2]).all()
        # Should have actual values after the window
        assert not np.isnan(trend[2:]).any()
        # Trend should be generally increasing for increasing data
        assert slopes[-1] > 0
    
    def test_linear_regression_trend_invalid_window(self):
        """Test linear regression with invalid window size."""
        with pytest.raises(ValueError):
            linear_regression_trend(self.sample_data.values, window_size=1)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        ma, upper, lower = bollinger_bands(self.price_data, window_size=5, num_std=2)
        
        # Upper band should be above moving average
        valid_idx = ~ma.isna()
        assert (upper[valid_idx] >= ma[valid_idx]).all()
        # Lower band should be below moving average
        assert (lower[valid_idx] <= ma[valid_idx]).all()
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        result = rsi(self.price_data, window_size=5)
        
        # RSI should be between 0 and 100
        valid_rsi = result.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_exponential_moving_average(self):
        """Test exponential moving average."""
        result = exponential_moving_average(self.sample_data, span=3)
        
        # EMA should have no NaN values
        assert not result.isna().any()
        # EMA should be different from simple MA
        ma_result = moving_average(self.sample_data, window_size=3)
        assert not result.equals(ma_result)
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        # Create data with clear outliers
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
        result = detect_outliers(data_with_outliers, method='iqr', threshold=1.5)
        
        # Should detect the outlier (100)
        assert result.any()
        assert result.iloc[5]  # The outlier at position 5
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
        result = detect_outliers(data_with_outliers, method='zscore', threshold=2)
        
        # Should detect the outlier
        assert result.any()
        assert result.iloc[5]  # The outlier at position 5
    
    def test_detect_outliers_invalid_method(self):
        """Test outlier detection with invalid method."""
        with pytest.raises(ValueError):
            detect_outliers(self.sample_data, method='invalid')


class TestVisualizationFunctions:
    """Test cases for visualization functions."""
    
    def setup_method(self):
        """Setup test data and temporary files."""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_generate_summary_report(self):
        """Test summary report generation."""
        report = generate_summary_report(self.temp_file.name, window_size=10)
        
        # Check report structure
        assert 'symbol' in report
        assert 'period' in report
        assert 'price_stats' in report
        assert 'volatility_stats' in report
        assert 'trend_analysis' in report
        
        # Check period information
        assert 'start' in report['period']
        assert 'end' in report['period']
        assert 'days' in report['period']
        assert report['period']['days'] == 100
        
        # Check price statistics
        price_stats = report['price_stats']
        assert all(key in price_stats for key in ['min', 'max', 'mean', 'median', 'std', 'current', 'change_pct'])
        assert price_stats['max'] >= price_stats['min']
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualise_stock_data_display(self, mock_savefig, mock_show):
        """Test visualization without saving."""
        from stock_price_tsa.visualize import visualise_stock_data
        
        # Should not raise any exceptions
        visualise_stock_data(self.temp_file.name, window_size=10)
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualise_stock_data_save(self, mock_savefig, mock_show):
        """Test visualization with saving."""
        from stock_price_tsa.visualize import visualise_stock_data
        
        save_path = tempfile.mktemp(suffix='.png')
        visualise_stock_data(self.temp_file.name, window_size=10, save_path=save_path)
        
        mock_savefig.assert_called_once()
        mock_show.assert_not_called()
    
    def test_visualise_stock_data_file_not_found(self):
        """Test visualization with non-existent file."""
        from stock_price_tsa.visualize import visualise_stock_data
        
        with pytest.raises(FileNotFoundError):
            visualise_stock_data('non_existent_file.csv')
    
    def test_visualise_stock_data_missing_columns(self):
        """Test visualization with missing required columns."""
        from stock_price_tsa.visualize import visualise_stock_data
        
        # Create file with missing columns
        bad_data = pd.DataFrame({'Date': ['2020-01-01'], 'Price': [100]})
        temp_bad_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        bad_data.to_csv(temp_bad_file.name, index=False)
        temp_bad_file.close()
        
        try:
            with pytest.raises(ValueError, match="CSV file must contain columns"):
                visualise_stock_data(temp_bad_file.name)
        finally:
            os.unlink(temp_bad_file.name)
    
    def test_visualise_stock_data_invalid_window(self):
        """Test visualization with invalid window size."""
        from stock_price_tsa.visualize import visualise_stock_data
        
        with pytest.raises(ValueError, match="window_size must be between"):
            visualise_stock_data(self.temp_file.name, window_size=1000)


class TestDataValidation:
    """Test cases for data validation and edge cases."""
    
    def test_empty_data_series(self):
        """Test functions with empty data."""
        empty_series = pd.Series([], dtype=float)
        
        # Moving average should handle empty data gracefully
        result = moving_average(empty_series, window_size=5)
        assert len(result) == 0
    
    def test_single_value_series(self):
        """Test functions with single value."""
        single_value = pd.Series([100.0])
        
        # Moving average with single value
        result = moving_average(single_value, window_size=1)
        assert result.iloc[0] == 100.0
    
    def test_data_with_nans(self):
        """Test functions with NaN values in data."""
        data_with_nans = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])
        
        # Moving average should handle NaNs
        result = moving_average(data_with_nans, window_size=3)
        assert not result.isna().all()  # Should have some valid values
    
    def test_constant_data(self):
        """Test functions with constant data."""
        constant_data = pd.Series([100] * 10)
        
        # Volatility of constant data should be zero
        vol_result = volatility(constant_data, window_size=5)
        assert (vol_result.dropna() == 0).all()
        
        # Moving average should equal the constant value
        ma_result = moving_average(constant_data, window_size=5)
        assert (ma_result.dropna() == 100).all()


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create realistic stock data
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2020-01-01', periods=252, freq='D')  # One year of trading days
        
        # Generate realistic stock price data with trend and volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [100]  # Starting price
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.realistic_data = pd.DataFrame({
            'Date': dates,
            'Open': np.array(prices) * (1 + np.random.normal(0, 0.01, len(prices))),
            'High': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.015, len(prices)))),
            'Low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.015, len(prices)))),
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.5, len(dates)).astype(int)
        })
        
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.realistic_data.to_csv(self.test_file.name, index=False)
        self.test_file.close()
    
    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)
    
    @patch('matplotlib.pyplot.show')
    def test_full_analysis_workflow(self, mock_show):
        """Test the complete analysis workflow."""
        from stock_price_tsa.visualize import visualise_stock_data, generate_summary_report
        
        # Generate report
        report = generate_summary_report(self.test_file.name, window_size=20)
        
        # Verify report completeness
        assert report['period']['days'] == 252
        assert 'TEMPFILE' in report['symbol'] or report['symbol'] != ''
        
        # Generate visualization with all features
        visualise_stock_data(
            self.test_file.name,
            window_size=20,
            include_volume=True,
            include_bollinger=True,
            include_rsi=True
        )
        
        # Should complete without errors
        mock_show.assert_called_once()
    
    def test_multiple_window_sizes(self):
        """Test analysis with different window sizes."""
        from stock_price_tsa.analysis import moving_average, volatility
        
        closing_prices = self.realistic_data['Close']
        
        for window in [5, 10, 20, 50]:
            ma_result = moving_average(closing_prices, window)
            vol_result = volatility(closing_prices, window)
            
            # Results should be valid
            assert not ma_result.dropna().empty
            assert not vol_result.dropna().empty
            
            # Volatility should be positive for realistic data
            assert (vol_result.dropna() > 0).all(), f"Window {window}: Non-positive volatility detected"


# Fixtures for pytest
@pytest.fixture
def sample_stock_data():
    """Fixture providing sample stock data."""
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(50).cumsum() + 100,
        'High': np.random.randn(50).cumsum() + 102,
        'Low': np.random.randn(50).cumsum() + 98,
        'Close': np.random.randn(50).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 50)
    })
    
    return data


# Test runner configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v'])