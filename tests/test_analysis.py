import pandas as pd
import numpy as np
from stock_price_tsa.analysis import moving_average

def test_moving_average():
    """
    Test moving_average function computes the correct rolling mean with given window size.
    The first value is NaN because the window is larger than initial data points.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    result = moving_average(data, window_size=2)
    expected = pd.Series([np.nan, 1.5, 2.5, 3.5, 4.5])
    pd.testing.assert_series_equal(result, expected)
