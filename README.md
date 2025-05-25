# Stock Price Time Series Analysis (TSA)

A comprehensive Python package for advanced stock price time series analysis, featuring technical indicators, trend analysis, volatility calculations, and professional visualizations.

## Features

### Technical Analysis
- **Moving Averages**: Simple Moving Average (SMA) and Exponential Moving Average (EMA)
- **Volatility Analysis**: Rolling standard deviation calculations
- **Trend Analysis**: Linear regression-based trend detection with slope calculations
- **Technical Indicators**: 
  - Bollinger Bands with customizable standard deviation multipliers
  - Relative Strength Index (RSI) for momentum analysis
- **Outlier Detection**: IQR and Z-score based outlier identification

### Visualization
- **Multi-panel Charts**: Price, volatility, volume, and technical indicators
- **Professional Styling**: Clean, publication-ready plots with seaborn styling
- **Interactive Options**: Configurable plot elements and technical overlays
- **Export Capabilities**: High-resolution PNG/PDF output with customizable DPI

### Reporting
- **Summary Statistics**: Comprehensive price and volatility metrics
- **Trend Analysis**: Overall and recent trend direction with quantitative measures
- **Export Formats**: JSON, CSV, and HTML report generation
- **Performance Metrics**: Total returns, volatility measures, and risk statistics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/andreafc11/stock-price-tsa.git
cd stock
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
Analyze a single stock file with default settings:
```bash
python3 run.py data/AAPL_2006-01-01_to_2018-01-01.csv
```

### Advanced Usage

#### Custom Window Size
```bash
python3 run.py data/AAPL.csv --window 60
```

#### Save Plots and Reports
```bash
python3 run.py data/AAPL.csv --save --output-dir results
```

#### Include Technical Indicators
```bash
python3 run.py data/AAPL.csv --bollinger --rsi --volume
```

#### Generate Comprehensive Analysis
```bash
python3 run.py data/*.csv -w 50 --save --bollinger --rsi --volume --report -o analysis_results
```

#### Report-Only Mode
```bash
python3 run.py data/AAPL.csv --report --report-only
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--window` | `-w` | Window size for moving averages and volatility | 30 |
| `--save` | `-s` | Save plots and reports to files | False |
| `--output-dir` | `-o` | Output directory for saved files | output |
| `--volume` | | Include volume analysis subplot | False |
| `--bollinger` | | Include Bollinger Bands overlay | False |
| `--rsi` | | Include RSI technical indicator | False |
| `--report` | | Generate summary statistics report | False |
| `--report-only` | | Generate only report, skip visualization | False |
| `--verbose` | `-v` | Enable detailed logging output | False |

## Configuration

The application supports YAML and JSON configuration files for advanced customization:

### Create Default Configuration
```bash
python3 -c "from config import create_default_config; create_default_config()"
```

### Configuration Options
- **Analysis Parameters**: Window sizes, technical indicator settings, outlier detection methods
- **Visualization Settings**: Plot styling, colors, figure dimensions, DPI settings
- **Output Configuration**: File formats, directory structure, naming templates

### Example Configuration (stock_analysis_config.yaml)
```yaml
analysis:
  window_size: 60
  volatility_window: 30
  rsi_period: 14
  bollinger_std: 2.0
  outlier_method: zscore

visualization:
  figure_size: [16, 10]
  dpi: 300
  include_volume: true
  include_bollinger: true
  include_rsi: true

output:
  save_plots: true
  save_reports: true
  output_dir: results
  report_format: json
```

## Data Format

Input CSV files must contain the following columns:
- `Date`: Date in YYYY-MM-DD format
- `Close`: Closing price (required)
- `Open`: Opening price (optional, for OHLC analysis)
- `High`: Daily high price (optional)
- `Low`: Daily low price (optional)
- `Volume`: Trading volume (optional, for volume analysis)

### Example Data Structure
```csv
Date,Open,High,Low,Close,Volume
2020-01-01,100.00,102.50,99.75,101.25,1500000
2020-01-02,101.50,103.00,100.50,102.75,1750000
...
```

## API Reference

### Core Analysis Functions
```python
from stock_price_tsa.analysis import (
    moving_average, volatility, linear_regression_trend,
    bollinger_bands, rsi, exponential_moving_average, detect_outliers
)

# Calculate 30-day moving average
ma = moving_average(price_series, window_size=30)

# Calculate Bollinger Bands
ma, upper_band, lower_band = bollinger_bands(price_series, window_size=20, num_std=2)

# Detect outliers using IQR method
outliers = detect_outliers(price_series, method='iqr', threshold=1.5)
```

### Visualization Functions
```python
from stock_price_tsa.visualize import visualise_stock_data, generate_summary_report

# Generate comprehensive visualization
visualise_stock_data(
    'data/AAPL.csv',
    window_size=30,
    save_path='output/AAPL_analysis.png',
    include_volume=True,
    include_bollinger=True,
    include_rsi=True
)

# Generate summary report
report = generate_summary_report('data/AAPL.csv', window_size=30)
```

## Testing

Run the comprehensive test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_analysis.py::TestAnalysisFunctions -v
pytest tests/test_analysis.py::TestVisualizationFunctions -v
pytest tests/test_analysis.py::TestIntegration -v
```

### Test Coverage
- Unit tests for all analysis functions
- Integration tests for complete workflows
- Data validation and edge case testing
- Visualization and reporting functionality tests

## Examples

### Multiple Stock Analysis
```bash
# Analyze multiple stocks with comprehensive settings
python3 run.py data/AAPL.csv data/GOOGL.csv data/MSFT.csv \
    --window 45 \
    --save \
    --bollinger \
    --rsi \
    --volume \
    --report \
    --output-dir portfolio_analysis \
    --verbose
```

### Batch Processing
```bash
# Process all CSV files in data directory
python3 run.py data/*.csv -w 60 --save --report -o batch_results
```

## Output Files

When using the `--save` option, the following files are generated:
- `{symbol}_plot.png`: Comprehensive visualization with all requested indicators
- `{symbol}_report.json`: Detailed analysis report with statistics and metrics
- Log files with processing information (when `--verbose` is enabled)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### Version 2.0
- Added comprehensive technical indicators (Bollinger Bands, RSI)
- Implemented advanced configuration system
- Enhanced visualization with multi-panel layouts
- Added professional reporting capabilities
- Improved error handling and logging
- Expanded test coverage with integration tests

### Version 1.0
- Basic moving average and volatility analysis
- Simple visualization capabilities
- Command-line interface

## Roadmap

- [ ] Additional technical indicators (MACD, Stochastic Oscillator)
- [ ] Real-time data integration with financial APIs
- [ ] Interactive web dashboard
- [ ] Portfolio-level analysis capabilities
- [ ] Machine learning-based trend prediction
- [ ] Backtesting framework for trading strategies
