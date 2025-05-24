# stock-price-tsa

A Python package for stock price time series analysis including moving average, volatility, and trend visualization.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 run.py data/AAPL_2006-01-01_to_2018-01-01.csv -w 60 --save
```

## Tests

Run tests with:

```bash
pytest tests/
```

## License

MIT License
