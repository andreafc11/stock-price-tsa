import argparse
import os
import logging
from stock_price_tsa.visualize import visualise_stock_data

def main() -> None:
    """
    Entry point for the stock price visualization CLI tool.
    Processes one or more CSV files to plot stock data with
    moving averages, volatility, and linear regression trends.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description="Visualize stock price time series data with moving average, volatility, and linear regression trend."
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="+",
        help="One or more CSV files containing stock data."
    )
    parser.add_argument(
        "-w", "--window",
        type=int,
        default=30,
        help="Window size for moving average and volatility calculations (default: 30)"
    )
    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="Save plots as PNG files instead of displaying."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots",
        help="Directory to save plots (default: plots)"
    )
    parser.add_argument(
        '--version',
        action='version',
        version='stock-price-tsa 1.0'
    )

    args = parser.parse_args()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    for stock_file in args.files:
        if not os.path.isfile(stock_file):
            logging.error(f"File not found: {stock_file}")
            continue

        logging.info(f"Processing file: {stock_file}")
        try:
            save_path = None
            if args.save:
                base_name = os.path.basename(stock_file).replace('.csv', '')
                save_path = os.path.join(args.output_dir, f"{base_name}_plot.png")
            visualise_stock_data(stock_file, window_size=args.window, save_path=save_path)
        except Exception as e:
            logging.error(f"Error processing {stock_file}: {e}")

if __name__ == "__main__":
    main()
