import argparse
import os
import logging
import json
from typing import List
from stock_price_tsa.visualize import visualise_stock_data, generate_summary_report

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_single_file(
    stock_file: str, 
    args: argparse.Namespace
) -> bool:
    """
    Process a single stock file.
    
    Args:
        stock_file (str): Path to stock file
        args (argparse.Namespace): CLI arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.isfile(stock_file):
        logging.error(f"File not found: {stock_file}")
        return False

    logging.info(f"Processing file: {stock_file}")
    
    try:
        # Generate report if requested
        if args.report:
            report = generate_summary_report(stock_file, args.window)
            
            if args.save:
                base_name = os.path.basename(stock_file).replace('.csv', '')
                report_path = os.path.join(args.output_dir, f"{base_name}_report.json")
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logging.info(f"Report saved to: {report_path}")
            else:
                print(f"\n=== Analysis Report for {report['symbol']} ===")
                print(f"Period: {report['period']['start']} to {report['period']['end']} ({report['period']['days']} days)")
                print(f"Price Range: ${report['price_stats']['min']:.2f} - ${report['price_stats']['max']:.2f}")
                print(f"Current Price: ${report['price_stats']['current']:.2f}")
                print(f"Total Return: {report['price_stats']['change_pct']:.2f}%")
                print(f"Average Volatility: ${report['volatility_stats']['avg_volatility']:.2f}")
                print(f"Overall Trend: {report['trend_analysis']['overall_trend']}")
        
        # Generate visualization
        if not args.report_only:
            save_path = None
            if args.save:
                base_name = os.path.basename(stock_file).replace('.csv', '')
                save_path = os.path.join(args.output_dir, f"{base_name}_plot.png")
            
            visualise_stock_data(
                stock_file, 
                window_size=args.window, 
                save_path=save_path,
                include_volume=args.volume,
                include_bollinger=args.bollinger,
                include_rsi=args.rsi
            )
            
        return True
        
    except Exception as e:
        logging.error(f"Error processing {stock_file}: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        return False

def validate_files(files: List[str]) -> List[str]:
    """Validate and filter existing files."""
    valid_files = []
    for file in files:
        if os.path.isfile(file):
            valid_files.append(file)
        else:
            logging.warning(f"Skipping non-existent file: {file}")
    return valid_files

def main() -> None:
    """
    Enhanced entry point for the stock price visualization CLI tool.
    """
    parser = argparse.ArgumentParser(
        description="Advanced stock price time series analysis with moving averages, volatility, technical indicators, and trend analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/AAPL_2020-01-01_to_2023-01-01.csv
  %(prog)s data/*.csv -w 50 --save --bollinger --rsi
  %(prog)s data/AAPL.csv --report --volume
  %(prog)s data/TSLA.csv -w 20 --save -o results --verbose
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="+",
        help="One or more CSV files containing stock data."
    )
    
    # Analysis parameters
    parser.add_argument(
        "-w", "--window",
        type=int,
        default=30,
        help="Window size for moving average and volatility calculations (default: 30)"
    )
    
    # Output options
    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="Save plots and reports as files instead of displaying."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to save plots and reports (default: output)"
    )
    
    # Visualization options
    parser.add_argument(
        "--volume",
        action="store_true",
        help="Include volume analysis in the plot."
    )
    parser.add_argument(
        "--bollinger",
        action="store_true",
        help="Include Bollinger Bands in the price plot."
    )
    parser.add_argument(
        "--rsi",
        action="store_true",
        help="Include RSI (Relative Strength Index) subplot."
    )
    
    # Report options
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate summary statistics report."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate only the report, skip visualization."
    )
    
    # Utility options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    parser.add_argument(
        '--version',
        action='version',
        version='stock-price-tsa 2.0'
    )

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate window size
    if args.window < 1:
        logging.error("Window size must be at least 1")
        return
    
    # Create output directory if saving
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Output directory: {args.output_dir}")
    
    # Validate files
    valid_files = validate_files(args.files)
    if not valid_files:
        logging.error("No valid files found to process")
        return
    
    logging.info(f"Processing {len(valid_files)} file(s) with window size {args.window}")
    
    # Process files
    successful = 0
    failed = 0
    
    for stock_file in valid_files:
        if process_single_file(stock_file, args):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logging.info(f"Processing complete: {successful} successful, {failed} failed")
    
    if failed > 0:
        exit(1)

if __name__ == "__main__":
    main()