#!/usr/bin/env python3
"""
Standalone script to fetch historical ticker data.
"""

import sys
from podscan_client.ticker_data import get_ticker_column_name, get_weekly_prices, save_to_csv
from podscan_client.config import COMPANIES
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch historical weekly stock prices")
    
    parser.add_argument(
        "-t", "--tickers",
        type=str,
        help="Comma-separated list of specific tickers to fetch (defaults to all in config)"
    )
    
    parser.add_argument(
        "-w", "--weeks",
        type=int,
        default=4,
        help="Number of weeks of historical data to fetch"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="ticker_prices.csv",
        help="Path to save the output CSV file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Determine which tickers to fetch
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
        logger.info(f"Fetching data for specified tickers: {', '.join(tickers)}")
    else:
        # Use all tickers from config
        tickers = [company['ticker'] for company in COMPANIES]
        logger.info(f"Fetching data for all {len(tickers)} tickers in config")
    
    # Fetch the data
    ticker_data = get_weekly_prices(tickers, weeks=args.weeks)
    
    # Save to CSV
    if ticker_data:
        for ticker, data in ticker_data.items():
            # Keep only the opening prices instead of closing prices
            ticker_data[ticker] = data[[get_ticker_column_name(ticker)]].copy()
            # ticker_data[ticker].rename(columns={'Open': f'{ticker} Open'}, inplace=True)
        
        df = save_to_csv(ticker_data, output_path=args.output)
        # Print a summary of the data
        print("\nWeekly Closing Prices Summary:")
        print(df.tail(args.weeks).to_string())
    

if __name__ == "__main__":
    main() 