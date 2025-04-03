#!/usr/bin/env python3
"""
Module for fetching historical stock price data for company tickers.
"""

import yfinance as yf
import pandas as pd
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from podscan_client.config import COMPANIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_ticker_column_name(ticker: str, column_name: str = 'Open') -> str:
    """
    Get the column name for the ticker.
    """
    return f'{ticker} {column_name}'


def get_weekly_prices(tickers: List[str], weeks: int = 4) -> Dict[str, pd.DataFrame]:
    """
    Fetch weekly closing prices for the specified tickers.
    
    Args:
        tickers: List of ticker symbols
        weeks: Number of weeks of historical data to fetch
        
    Returns:
        Dictionary mapping ticker symbols to dataframes with price data
    """
    # Calculate the date range starting with last monday
    end_date = datetime.now() - timedelta(days=datetime.now().weekday())  # Monday
    # Go back enough days to ensure we get the specified number of weeks
    start_date = end_date - timedelta(days=weeks * 7 + 5)  # Add buffer days to ensure we get full weeks
    
    ticker_data = {}
    
    for ticker in tickers:
        try:
            # Fetch the historical data
            data = yf.download(
                ticker, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1wk',  # Weekly data
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data returned for ticker {ticker}")
                continue
                
            # Keep the opening prices
            ticker_data[ticker] = data[['Open']].copy()
            ticker_data[ticker].rename(columns={'Open': get_ticker_column_name(ticker)}, inplace=True)
            
            # Calculate weekly returns
            price_col = get_ticker_column_name(ticker)
            return_col = get_ticker_column_name(ticker, 'weekly_return')
            ticker_data[ticker][return_col] = ticker_data[ticker][price_col].pct_change() * 100
            
            logger.info(f"Successfully fetched data for {ticker} with {len(ticker_data[ticker])} weeks")
            
        except Exception as e:
            logger.error(f"Error fetching data for ticker {ticker}: {str(e)}")
    
    return ticker_data


def save_to_csv(ticker_data: Dict[str, pd.DataFrame], output_path: str = "ticker_prices.csv"):
    """
    Save the ticker data to a CSV file.
    
    Args:
        ticker_data: Dictionary of ticker dataframes
        output_path: Path to save the CSV file
    """
    if not ticker_data:
        logger.warning("No ticker data to save")
        return
        
    # Combine all dataframes
    combined = pd.concat(ticker_data.values(), axis=1)
    
    # Save to CSV
    combined.to_csv(output_path)
    logger.info(f"Saved ticker data to {output_path}")
    
    return combined


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
        df = save_to_csv(ticker_data, output_path=args.output)
        # Print a summary of the data
        print("\nWeekly Ticker Data Summary:")
        print(df.tail(args.weeks).to_string())
    

if __name__ == "__main__":
    main() 