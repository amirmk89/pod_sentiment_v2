#!/usr/bin/env python3
"""
Main script for the Podcast Sentiment Analyzer.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import List

from podscan_client.orchestrator import Orchestrator
from podscan_client.storage import DatabaseManager
from podscan_client.config import COMPANIES
from tests.test_components import MockLLMClassifier  # Import the mock classifier for testing
from podscan_client.ticker_data import get_weekly_prices, save_to_csv  # Import ticker data functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sentiment_analyzer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Podcast Sentiment Analyzer")

    parser.add_argument(
        "--company",
        type=str,
        help="Process a specific company by name (must be in config.py)",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,  # Reduced default for testing
        help="Maximum number of episodes to process per company",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,  # Default is 5 pages
        help="Maximum number of pages to fetch from API (up to 50 episodes per page)",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",  # Default is English
        help="Language code to filter podcasts (e.g., 'en' for English, 'es' for Spanish)",
    )

    parser.add_argument(
        "--export-path",
        type=str,
        default="./results",
        help="Directory to save exported CSV files",
    )

    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock LLM classifier for testing",
    )
    
    parser.add_argument(
        "--snippet-level",
        action="store_true",
        help="Also perform snippet-level classification (default: episode-level only)",
    )
    
    parser.add_argument(
        "--multi-company",
        action="store_true",
        help="Process episodes that mention multiple companies",
    )
    
    parser.add_argument(
        "--override-db",
        action="store_true",
        help="Override existing sentiment in database",
    )

    parser.add_argument(
        "--export-companies",
        type=str,
        help="Comma-separated list of specific companies to export (overrides --company for export)",
    )
    
    # Add ticker data arguments
    parser.add_argument(
        "--fetch-ticker-data",
        action="store_true",
        help="Fetch historical weekly ticker data",
    )
    
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of specific tickers to fetch (defaults to all in config)",
    )
    
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        help="Number of weeks of historical data to fetch",
    )
    
    parser.add_argument(
        "--ticker-output",
        type=str,
        default="ticker_prices.csv",
        help="Path to save the ticker data CSV file",
    )

    # Add cache-only mode argument
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only make API calls and generate cache, without processing sentiment",
    )
    
    # Add before date argument
    parser.add_argument(
        "--before-date",
        type=str,
        help="Only get episodes before this date (format: 'YYYY-MM-DD HH:MM:SS')",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    try:
        # Handle ticker data fetching if requested
        if args.fetch_ticker_data:
            fetch_ticker_data(args)
            # Exit if only fetching ticker data and not processing podcasts
            if not args.company and not args.multi_company:
                return
        
        # Create the orchestrator
        orchestrator = Orchestrator()

        # Use mock classifier for testing if specified
        if args.use_mock: # or True:  # todo - LLM not used for now
            logger.info("Using mock LLM classifier for testing")
            orchestrator.set_llm_classifier(MockLLMClassifier())

        # Process companies
        if args.company:
            # Find the company in the config
            company_config = None
            for company in COMPANIES:
                if company["name"].lower() == args.company.lower():
                    company_config = company
                    break

            if not company_config:
                logger.error(f"Company '{args.company}' not found in configuration")
                sys.exit(1)

            # Process the specific company
            logger.info(f"Processing company: {company_config['name']}")
            
            # Process the company with the specified options
            stats = orchestrator.process_company(
                company_config["name"],
                company_config["ticker"],
                max_episodes=args.max_episodes,
                snippet_level=args.snippet_level,
                override_db=args.override_db,
                max_pages=args.max_pages,
                language=args.language,
                cache_only=args.cache_only,
                before_date=args.before_date
            )
            
            logger.info(f"Processing complete for {company_config['name']}")
            logger.info(f"Stats: {stats}")

        elif args.multi_company:
            # Process episodes with multiple company mentions
            logger.info("Processing episodes with multiple company mentions")
            orchestrator._process_multi_company_episodes(
                max_episodes=args.max_episodes,
                override_db=args.override_db,
                max_pages=args.max_pages,
                language=args.language,
                cache_only=args.cache_only,
                before_date=args.before_date
            )
            logger.info("Multi-company episode processing complete")
            # Export all results since we're processing multiple companies
            if not args.cache_only:
                export_results(args.export_path)
            else:
                logger.info("Cache-only mode: skipping export")
            
        else:
            # Process all companies
            logger.info("Processing all companies")
            all_stats = orchestrator.process_all_companies(
                max_episodes_per_company=args.max_episodes,
                snippet_level=args.snippet_level,
                override_db=args.override_db,
                max_pages=args.max_pages,
                language=args.language,
                cache_only=args.cache_only,
                before_date=args.before_date
            )

            logger.info("Processing complete for all companies")
            for stats in all_stats:
                logger.info(f"Stats for {stats['company']}: {stats}")

        # Skip export if in cache-only mode
        if args.cache_only:
            logger.info("Cache-only mode: skipping export")
            return

        # Export results
        if args.export_companies:
            # Split the comma-separated list and strip whitespace
            companies_to_export = [company.strip() for company in args.export_companies.split(',')]
            export_results(args.export_path, companies=companies_to_export)
        elif args.company:
            export_results(args.export_path, companies=args.company)
        else:
            export_results(args.export_path)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        sys.exit(1)


def export_results(export_path: str, companies=None):
    """Export results to CSV files.

    Args:
        export_path: Directory to save CSV files
        companies: If specified, only export results for these companies
                  Can be a single company name (str) or a list of company names
    """
    try:
        # Create export directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)

        # Initialize database manager
        db_manager = DatabaseManager()

        # Handle specific companies
        if companies:
            # Convert string to single-item list if needed
            if isinstance(companies, str):
                companies = [companies]
                
            if len(companies) == 1:
                logger.info(f"Exporting data only for company: {companies[0]}")
            else:
                logger.info(f"Exporting data for specific companies: {', '.join(companies)}")
            
            for company_name in companies:
                # Find the matching company in config
                company_config = None
                for company in COMPANIES:
                    if company["name"].lower() == company_name.lower():
                        company_config = company
                        break
                        
                if not company_config:
                    logger.warning(f"Company '{company_name}' not found in configuration, skipping")
                    continue
                    
                company_name = company_config["name"]
                
                # Export episode-level results for this company
                company_episode_file = f"{company_name.lower().replace(' ', '_')}_episode_sentiments.csv"
                company_episode_path = os.path.join(export_path, company_episode_file)
                db_manager.export_to_csv(company_episode_path, company_name=company_name, is_episode_level=True)
                logger.info(f"Exported episode-level results for {company_name} to {company_episode_path}")
                
                # Export all results for this company
                company_all_file = f"{company_name.lower().replace(' ', '_')}_all_sentiments.csv"
                company_all_path = os.path.join(export_path, company_all_file)
                db_manager.export_to_csv(company_all_path, company_name=company_name)
                logger.info(f"Exported all results for {company_name} to {company_all_path}")
            
            return
            
        # Export all data if no specific companies
        # Export episode-level results (primary)
        episode_results_path = os.path.join(export_path, "episode_level_sentiments.csv")
        db_manager.export_to_csv(episode_results_path, is_episode_level=True)
        logger.info(f"Exported episode-level results to {episode_results_path}")

        # Export snippet-level results (secondary)
        snippet_results_path = os.path.join(export_path, "snippet_level_sentiments.csv")
        db_manager.export_to_csv(snippet_results_path, is_episode_level=False)
        logger.info(f"Exported snippet-level results to {snippet_results_path}")
        
        # Export all results combined
        all_results_path = os.path.join(export_path, "all_sentiments.csv")
        db_manager.export_to_csv(all_results_path)
        logger.info(f"Exported all results to {all_results_path}")

        # Export results for each company
        for company in COMPANIES:
            company_name = company["name"]
            
            # Export episode-level results for this company
            company_episode_file = f"{company_name.lower().replace(' ', '_')}_episode_sentiments.csv"
            company_episode_path = os.path.join(export_path, company_episode_file)
            db_manager.export_to_csv(company_episode_path, company_name=company_name, is_episode_level=True)
            logger.info(f"Exported episode-level results for {company_name} to {company_episode_path}")
            
            # Export all results for this company
            company_all_file = f"{company_name.lower().replace(' ', '_')}_all_sentiments.csv"
            company_all_path = os.path.join(export_path, company_all_file)
            db_manager.export_to_csv(company_all_path, company_name=company_name)
            logger.info(f"Exported all results for {company_name} to {company_all_path}")

    except Exception as e:
        logger.error(f"Error exporting results: {e}")


def fetch_ticker_data(args):
    """Fetch historical ticker data based on command line arguments."""
    logger.info("Fetching historical ticker data")
    
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
        output_path = args.ticker_output
        df = save_to_csv(ticker_data, output_path=output_path)
        logger.info(f"Saved ticker data to {output_path}")
        
        # Print a summary to the console
        print("\nWeekly Closing Prices Summary:")
        print(df.tail(args.weeks).to_string())
    else:
        logger.warning("No ticker data was retrieved")


if __name__ == "__main__":
    main()
