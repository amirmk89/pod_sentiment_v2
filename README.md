# Podcast Sentiment Analyzer

A Python system that queries the Podscan.fm API for company mentions in podcast transcripts, processes text via an LLM, and classifies sentiment into Buy/Outperform, Hold, or Sell/Underperform categories.

## Features

- **API Client**: Search podcast transcripts for company mentions with rate limiting
- **Data Processor**: Extract and validate relevant text
- **LLM Classifier**: Classify sentiment at episode and snippet level using cost-effective LLMs
- **Storage**: Store results in SQLite database
- **Orchestrator**: Coordinate the entire workflow
- **Ticker Data**: Fetch historical weekly stock prices for companies using yfinance

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   PODSCAN_API_KEY=your_podscan_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run the main script:
   ```
   python main.py
   ```

## Configuration

Edit the `config.py` file to customize:
- Companies to track
- Rate limits
- LLM model selection
- Database settings

## Usage

By default, the system performs episode-level sentiment analysis, classifying the sentiment of entire episodes towards companies.

```bash
# Process all companies with episode-level sentiment analysis
python main.py

# Process a specific company
python main.py --company "Apple"

# Also perform snippet-level sentiment analysis
python main.py --snippet-level

# Process episodes that mention multiple companies
python main.py --multi-company

# Limit the number of episodes processed
python main.py --max-episodes 10

# Fetch historical weekly stock prices
python main.py --fetch-ticker-data

# Fetch historical stock prices for specific tickers
python main.py --fetch-ticker-data --tickers AAPL,MSFT,TSLA

# Specify number of weeks of historical data
python main.py --fetch-ticker-data --weeks 8

# Specify output file for ticker data
python main.py --fetch-ticker-data --ticker-output "stock_prices.csv"
```

### Standalone Ticker Data Script

For convenience, you can also use the standalone script to fetch ticker data:

```bash
# Fetch data for all tickers in config.py
python fetch_ticker_data.py

# Fetch data for specific tickers
python fetch_ticker_data.py --tickers AAPL,MSFT,TSLA

# Specify number of weeks of historical data
python fetch_ticker_data.py --weeks 8

# Specify output file
python fetch_ticker_data.py --output "stock_prices.csv"
```

## Output

Results are stored in an SQLite database with the following information:
- Company name/ticker
- Podcast and episode details
- Sentiment classification (Buy/Hold/Sell)
- Episode-level or snippet-level flag
- Snippet text (for snippet-level analysis)
- Timestamp and metadata

Results are exported to CSV files in the specified output directory (default: ./results). 

Ticker data is saved to CSV files (default: ticker_prices.csv) with weekly closing prices for each ticker. 