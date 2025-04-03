#!/usr/bin/env python3
"""
Export podcast data with episodes, transcripts, and ticker sentiments to a CSV file.
"""

import os
import pandas as pd
import argparse
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define sentiment categories
SENTIMENT_CATEGORIES = ['Buy', 'Hold', 'Sell']

from podscan_client.storage import DatabaseManager, Podcast, Episode, EntitySentiment

def analyze_sentiment_price_relationship(ticker_df, price_data):
    """
    Analyze the relationship between sentiment data and price movements.
    
    Args:
        ticker_df: DataFrame containing sentiment data for a ticker
        price_data: DataFrame containing price data for the ticker
        
    Returns:
        Dictionary with analysis metrics and enhanced weekly data
    """
    # Create a copy of the weekly counts
    weekly_counts = pd.crosstab(ticker_df['week'], ticker_df['sentiment'])
    weekly_counts.sort_index(inplace=True)
    
    # Fill missing sentiment columns
    for col in SENTIMENT_CATEGORIES:
        if col not in weekly_counts.columns:
            weekly_counts[col] = 0
    
    # Calculate totals and percentages
    weekly_counts['total'] = weekly_counts.sum(axis=1)
    for col in SENTIMENT_CATEGORIES:
        weekly_counts[f'{col}_pct'] = (weekly_counts[col] / weekly_counts['total'] * 100).round(1)
    
    # Merge with price data
    # Ensure both date columns are datetime objects for merge_asof
    weekly_df = weekly_counts.reset_index().sort_values('week')
    weekly_df['week'] = pd.to_datetime(weekly_df['week'])
    
    price_df = price_data[['date', 'open', 'weekly_return']].sort_values('date')
    price_df['date'] = pd.to_datetime(price_df['date'])
    
    # Debug info
    print(f"Weekly data: {len(weekly_df)} rows, week dtype: {weekly_df['week'].dtype}")
    print(f"Price data: {len(price_df)} rows, date dtype: {price_df['date'].dtype}")
    if not price_df.empty:
        print(f"Price data date range: {price_df['date'].min()} to {price_df['date'].max()}")
    if not weekly_df.empty:
        print(f"Weekly data date range: {weekly_df['week'].min()} to {weekly_df['week'].max()}")
    
    # If price_df is empty, create a placeholder with the same dates as weekly_df
    if price_df.empty and not weekly_df.empty:
        print("Warning: Empty price data. Creating placeholder data to avoid empty merge.")
        price_df = pd.DataFrame({
            'date': weekly_df['week'].unique(),
            'open': 100.0,
            'close': 100.0,
            'weekly_return': 0.0
        })
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.sort_values('date')
        
    # Use tolerance parameter to allow for small date differences
    merged_data = pd.merge_asof(
        weekly_df,
        price_df,
        left_on='week',
        right_on='date',
        direction='nearest',
        tolerance=pd.Timedelta('7 days')
    ).set_index('week')
    
    print(f"Merged data: {len(merged_data)} rows")
    
    # Calculate weighted sentiment score
    merged_data['sentiment_score'] = (
        merged_data['Buy'] * 1 + 
        merged_data['Hold'] * 0 + 
        merged_data['Sell'] * -1
    ) / merged_data['total']
    
    # Calculate normalized mention volume (for trend analysis)
    merged_data['normalized_volume'] = merged_data['total'] / merged_data['total'].mean()
    
    # Add future return columns for predictive analysis
    merged_data['next_week_return'] = merged_data['weekly_return'].shift(-1)
    merged_data['two_week_return'] = merged_data['weekly_return'].shift(-1) + merged_data['weekly_return'].shift(-2)
    
    # Calculate various correlation metrics
    correlations = {
        'same_week': merged_data['sentiment_score'].corr(merged_data['weekly_return']),
        'next_week': merged_data['sentiment_score'].corr(merged_data['next_week_return']),
        'two_week': merged_data['sentiment_score'].corr(merged_data['two_week_return']),
        'volume_price': merged_data['total'].corr(merged_data['weekly_return']),
        'volume_volatility': merged_data['total'].corr(merged_data['weekly_return'].abs())
    }
    
    # Calculate predictive accuracy
    merged_data['sentiment_direction'] = merged_data['sentiment_score'] > 0
    merged_data['price_up_next_week'] = merged_data['next_week_return'] > 0
    correct_predictions = (merged_data['sentiment_direction'] == merged_data['price_up_next_week']).sum()
    total_predictions = merged_data['sentiment_direction'].count() - 1  # Exclude the last week
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return {
        'weekly_data': merged_data,
        'correlations': correlations,
        'predictive_accuracy': accuracy
    }

def analyze_ticker_timeline(df, ticker=None, include_price_data=False):
    """
    Analyze sentiment for a specific ticker (or all tickers) across time periods.
    
    Args:
        df: DataFrame containing the sentiment data
        ticker: Optional ticker symbol to filter by
        include_price_data: Whether to include historical price data
    
    Returns:
        Dictionary of DataFrames with daily, weekly, and monthly analyses
    """
    # Filter by ticker if provided
    if ticker:
        ticker_df = df[df['company_ticker'] == ticker.upper()].copy()
        if ticker_df.empty:
            print(f"No data found for ticker {ticker}")
            return None
    else:
        ticker_df = df.copy()
    
    # Ensure date column is datetime
    ticker_df['episode_date'] = pd.to_datetime(ticker_df['episode_date'])
    
    # Convert sentiment to categories
    # def categorize_sentiment(value):
    #     if value < -0.3:
    #         return 'sell'
    #     elif value > 0.3:
    #         return 'buy'
    #     else:
    #         return 'hold'
    
    # ticker_df['sentiment_category'] = ticker_df['sentiment'].apply(categorize_sentiment)
    
    # Count total mentions by category
    mention_counts = ticker_df['sentiment'].value_counts()
    total_mentions = len(ticker_df)
    
    # Calculate percentages
    mention_percentages = (mention_counts / total_mentions * 100).round(1)
    
    # Combine into a summary DataFrame
    mention_summary = pd.DataFrame({
        'count': mention_counts,
        'percentage': mention_percentages
    })
    
    # Create date periods
    ticker_df['day'] = ticker_df['episode_date'].dt.date
    ticker_df['week'] = ticker_df['episode_date'].dt.to_period('W-MON').apply(lambda r: r.end_time.date())
    ticker_df['month'] = ticker_df['episode_date'].dt.to_period('M').apply(lambda r: r.end_time.date())
    
    # Group by periods and count sentiment categories
    daily_counts = pd.crosstab(ticker_df['day'], ticker_df['sentiment'])
    weekly_counts = pd.crosstab(ticker_df['week'], ticker_df['sentiment'])
    monthly_counts = pd.crosstab(ticker_df['month'], ticker_df['sentiment'])
    
    # Sort by date
    daily_counts.sort_index(inplace=True)
    weekly_counts.sort_index(inplace=True)
    monthly_counts.sort_index(inplace=True)
    
    # Fill missing values with 0 and calculate percentages
    for df_counts in [daily_counts, weekly_counts, monthly_counts]:
        for col in SENTIMENT_CATEGORIES:
            if col not in df_counts.columns:
                df_counts[col] = 0
        
        df_counts['total'] = df_counts.sum(axis=1)
        
        # Add percentage columns
        for col in SENTIMENT_CATEGORIES:
            df_counts[f'{col}_pct'] = (df_counts[col] / df_counts['total'] * 100).round(1)
    
    # Add price data if requested
    price_analysis = None
    if include_price_data and ticker:
        # Get historical price data
        price_data = get_ticker_price_data(ticker, min(ticker_df['episode_date']), max(ticker_df['episode_date']))
        
        # Perform sentiment-price relationship analysis
        if not price_data.empty:
            price_analysis = analyze_sentiment_price_relationship(ticker_df, price_data)
        
            # Replace weekly counts with enhanced data
            weekly_counts = price_analysis['weekly_data']
    
    result = {
        'summary': mention_summary,
        'daily': daily_counts,
        'weekly': weekly_counts,
        'monthly': monthly_counts
    }
    
    # Add price analysis if available
    if price_analysis:
        result['correlations'] = price_analysis['correlations']
        result['predictive_accuracy'] = price_analysis['predictive_accuracy']
    
    return result

def get_ticker_price_data(ticker, start_date, end_date):
    """
    Retrieve historical price data for a ticker from a CSV file.
    
    Args:
        ticker: The ticker symbol
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        DataFrame with price data
    """
    try:
        # Load data from CSV file instead of using yfinance
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../ticker_prices.csv')
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Warning: ticker_prices.csv not found at {csv_path}")
            return pd.DataFrame(columns=['date', 'open', 'close', 'weekly_return'])
        
        # Read the CSV file
        all_price_data = pd.read_csv(csv_path)
        # Drop rows where the first column isn't a valid date after 2020
        all_price_data = all_price_data[pd.to_datetime(all_price_data.iloc[:, 0], errors='coerce') > '2020-01-01']
        all_price_data.rename(columns={all_price_data.columns[0]: 'Date'}, inplace=True)
        
        # Convert date column to datetime
        all_price_data['Date'] = pd.to_datetime(all_price_data['Date'])
        
        # Extract columns for this ticker (format is "ticker_name column_type")
        ticker_lower = ticker.lower()
        cols = [c for c in all_price_data.columns if ticker_lower in c.lower()]
        
        # Create DataFrame with just the ticker data
        cols_to_use = ['Date'] + cols
        ticker_data = all_price_data[cols_to_use].copy()
        
        # Rename columns to expected format
        rename_map = {}
        for col in cols:
            if 'open' in col.lower():
                rename_map[col] = 'open'
            elif 'close' in col.lower():
                rename_map[col] = 'close'
            elif 'return' in col.lower():
                rename_map[col] = 'weekly_return'
        
        # Filter by date range
        ticker_data = ticker_data[
            (ticker_data['Date'] >= pd.Timestamp(start_date)) & 
            (ticker_data['Date'] <= pd.Timestamp(end_date))
        ]
        
        if ticker_data.empty:
            print(f"No price data found for {ticker} in the specified date range")
            return pd.DataFrame(columns=['date', 'open', 'close', 'weekly_return'])
        
        # Rename and return
        rename_map['Date'] = 'date'
        return ticker_data.rename(columns=rename_map)
    
    except Exception as e:
        print(f"Error retrieving price data for {ticker}: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['date', 'open', 'close', 'weekly_return'])

def visualize_sentiment_vs_price(ticker_analysis, ticker, output_path=None):
    """
    Create visualization of sentiment vs price for a ticker.
    
    Args:
        ticker_analysis: Dictionary containing analysis results
        ticker: Ticker symbol
        output_path: Optional path to save visualization
        
    Returns:
        Path to saved visualization if output_path provided
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    weekly_data = ticker_analysis['weekly']
    
    # Create a figure with 2 subplots (price & sentiment on top, volume on bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot price line on top subplot
    ax1.plot(weekly_data.index, weekly_data['close'], 'b-', label='Stock Price', linewidth=2)
    ax1.set_ylabel('Price ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for sentiment score
    ax3 = ax1.twinx()
    ax3.plot(weekly_data.index, weekly_data['sentiment_score'], 'g-', label='Sentiment Score', linewidth=2)
    ax3.set_ylabel('Sentiment Score', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot volume in bottom subplot
    # Use color based on sentiment direction
    colors = weekly_data.apply(
        lambda row: 'green' if row['sentiment_score'] > 0.2 else 
                   'red' if row['sentiment_score'] < -0.2 else 'gray',
        axis=1
    )
    
    ax2.bar(weekly_data.index, weekly_data['total'], color=colors, alpha=0.7, label='Mention Volume')
    ax2.set_ylabel('Mention Count')
    ax2.set_xlabel('Date')
    
    # Add correlation and accuracy info in title
    accuracy = ticker_analysis.get('predictive_accuracy', 0) * 100
    correlations = ticker_analysis.get('correlations', {})
    
    title_text = f"{ticker} Price vs. Podcast Sentiment\n"
    title_text += f"Next Week Correlation: {correlations.get('next_week', 0):.2f}, "
    title_text += f"Directional Accuracy: {accuracy:.1f}%"
    
    fig.suptitle(title_text, fontsize=14)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines3, labels1 + labels3, loc='upper left')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        return output_path
    else:
        plt.show()

def query_podcast_data(session):
    """Fetch podcast data with sentiments from the database."""
    query = session.query(
        Podcast.id.label('podcast_id'),
        Podcast.name.label('podcast_name'),
        Episode.id.label('episode_id'),
        Episode.title.label('episode_title'),
        Episode.date.label('episode_date'),
        Episode.transcript,
        EntitySentiment.company_name,
        EntitySentiment.company_ticker,
        EntitySentiment.sentiment,
        EntitySentiment.confidence,
        EntitySentiment.snippet_text,
        EntitySentiment.is_episode_level == True
    ).join(
        Episode, Podcast.id == Episode.podcast_id
    ).join(
        EntitySentiment, Episode.id == EntitySentiment.episode_id
    ).order_by(
        Podcast.name, Episode.date, EntitySentiment.company_ticker
    )
    
    return query.all()

def convert_to_dataframe(results):
    """Convert query results to a pandas DataFrame."""
    df = pd.DataFrame(results, columns=[
        'podcast_id', 'podcast_name', 'episode_id', 'episode_title', 
        'episode_date', 'transcript', 'company_name', 'company_ticker',
        'sentiment', 'confidence', 'snippet_text', 'is_episode_level'
    ])
    # Add dashboard link column
    df['dashboard_link'] = df.apply(
        lambda x: f"https://podscan.fm/dashboard/podcasts/{x['podcast_id']}/episode/{x['episode_id']}", 
        axis=1
    )
    return df

def create_podcast_summary(df, output_dir, base_name):
    """Create podcast summary report."""
    podcast_summary = df.groupby(['podcast_id', 'podcast_name']).agg(
        episode_count=('episode_id', 'nunique'),
        sentiment_count=('sentiment', 'count'),
    ).reset_index()
    podcast_file = os.path.join(output_dir, f"{base_name}_podcast_summary.xlsx")
    # podcast_summary.to_excel(podcast_file, index=False)
    print(f"Saved podcast summary to {podcast_file}")
    return podcast_summary

def create_episode_summary(df, output_dir, base_name):
    """Create episode summary report."""
    # First, get unique episodes
    episode_base = df[['podcast_id', 'podcast_name', 'episode_id', 'episode_title', 
                      'episode_date', 'dashboard_link']].drop_duplicates()
    
    # Then count tickers per episode
    ticker_counts = df.groupby('episode_id')['company_ticker'].nunique().reset_index()
    ticker_counts.columns = ['episode_id', 'ticker_count']
    
    # Get unique tickers per episode as a list
    ticker_list = df.groupby('episode_id')['company_ticker'].apply(
        lambda x: ', '.join(sorted(set(x)))
    ).reset_index()
    ticker_list.columns = ['episode_id', 'tickers']
    
    # Merge everything
    episode_summary = episode_base.merge(ticker_counts, on='episode_id', how='left')
    episode_summary = episode_summary.merge(ticker_list, on='episode_id', how='left')
    
    episode_file = os.path.join(output_dir, f"{base_name}_episode_summary.xlsx")
    # episode_summary.to_excel(episode_file, index=False)
    print(f"Saved episode summary to {episode_file}")
    return episode_summary

def print_data_summary(df):
    """Print summary information about the dataset."""
    print(f"Total rows: {len(df)}")
    print(f"Unique podcasts: {df['podcast_id'].nunique()}")
    print(f"Unique episodes: {df['episode_id'].nunique()}")
    print(f"Unique tickers: {df['company_ticker'].nunique()}")
    
    print("\n--- Sentiment Analysis ---")
    
    # 1. Sentiments per ticker
    ticker_counts = df.groupby('company_ticker').size().sort_values(ascending=False)
    print("\nSentiments per ticker:")
    print(ticker_counts.to_string())
    
    # 2. Sentiments per episode
    episode_counts = df.groupby(['episode_id', 'episode_title']).size().sort_values(ascending=False)
    print("\nSentiments per episode:")
    print("Top 3:")
    print(episode_counts.head(5).to_string())
    print("Bottom 3:")
    print(episode_counts.tail(5).to_string())
    
    # 3. Sentiments per podcast
    podcast_counts = df.groupby(['podcast_id', 'podcast_name']).size().sort_values(ascending=False)
    print("\nSentiments per podcast:")
    print("Top 3:")
    print(podcast_counts.head(10).to_string())
    print("Bottom 3:")
    print(podcast_counts.tail(3).to_string())
    # Count podcasts with less than 5 sentiments
    small_podcasts = podcast_counts[podcast_counts < 5]
    print(f"\nPodcasts with <5 sentiments: {len(small_podcasts)}")

def perform_ticker_analysis(df, analyze_ticker, output_dir, base_name):
    """Perform detailed analysis for a specific ticker."""
    if not analyze_ticker:
        return None
        
    print(f"\n--- Temporal Analysis for {analyze_ticker} ---")
    timeline_analysis = analyze_ticker_timeline(df, analyze_ticker, include_price_data=True)
    
    if not timeline_analysis:
        return None
        
    # Save the analysis to Excel
    ticker_file = os.path.join(output_dir, f"{base_name}_{analyze_ticker}_timeline.xlsx")
    
    with pd.ExcelWriter(ticker_file) as writer:
        timeline_analysis['summary'].to_excel(writer, sheet_name='Overall Summary')
        timeline_analysis['daily'].to_excel(writer, sheet_name='Daily')
        timeline_analysis['weekly'].to_excel(writer, sheet_name='Weekly')
        timeline_analysis['monthly'].to_excel(writer, sheet_name='Monthly')
    
    print(f"Saved ticker timeline analysis to {ticker_file}")
    
    # Print summary information
    print_ticker_summary(timeline_analysis, analyze_ticker)
    
    # Generate visualization
    viz_file = os.path.join(output_dir, f"{base_name}_{analyze_ticker}_analysis.png")
    #visualize_sentiment_vs_price(timeline_analysis, analyze_ticker, viz_file)
    print(f"Saved visualization to {viz_file}")
    
    return timeline_analysis

def print_ticker_summary(timeline_analysis, ticker):
    """Print summary information for a specific ticker."""
    # Print the summary of all mentions
    summary = timeline_analysis['summary']
    print(f"\nOverall mention counts for {ticker}:")
    for category in ['buy', 'hold', 'sell']:
        if category in summary.index:
            count = summary.loc[category, 'count']
            pct = summary.loc[category, 'percentage']
            print(f"{category.capitalize()}: {count} ({pct}%)")
    
    total = summary['count'].sum()
    print(f"Total mentions: {total}")
    
    # Print the most recent month's data
    recent_month = timeline_analysis['monthly'].iloc[-1]
    print(f"\nMost recent month ({timeline_analysis['monthly'].index[-1]}):")
    print(f"Buy: {recent_month.get('buy', 0)} ({recent_month.get('buy_pct', 0)}%)")
    print(f"Hold: {recent_month.get('hold', 0)} ({recent_month.get('hold_pct', 0)}%)")
    print(f"Sell: {recent_month.get('sell', 0)} ({recent_month.get('sell_pct', 0)}%)")
    print(f"Total mentions: {recent_month.get('total', 0)}")
    
    # Print correlation results
    if 'correlations' in timeline_analysis:
        print("\nPrice-Sentiment Correlations:")
        print(f"Same week: {timeline_analysis['correlations']['same_week']:.4f}")
        print(f"Next week: {timeline_analysis['correlations']['next_week']:.4f}")

def export_podcast_data(output_path, db_path=None, analyze_ticker=None):
    """
    Export podcast data with sentiments to an Excel file and perform analysis.
    
    Args:
        output_path: Path to save the Excel file
        db_path: Optional custom database path
        analyze_ticker: Optional ticker to analyze over time
    
    Returns:
        Path to the saved Excel file
    """
    # Initialize database manager
    db_manager = DatabaseManager(db_path)
    
    # Create a read-only session
    engine = db_manager.engine
    ReadOnlySession = sessionmaker(
        bind=engine,
        info={"readonly": True}
    )
    session = ReadOnlySession()

    # Create directory and base name for additional files
    output_dir = os.path.dirname(os.path.abspath(output_path))
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    try:
        # Query data and convert to DataFrame
        results = query_podcast_data(session)
        df = convert_to_dataframe(results)
        
       ## Save the main data to Excel
       #df.to_excel(output_path, index=False)
       #print(f"Exported data to {output_path}")
       #
       ## Create summary reports
       #create_podcast_summary(df, output_dir, base_name)
       #create_episode_summary(df, output_dir, base_name)
        
        # Print data summary
        # print_data_summary(df)
        
        # Perform ticker analysis if requested
        if analyze_ticker:
            perform_ticker_analysis(df, analyze_ticker, output_dir, base_name)
        
        return output_path
    
    finally:
        session.close()

def main():
    parser = argparse.ArgumentParser(description="Export podcast data with transcripts and ticker sentiments")
    parser.add_argument("--output", "-o", default=f"podcast_data_export_{datetime.now().strftime('%b%d_%H%M')}.xlsx",
                        help="Output Excel file path")
    parser.add_argument("--db", help="Custom database path")
    parser.add_argument("--ticker", "-t", help="Analyze a specific ticker across time")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    export_podcast_data(args.output, args.db, args.ticker)

if __name__ == "__main__":
    main()