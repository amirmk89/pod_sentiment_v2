#!/usr/bin/env python3
"""
Script to update MSFT records with "No sentiment" in their metadata to have sentiment='No sentiment' instead of 'Hold'.
Uses SQLAlchemy ORM to interact with the database.
"""

import json
import logging
import sys
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import models from storage module
from llm_classifier import LLMClassifier
from storage import EntitySentiment, Base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Path to the database - can be passed as a command line argument
DB_PATH = "podcast_sentiment.db"

def get_msft_records_with_no_sentiment_in_metadata(session) -> List[EntitySentiment]:
    """Get MSFT records that have 'No sentiment' in their metadata but different sentiment value."""
    # Query for all MSFT episode-level records
    records = session.query(EntitySentiment).filter(
        EntitySentiment.company_ticker == 'MSFT',
        EntitySentiment.sentiment == 'Hold',
        EntitySentiment.is_episode_level == True
    ).all()
    
    records_to_update = []
    
    for record in records:
        # Parse the metadata JSON
        try:
            if record.meta_data:
                meta_data = json.loads(record.meta_data)
                raw_response = meta_data.get("raw_response", "")
                if raw_response:
                   sentiment, references = LLMClassifier._parse_detailed_result(raw_response)
                   if sentiment == "No sentiment" and record.sentiment != "No sentiment":
                        records_to_update.append(record)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse metadata for record {record.id}")
            continue
    
    return records_to_update

def update_sentiment_to_no_sentiment(session, records: List[EntitySentiment]) -> int:
    """Update the sentiment to 'No sentiment' for the given records."""
    if not records:
        return 0
    
    count = 0
    for record in records:
        record.sentiment = "No sentiment"
        count += 1
    
    session.commit()
    return count

def main():
    """Main function to run the script."""
    db_path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    
    # Create SQLite URL
    db_url = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database: {db_url}")
    
    try:
        # Create engine and session
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get records to update
        records_to_update = get_msft_records_with_no_sentiment_in_metadata(session)
        logger.info(f"Found {len(records_to_update)} MSFT records with 'No sentiment' in metadata")
        
        # Print records that will be updated for review
        if records_to_update:
            logger.info("Records that will be updated:")
            for i, record in enumerate(records_to_update[:5], 1):
                logger.info(f"{i}. ID: {record.id}, Current sentiment: {record.sentiment}")
            
            if len(records_to_update) > 5:
                logger.info(f"... and {len(records_to_update) - 5} more records")
                
            # Ask for confirmation before updating
            confirm = input("Do you want to proceed with updating these records? (y/n): ")
            
            if confirm.lower() == "y":
                # Update records
                updated_count = update_sentiment_to_no_sentiment(session, records_to_update)
                logger.info(f"Updated {updated_count} records to have sentiment='No sentiment'")
            else:
                logger.info("Update canceled")
        else:
            logger.info("No records need to be updated")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        if 'session' in locals():
            session.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 