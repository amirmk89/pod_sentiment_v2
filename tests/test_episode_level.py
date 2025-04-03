#!/usr/bin/env python3
"""
Test script to verify episode-level sentiment classification.
"""

import os
import sys
import logging
from datetime import datetime

from orchestrator import Orchestrator
from storage import DatabaseManager
from tests.test_components import MockLLMClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_episode_level_classification():
    """Test episode-level sentiment classification."""
    logger.info("Testing episode-level sentiment classification...")
    
    # Initialize database manager with test database
    db_path = "test_episode_level.db"
    db_manager = DatabaseManager(db_path=db_path)
    
    try:
        # Create test episode
        podcast_id = "test_podcast_id"
        podcast_name = "Test Podcast"
        episode_id = "test_episode_id"
        episode_title = "Test Episode"
        episode_description = "Test episode about Apple and Microsoft"
        episode_transcript = """
        Welcome to our podcast. Today we're discussing tech companies.
        
        Apple has been performing exceptionally well this quarter with strong iPhone sales.
        Their services segment showed particularly strong growth, and management provided optimistic guidance.
        
        Microsoft is also doing well with their cloud services, but they face some challenges
        with increased competition in the market.
        """
        
        # Create podcast and episode
        db_manager.get_or_create_podcast(podcast_id, podcast_name)
        db_manager.get_or_create_episode(
            episode_id, 
            podcast_id, 
            episode_title, 
            episode_description,
            episode_transcript,
            datetime.now()
        )
        
        # Initialize orchestrator with mock classifier
        orchestrator = Orchestrator()
        orchestrator.db_manager = db_manager  # Use our test database
        orchestrator.set_llm_classifier(MockLLMClassifier())
        
        # Test companies
        companies = [
            {"name": "Apple", "ticker": "AAPL"},
            {"name": "Microsoft", "ticker": "MSFT"}
        ]
        
        # Process episode for each company
        for company in companies:
            logger.info(f"Processing company: {company['name']}")
            
            # Get episode
            episode = db_manager.get_all_processed_episodes()[0]
            
            # Create company info for classifier
            company_info = [{"name": company["name"], "ticker": company["ticker"]}]
            
            # Classify episode
            episode_sentiments = orchestrator.llm_classifier.classify_episode_for_companies(
                episode["transcript"], company_info
            )
            
            if not episode_sentiments:
                logger.error(f"No sentiments returned for {company['name']}")
                continue
                
            # Prepare episode-level results
            episode_results = []
            for sentiment_result in episode_sentiments:
                # Prepare episode-level result data
                episode_result = {
                    "company_name": sentiment_result["company_name"],
                    "company_ticker": sentiment_result["company_ticker"],
                    "podcast_id": episode["podcast_id"],
                    "podcast_name": episode["podcast_name"],
                    "episode_id": episode["id"],
                    "episode_title": episode["title"],
                    "episode_description": episode["description"],
                    "episode_transcript": episode["transcript"],
                    "episode_date": episode["date"],
                    # No snippet_text for episode-level sentiment
                    "sentiment": sentiment_result["sentiment"],
                    "processed_at": datetime.utcnow(),
                    "metadata": {
                        "is_full_episode": True,
                    },
                }
                
                episode_results.append(episode_result)
            
            # Save episode-level results
            if episode_results:
                db_manager.save_batch_sentiments(episode_results, is_episode_level=True)
                logger.info(f"Saved episode-level sentiment for {company['name']}")
        
        # Verify results
        for company in companies:
            sentiments = db_manager.get_sentiments_by_company(
                company_name=company["name"], 
                is_episode_level=True
            )
            
            if not sentiments:
                logger.error(f"No episode-level sentiments found for {company['name']}")
                continue
                
            logger.info(f"Found {len(sentiments)} episode-level sentiments for {company['name']}")
            for sentiment in sentiments:
                logger.info(f"  Sentiment: {sentiment['sentiment']}")
                logger.info(f"  Is episode level: {sentiment['is_episode_level']}")
                logger.info(f"  Episode: {sentiment['episode_title']}")
        
        logger.info("Episode-level sentiment classification test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing episode-level classification: {e}", exc_info=True)
        return False
        
    finally:
        # Clean up test database
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Removed test database: {db_path}")


if __name__ == "__main__":
    test_episode_level_classification() 