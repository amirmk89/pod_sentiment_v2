"""
Orchestrator for coordinating the podcast sentiment analysis workflow.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from tqdm import tqdm

from api_client import PodcastAPIClient
from data_processor import DataProcessor
from storage import DatabaseManager
from config import COMPANIES
from tests.test_components import MockLLMClassifier

# Import the LLM classifier with error handling
try:
    from llm_classifier import LLMClassifier

    llm_import_error = None
except Exception as e:
    llm_import_error = str(e)
    LLMClassifier = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PODCAST_CATEGORIES = [
    "business",
    "technology",
    "entrepreneurship",
    "investing",
    "politics",
    "news",
    "management",
    "government",
    "cryptocurrency",
    "artificial-intelligence",
    "machine-learning",
    "science",
    "startups",
    "fintech",
]

PODCAST_LANGUAGE = "en"


class Orchestrator:
    """Orchestrator for coordinating the podcast sentiment analysis workflow."""

    def __init__(self):
        """Initialize the orchestrator with all required components."""
        self.api_client = PodcastAPIClient()
        self.data_processor = DataProcessor()
        self.db_manager = DatabaseManager()

        # Initialize LLM classifier with error handling
        if LLMClassifier is not None:
            try:
                self.llm_classifier = LLMClassifier()
            except Exception as e:
                logger.warning(f"Failed to initialize LLM classifier: {e}")
                self.llm_classifier = None
                logger.warning(
                    "LLM classifier will need to be set manually before processing"
                )
        else:
            logger.warning(f"Failed to import LLM classifier: {llm_import_error}")
            self.llm_classifier = None
            logger.warning(
                "LLM classifier will need to be set manually before processing"
            )

        logger.info("Orchestrator initialized with all components")

    def set_llm_classifier(self, classifier):
        """Set the LLM classifier manually.

        Args:
            classifier: An instance of a classifier with classify_sentiment and batch_classify methods
        """
        self.llm_classifier = classifier
        logger.info("LLM classifier set manually")

    def process_company(
        self, company_name: str, company_ticker: str, max_episodes: int = None, 
        snippet_level: bool = False, override_db: bool = False, search_for_ticker: bool = False,
        max_pages: int = 5, language: Optional[str] = PODCAST_LANGUAGE,
        cache_only: bool = False, before_date: Optional[str] = None
    ) -> Dict:
        """Process a single company.

        Args:
            company_name: Company name
            company_ticker: Company ticker symbol
            max_episodes: Maximum number of episodes to process
            snippet_level: Whether to also perform snippet-level classification (default: False)
            override_db: Whether to override existing sentiment in database (default: False)
            search_for_ticker: Whether to search for episodes mentioning the ticker (default: False - Generates a lot of FPs)
            max_pages: Maximum number of pages to fetch from API (default: 5)
            language: Language code to filter podcasts (default: "en")
            cache_only: Whether to only make API calls and cache results without processing (default: False)
            before_date: Only get episodes before this date (format: 'YYYY-MM-DD HH:MM:SS')
        Returns:
            Summary of processing results
        """
        # Check if LLM classifier is available
        if not cache_only and self.llm_classifier is None:
            raise ValueError(
                "LLM classifier is not initialized. Set it manually before processing."
            )

        logger.info(f"Processing company: {company_name} ({company_ticker})")

        # Track statistics
        stats = {
            "company": company_name,
            "ticker": company_ticker,
            "episodes_processed": 0,
            "episode_level_classified": 0,
            "episode_sentiment_counts": {"Buy": 0, "Hold": 0, "Sell": 0, "No sentiment": 0},
            "start_time": datetime.now(),
            "end_time": None,
        }
        
        # Add snippet stats if needed
        if snippet_level:
            stats.update({
                "snippets_found": 0,
                "snippets_classified": 0,
                "snippet_sentiment_counts": {"Buy": 0, "Hold": 0, "Sell": 0, "No sentiment": 0},
            })

        categories_info = PODCAST_CATEGORIES if PODCAST_CATEGORIES else "None"
        logger.info(f"Searching for episodes mentioning '{company_name}', filtering: language '{language}', before date '{before_date}', and categories {categories_info}")
        name_results = self.api_client.search_all_pages(
            company_name, max_pages=max_pages, category_names=PODCAST_CATEGORIES,
            language=language, before_date=before_date
        )

        if search_for_ticker:
            # Search for episodes mentioning the ticker
            logger.info(f"Searching for episodes mentioning '{company_ticker}', filtering: language '{language}' and categories {categories_info}")
            ticker_results = self.api_client.search_all_pages(
                company_ticker, max_pages=max_pages, category_names=PODCAST_CATEGORIES,
                language=language, before_date=before_date
            )
        else:
            ticker_results = []

        # Combine and deduplicate results
        all_episodes = name_results + ticker_results
        unique_episodes = self._deduplicate_episodes(all_episodes)
        unique_episodes = [ep for ep in unique_episodes if ep.get('episode_fully_processed')]

        # Limit to max_episodes
        episodes_to_process = (
            unique_episodes[:max_episodes] if max_episodes else unique_episodes
        )

        logger.info(
            f"Found {len(unique_episodes)} unique episodes mentioning {company_name} or {company_ticker}"
        )
        logger.info(f"Processing {len(episodes_to_process)} episodes")
        
        # If cache_only mode, skip processing and return stats
        if cache_only:
            stats["episodes_processed"] = len(episodes_to_process)
            stats["end_time"] = datetime.now()
            logger.info(f"Cache-only mode: skipped processing {len(episodes_to_process)} episodes")
            return stats

        # Process each episode
        for episode in tqdm(
            episodes_to_process, desc=f"Processing {company_name} episodes"
        ):
            try:
                # Get episode details
                episode_id = episode.get("episode_id")
                episode_title = episode.get("episode_title", "Unknown Episode")
                podcast_id = episode.get("podcast", {}).get("podcast_id", None)
                podcast_name = episode.get("podcast", {}).get(
                    "podcast_name", "Unknown Podcast"
                )
                episode_description = episode.get("episode_description", "")

                # Get episode date
                episode_date_str = episode.get("posted_at")
                episode_date = None
                if episode_date_str:
                    try:
                        episode_date = datetime.fromisoformat(
                            episode_date_str.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass

                # Get transcript
                transcript_text = episode.get("episode_transcript", {})

                if not transcript_text:
                    logger.warning(f"No transcript available for episode: {episode_id}")
                    continue
                
                # Process episode-level sentiment first
                # Check if we already have an episode-level sentiment for this company and episode
                if override_db or not self.db_manager.check_duplicate_sentiment(
                    company_name, episode_id, is_episode_level=True
                ):
                    # Create a company info dictionary for the classifier
                    company_info = [{"name": company_name, "ticker": company_ticker}]
                    
                    # Use the detailed sentiment analysis with override capability
                    detailed_results = self.llm_classifier.analyze_and_override_sentiment(
                        podcast_name=podcast_name,
                        episode_name=episode_title,
                        episode_description=episode_description,
                        transcript=transcript_text,
                        companies=company_info,
                    )
                    
                    if detailed_results:
                        # Prepare episode-level results
                        episode_results = []
                        for result in detailed_results:
                            # If overriding, delete existing sentiment first
                            if override_db:
                                self.db_manager.delete_episode_level_sentiment(company_name, episode_id)
                            
                            # Update episode-level sentiment counts
                            sentiment = result.get("sentiment", "Hold")
                            
                            stats["episode_sentiment_counts"][sentiment] += 1
                            stats["episode_level_classified"] += 1
                            
                            # Prepare episode-level result data
                            episode_result = {
                                "company_name": result["company"],
                                "company_ticker": result["company_ticker"],
                                "podcast_id": podcast_id,
                                "podcast_name": podcast_name,
                                "episode_id": episode_id,
                                "episode_title": episode_title,
                                "episode_description": episode_description,
                                "episode_transcript": transcript_text,
                                "episode_date": episode_date,
                                # No snippet_text for episode-level sentiment
                                "sentiment": sentiment,
                                "processed_at": datetime.utcnow(),
                                "metadata": {
                                    "is_full_episode": True,
                                    "references": result.get("references", []),
                                    "raw_response": result.get("raw_response", ""),
                                    "model": result.get("model", "")
                                },
                            }
                            
                            episode_results.append(episode_result)
                        
                        # Save episode-level results
                        if episode_results:
                            self.db_manager.save_batch_sentiments(episode_results, is_episode_level=True)
                            logger.info(f"Saved episode-level sentiment for {company_name} in episode: {episode_id}")

                # Process snippet-level sentiment if requested
                if snippet_level:
                    # Extract snippets
                    keywords = [company_name, company_ticker]
                    snippets = self.data_processor.extract_snippets(
                        transcript_text, keywords
                    )

                    # Validate snippets are about the company
                    valid_snippets = []
                    for snippet in snippets:
                        if self.data_processor.validate_company_context(
                            snippet["text"], company_name, company_ticker
                        ):
                            valid_snippets.append(snippet)

                    stats["snippets_found"] += len(valid_snippets)

                    # Skip if no valid snippets
                    if not valid_snippets:
                        logger.info(
                            f"No valid snippets found for {company_name} in episode: {episode_id}"
                        )
                    else:
                        # Classify snippets
                        classified_snippets = self.llm_classifier.batch_classify(
                            valid_snippets, company_name
                        )

                        stats["snippets_classified"] += len(classified_snippets)

                        # Prepare results for database
                        results_to_save = []
                        for snippet in classified_snippets:
                            # Check for duplicates
                            if self.db_manager.check_duplicate_sentiment(
                                company_name, episode_id, snippet["text"]
                            ):
                                logger.info(
                                    f"Skipping duplicate snippet for {company_name} in episode: {episode_id}"
                                )
                                continue

                            # Update sentiment counts
                            sentiment = snippet.get("sentiment", "Hold")
                            stats["snippet_sentiment_counts"][sentiment] += 1

                            # Prepare result data
                            result_data = {
                                "company_name": company_name,
                                "company_ticker": company_ticker,
                                "podcast_id": podcast_id,
                                "podcast_name": podcast_name,
                                "episode_id": episode_id,
                                "episode_title": episode_title,
                                "episode_description": episode_description,
                                "episode_transcript": transcript_text,
                                "episode_date": episode_date,
                                "snippet_text": snippet["text"],
                                "sentiment": sentiment,
                                "processed_at": datetime.utcnow(),
                                "metadata": {
                                    "keyword": snippet.get("keyword"),
                                    "position": snippet.get("position"),
                                    "length": snippet.get("length"),
                                },
                            }

                            results_to_save.append(result_data)

                        # Save results to database
                        if results_to_save:
                            self.db_manager.save_batch_sentiments(results_to_save)

                stats["episodes_processed"] += 1

                # Add a small delay between episodes to avoid rate limits
                if not isinstance(self.llm_classifier, MockLLMClassifier):
                    time.sleep(0.2)

            except Exception as e:
                logger.error(
                    f"Error processing episode {episode.get('episode_id')}: {e}",
                    exc_info=True,
                )
                continue

        # Update end time
        stats["end_time"] = datetime.now()
        stats["duration_seconds"] = (
            stats["end_time"] - stats["start_time"]
        ).total_seconds()

        logger.info(f"Completed processing for {company_name}")
        logger.info(f"Stats: {stats}")

        return stats

    def process_all_companies(
        self, max_episodes_per_company: int = None, snippet_level: bool = False,
        override_db: bool = False, max_pages: int = 5, language: Optional[str] = PODCAST_LANGUAGE,
        cache_only: bool = False, before_date: Optional[str] = None
    ) -> List[Dict]:
        """Process all companies defined in the configuration.

        Args:
            max_episodes_per_company: Maximum number of episodes to process per company
            snippet_level: Whether to also perform snippet-level classification (default: False)
            override_db: Whether to override existing sentiment in database (default: False)
            max_pages: Maximum number of pages to fetch from API (default: 5)
            language: Language code to filter podcasts (default: "en")
            cache_only: Whether to only make API calls and cache results without processing (default: False)
            before_date: Only get episodes before this date (format: 'YYYY-MM-DD HH:MM:SS')

        Returns:
            List of processing statistics for each company
        """
        # Check if LLM classifier is available (unless cache_only mode)
        if not cache_only and self.llm_classifier is None:
            raise ValueError(
                "LLM classifier is not initialized. Set it manually before processing."
            )

        all_stats = []

        # Process each company
        for company in COMPANIES:
            try:
                company_name = company["name"]
                company_ticker = company["ticker"]

                logger.info(f"Processing company: {company_name} ({company_ticker})")

                # Process the company
                stats = self.process_company(
                    company_name,
                    company_ticker,
                    max_episodes=max_episodes_per_company,
                    snippet_level=snippet_level,
                    override_db=override_db,
                    max_pages=max_pages,
                    language=language,
                    cache_only=cache_only,
                    before_date=before_date
                )

                all_stats.append(stats)

                # Add a delay between companies to avoid rate limits
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error processing company {company_name}: {e}")
                continue

        # Now, process episodes that mention multiple companies for episode-level sentiment
        # Skip if in cache_only mode
        if not cache_only:
            try:
                self._process_multi_company_episodes(
                    max_episodes_per_company, 
                    override_db, 
                    max_pages, 
                    language,
                    cache_only=cache_only,
                    before_date=before_date
                )
            except Exception as e:
                logger.error(f"Error processing multi-company episodes: {e}")

        return all_stats
        
    def _process_multi_company_episodes(
        self, 
        max_episodes: int = None, 
        override_db: bool = False, 
        max_pages: int = 5, 
        language: Optional[str] = PODCAST_LANGUAGE,
        cache_only: bool = False,
        before_date: Optional[str] = None
    ) -> None:
        """Process episodes that mention multiple companies.
        
        This method finds episodes that mention multiple companies from our list
        and analyzes them for all mentioned companies.
        
        Args:
            max_episodes: Maximum number of episodes to process
            override_db: Whether to override existing sentiment in database
            max_pages: Maximum number of pages to fetch from API (default: 5)
            language: Language code to filter podcasts (default: "en")
            cache_only: Whether to only make API calls and cache results without processing (default: False)
            before_date: Only get episodes before this date (format: 'YYYY-MM-DD HH:MM:SS')
        """
        logger.info("Processing episodes with multiple company mentions")
        
        # Get all episodes that have been processed
        processed_episodes = self.db_manager.get_all_processed_episodes()
        episode_sentiments_processed = 0
        
        if max_episodes:
            processed_episodes = processed_episodes[:max_episodes]
            
        logger.info(f"Found {len(processed_episodes)} processed episodes to analyze")
        
        # If cache_only mode, skip processing and return
        if cache_only:
            logger.info(f"Cache-only mode: skipped processing {len(processed_episodes)} multi-company episodes")
            return

        for episode in tqdm(processed_episodes, desc="Processing multi-company episodes"):
            try:
                episode_id = episode.get("id")
                transcript = episode.get("transcript")
                
                if not transcript:
                    continue
                    
                # Find all companies mentioned in this episode
                mentioned_companies = []
                for company in COMPANIES:
                    company_name = company["name"]
                    company_ticker = company["ticker"]
                    
                    # Check if company is mentioned in transcript
                    if (
                        company_name.lower() in transcript.lower() or 
                        company_ticker.lower() in transcript.lower()
                    ):
                        mentioned_companies.append({
                            "name": company_name,
                            "ticker": company_ticker
                        })
                
                if len(mentioned_companies) > 1:
                    logger.info(f"Episode {episode_id} mentions {len(mentioned_companies)} companies")
                    
                    # Use the detailed sentiment analysis with override capability
                    detailed_results = self.llm_classifier.analyze_and_override_sentiment(
                        podcast_name=episode.get("podcast_name"),
                        episode_name=episode.get("title"),
                        episode_description=episode.get("description"),
                        transcript=transcript,
                        companies=mentioned_companies,
                    )
                    
                    if detailed_results:
                        # Prepare episode-level results
                        episode_results = []
                        for result in detailed_results:
                            company_name = result["company"]
                            company_ticker = result["company_ticker"]
                            
                            # If overriding, delete existing sentiment first
                            if override_db:
                                self.db_manager.delete_episode_level_sentiment(company_name, episode_id)
                            
                            # Prepare episode-level result data
                            sentiment = result.get("sentiment", "Hold")
                                
                            episode_result = {
                                "company_name": company_name,
                                "company_ticker": company_ticker,
                                "podcast_id": episode.get("podcast_id"),
                                "podcast_name": episode.get("podcast_name"),
                                "episode_id": episode_id,
                                "episode_title": episode.get("title"),
                                "episode_description": episode.get("description"),
                                "episode_transcript": transcript,
                                "episode_date": episode.get("date"),
                                # No snippet_text for episode-level sentiment
                                "sentiment": sentiment,
                                "processed_at": datetime.utcnow(),
                                "metadata": {
                                    "is_full_episode": True,
                                    "is_multi_company": True,
                                    "references": result.get("references", []),
                                    "raw_response": result.get("raw_response", ""),
                                    "model": result.get("model", "")
                                },
                            }
                            
                            episode_results.append(episode_result)
                        
                        # Save episode-level results
                        if episode_results:
                            self.db_manager.save_batch_sentiments(episode_results, is_episode_level=True)
                            logger.info(f"Saved multi-company episode-level sentiments for episode: {episode_id}")
                    
                    # Count this as processed
                    episode_sentiments_processed += 1
                    
                # Add a small delay between episodes to avoid rate limits
                if not isinstance(self.llm_classifier, MockLLMClassifier):
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error processing multi-company episode {episode.get('id')}: {e}", exc_info=True)
                continue

    def _deduplicate_episodes(self, episodes: List[Dict]) -> List[Dict]:
        """Remove duplicate episodes from the list.

        Args:
            episodes: List of episode dictionaries

        Returns:
            Deduplicated list of episodes
        """
        unique_episodes = {}

        for episode in episodes:
            episode_id = episode.get("episode_id")
            if episode_id and episode_id not in unique_episodes:
                unique_episodes[episode_id] = episode

        return list(unique_episodes.values())
