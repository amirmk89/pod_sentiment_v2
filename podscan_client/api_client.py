"""
API Client for interacting with the Podscan.fm API.
Handles authentication, rate limiting, and pagination.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any
import requests
import json
from datetime import datetime
from pathlib import Path
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv

from config import API_BASE_URL, API_ENDPOINTS, RATE_LIMITS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PodcastAPIClient:
    """Client for interacting with the Podscan.fm API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = "cache",
    ):
        """Initialize the API client.

        Args:
            api_key: API key for authentication. If not provided, will try to load from environment.
            use_cache: Whether to use caching for API responses
            cache_dir: Directory to store cached responses
        """
        self.api_key = api_key or os.getenv("PODSCAN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set PODSCAN_API_KEY environment variable or pass as parameter."
            )

        self.base_url = API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        # Configure rate limits
        self.requests_per_minute = RATE_LIMITS["requests_per_minute"]
        self.requests_per_day = RATE_LIMITS["requests_per_day"]

        # Configure caching
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)

        # Load categories
        self.categories = self._load_categories()

        logger.info(
            f"API client initialized with rate limits: {self.requests_per_minute}/min, {self.requests_per_day}/day"
        )
        if self.use_cache:
            logger.info(f"Caching enabled, using directory: {self.cache_dir}")

    def _load_categories(self) -> Dict[str, Dict[str, str]]:
        """Load categories from the categories.json file.

        Returns:
            Dictionary mapping category names to their details (id and display name)
        """
        try:
            with open("categories.json", "r") as f:
                categories_data = json.load(f)

            # Create a mapping of category names to their IDs and display names
            categories_map = {}
            for category in categories_data.get("categories", []):
                categories_map[category["category_name"]] = {
                    "id": category["category_id"],
                    "display_name": category["category_display_name"],
                }

            logger.info(f"Loaded {len(categories_map)} categories")
            return categories_map
        except Exception as e:
            logger.warning(f"Failed to load categories: {e}")
            return {}

    def category_names_to_ids(self, category_names: List[str]) -> str:
        """Convert a list of category names to a comma-separated list of category IDs.

        Args:
            category_names: List of category names to convert

        Returns:
            Comma-separated string of category IDs
        """
        if not category_names or not self.categories:
            return ""

        category_ids = []
        for name in category_names:
            name = name.lower()
            if name in self.categories:
                category_ids.append(self.categories[name]["id"])
            else:
                logger.warning(f"Category '{name}' not found in categories list")

        return ",".join(category_ids)

    def _get_cache_key(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> str:
        """Generate a unique cache key for a request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body

        Returns:
            A string hash to use as cache key
        """
        # Create a dictionary of all request components
        cache_dict = {
            "method": method,
            "endpoint": endpoint,
            "params": params or {},
            "data": data or {},
        }

        # Convert to a string and hash it for the filename
        cache_str = json.dumps(cache_dict, sort_keys=True)
        return f"{hash(cache_str)}.json"

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Try to get a response from cache.

        Args:
            cache_key: Cache key for the request

        Returns:
            Cached response or None if not found
        """
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / cache_key
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    logger.info(f"Cache hit: {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_key}: {e}")

        return None

    def _save_to_cache(self, cache_key: str, response_data: Dict) -> None:
        """Save a response to the cache.

        Args:
            cache_key: Cache key for the request
            response_data: Response data to cache
        """
        if not self.use_cache:
            return

        # Add timestamp to cached data
        cache_data = {"timestamp": datetime.now().isoformat(), "data": response_data}

        cache_file = self.cache_dir / cache_key
        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            logger.info(f"Saved response to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to write to cache file {cache_key}: {e}")

    @sleep_and_retry
    @limits(calls=10, period=60)
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        force_refresh: bool = False,
    ) -> Dict:
        """Make a rate-limited request to the API with caching.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body for POST requests
            force_refresh: Whether to bypass cache and force a new request

        Returns:
            API response as dictionary
        """
        # Check cache first if not forcing refresh  - Todo: removed, breaks pagination
        # if not force_refresh:
        #     cache_key = self._get_cache_key(method, endpoint, params, data)
        #     cached_response = self._get_from_cache(cache_key)
        #     if cached_response:
        #         return cached_response["data"]

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=30,
            )

            # Check for rate limiting headers and adjust if needed
            if "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                if remaining < 5:
                    logger.warning(
                        f"Rate limit approaching: {remaining} requests remaining"
                    )
                    # Add additional delay to avoid hitting the limit
                    time.sleep(2)

            response.raise_for_status()
            response_data = response.json()

            # Cache the successful response
            # if not force_refresh:
            # cache_key = self._get_cache_key(method, endpoint, params, data)
            # self._save_to_cache(cache_key, response_data)

            return response_data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")

            # Handle rate limiting errors
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    logger.warning(
                        f"Rate limit exceeded. Waiting {retry_after} seconds"
                    )
                    time.sleep(retry_after)
                    return self._make_request(
                        method, endpoint, params, data, force_refresh
                    )

                # Log the error response for debugging
                try:
                    error_data = e.response.json()
                    logger.error(f"API error response: {error_data}")
                except ValueError:
                    logger.error(f"API error status code: {e.response.status_code}")

            # Re-raise the exception after logging
            raise

    def search_episodes(
        self,
        query: str,
        page: int = 1,
        limit: int = 50,
        category_names: Optional[List[str]] = None,
        language: Optional[str] = None,
        force_refresh: bool = False,
        before_date: Optional[str] = None,
    ) -> Dict:
        """Search for podcast episodes containing the query.

        Args:
            query: Search query (company name or ticker)
            page: Page number for pagination
            limit: Number of results per page
            category_names: List of category names to filter by
            language: Language code to filter podcasts (e.g., "en" for English)
            force_refresh: Whether to bypass cache and force a new request
            before_date: Only get episodes before this date (format: 'YYYY-MM-DD HH:MM:SS')

        Returns:
            Search results with episode data
        """
        endpoint = API_ENDPOINTS["search"]
        params = {"query": query, "per_page": limit, "page": page}

        # Add category filtering if provided
        if category_names:
            category_ids = self.category_names_to_ids(category_names)
            if category_ids:
                params["category_ids"] = category_ids
                # logger.info(f"Filtering by categories: {', '.join(category_names)}")
                
        # Add language filtering if provided
        if language:
            params["podcast_language"] = language
            # logger.info(f"Filtering by language: {language}")
            
        # Add before_date parameter if provided
        if before_date:
            params["before"] = before_date
            # logger.info(f"Filtering by before date: {before_date}")

        logger.info(f"Searching episodes for query: {query} (page {page})")
        return self._make_request(
            "GET", endpoint, params=params, force_refresh=force_refresh
        )

    def get_transcript(self, episode_id: str, force_refresh: bool = False) -> Dict:
        """Get the transcript for a specific episode.

        Args:
            episode_id: ID of the episode
            force_refresh: Whether to bypass cache and force a new request

        Returns:
            Episode transcript data
        """
        endpoint = API_ENDPOINTS["transcript"].format(episode_id=episode_id)

        logger.info(f"Fetching transcript for episode: {episode_id}")
        return self._make_request("GET", endpoint, force_refresh=force_refresh)

    def search_all_pages(
        self,
        query: str,
        max_pages: int = 5,
        category_names: Optional[List[str]] = None,
        language: Optional[str] = None,
        force_refresh: bool = False,
        before_date: Optional[str] = None,
    ) -> List[Dict]:
        """Search across multiple pages and combine results.

        Args:
            query: Search query
            max_pages: Maximum number of pages to retrieve (default: 5, can be up to 50)
            category_names: List of category names to filter by
            language: Language code to filter podcasts (e.g., "en" for English)
            force_refresh: Whether to bypass cache and force a new request
            before_date: Only get episodes before this date (format: 'YYYY-MM-DD HH:MM:SS')

        Returns:
            Combined list of episode results
        """
        # Generate cache key for the entire multi-page search
        cache_key_parts = [f"search_{query}_pages_{max_pages}_limit_50"]
        if category_names:
            sorted_categories = sorted(category_names)
            cat_count = len(sorted_categories)
            first_three = sorted_categories[:3] if cat_count >= 3 else sorted_categories
            cache_key_parts.append(f"cats_{'-'.join(first_three)}_total_{cat_count}")
        if language:
            cache_key_parts.append(f"lang_{language}")
        if before_date:
            cache_key_parts.append(f"before_{before_date.replace(' ', '_').replace(':', '')}")
        cache_key = f"{'_'.join(cache_key_parts)}.json"

        # Check cache first if not forcing refresh
        if not force_refresh and self.use_cache:
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                date_info = f", before date: {before_date}" if before_date else ""
                logger.info(
                    f"Using cached search results for query: {query} (max_pages: {max_pages}{date_info})"
                )
                return cached_response["data"]

        # If not in cache or force refresh, perform the search
        all_results = []
        current_page = 1

        if max_pages > 20:
            logger.warning(f"Warning: max_pages is greater than 20. This exceeds the API limit for a page size of 50 (==1000 episodes).")
            max_pages = 20

        while current_page <= max_pages:
            response = self.search_episodes(
                query,
                page=current_page,
                category_names=category_names,
                language=language,
                force_refresh=force_refresh,
                before_date=before_date,
            )

            # Extract episodes from the response
            episodes = response.get("episodes", [])
            
            # Break if we get an empty episodes list
            if not episodes:
                logger.info(f"No more episodes found for query: {query} on page {current_page}")
                break
                
            all_results.extend(episodes)

            # Check if there are more pages
            pagination = response.get("pagination", {})
            last_page = pagination.get("last_page", 1)

            if current_page >= last_page:
                logger.info(f"Reached last page: {last_page} for query: {query}")
                break

            current_page += 1

            # Add a small delay between page requests
            time.sleep(1)

        logger.info(f"Retrieved {len(all_results)} total episodes for query: {query}")

        # Cache the results
        if self.use_cache and not force_refresh:
            self._save_to_cache(cache_key, all_results)
            date_info = f", before date: {before_date}" if before_date else ""
            logger.info(
                f"Cached search results for query: {query} (max_pages: {max_pages}{date_info})"
            )

        return all_results
