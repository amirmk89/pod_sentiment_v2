"""
Data Processor for extracting and validating relevant text snippets from API responses.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

from config import SNIPPET_CONFIG

# Configure logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processor for extracting and validating text snippets from podcast transcripts.
    This is de-facto is the chunking logic.
    """

    def __init__(self, context_window: Optional[int] = None):
        """Initialize the data processor.

        Args:
            context_window: Number of characters to include before and after the mention
        """
        self.context_window = context_window or SNIPPET_CONFIG["context_window"]
        self.min_length = SNIPPET_CONFIG["min_snippet_length"]
        self.max_length = SNIPPET_CONFIG["max_snippet_length"]

        logger.info(
            f"Data processor initialized with context window: {self.context_window} chars"
        )

    def extract_snippets(self, transcript_text: str, keywords: List[str]) -> List[Dict]:
        """Extract relevant snippets from transcript text containing the keywords.

        Args:
            transcript_text: Full transcript text
            keywords: List of keywords to search for (company name, ticker)

        Returns:
            List of dictionaries containing snippets and metadata
        """
        snippets = []

        # Normalize text for case-insensitive search
        normalized_text = transcript_text.lower()

        for keyword in keywords:
            # Create a regex pattern that matches the keyword as a whole word
            # This helps avoid partial matches (e.g., "Apple" in "Pineapple")
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

            # Find all matches
            for match in re.finditer(pattern, normalized_text):
                start_pos = match.start()
                end_pos = match.end()

                # Extract context around the mention
                snippet_start = max(0, start_pos - self.context_window)
                snippet_end = min(len(transcript_text), end_pos + self.context_window)

                # Get the snippet with context
                snippet = transcript_text[snippet_start:snippet_end]

                # Clean up the snippet
                snippet = self._clean_snippet(snippet)

                # Validate the snippet
                if self._is_valid_snippet(snippet, keyword):
                    snippets.append(
                        {
                            "text": snippet,
                            "keyword": keyword,
                            "position": (start_pos, end_pos),
                            "length": len(snippet),
                        }
                    )

        # Remove duplicates (snippets that overlap significantly)
        unique_snippets = self._remove_duplicate_snippets(snippets)

        logger.info(f"Extracted {len(unique_snippets)} unique snippets from transcript")
        return unique_snippets

    def _clean_snippet(self, text: str) -> str:
        """Clean up the snippet text.

        Args:
            text: Raw snippet text

        Returns:
            Cleaned snippet text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Ensure the snippet ends with a sentence boundary if possible
        if len(text) > 100:  # Only for longer snippets
            # Try to end at a sentence boundary
            sentence_end_match = re.search(r"[.!?]\s+[A-Z](?=[^.!?]*$)", text)
            if sentence_end_match:
                # End at the last sentence boundary
                text = text[: sentence_end_match.start() + 1]

        return text

    def _is_valid_snippet(self, snippet: str, keyword: str) -> bool:
        """Check if a snippet is valid and relevant.

        Args:
            snippet: Extracted snippet text
            keyword: The keyword that was matched

        Returns:
            True if the snippet is valid, False otherwise
        """
        # Check length constraints
        if len(snippet) < self.min_length:
            return False

        if len(snippet) > self.max_length:
            return False

        # Verify the keyword is actually in the snippet (case insensitive)
        if keyword.lower() not in snippet.lower():
            return False

        # Optional: Add more sophisticated validation here
        # For example, check if the keyword is used in a financial context

        return True

    def _remove_duplicate_snippets(self, snippets: List[Dict]) -> List[Dict]:
        """Remove duplicate or highly overlapping snippets.

        Args:
            snippets: List of extracted snippets

        Returns:
            Deduplicated list of snippets
        """
        if not snippets:
            return []

        # Sort snippets by position
        sorted_snippets = sorted(snippets, key=lambda x: x["position"][0])

        unique_snippets = [sorted_snippets[0]]

        for snippet in sorted_snippets[1:]:
            last_snippet = unique_snippets[-1]

            # Check if this snippet overlaps significantly with the previous one
            last_end = last_snippet["position"][1]
            current_start = snippet["position"][0]

            # If there's less than 50% overlap, consider it a new snippet
            if (
                current_start > last_end
                or (current_start - last_snippet["position"][0])
                > (last_end - last_snippet["position"][0]) * 0.5
            ):
                unique_snippets.append(snippet)

        return unique_snippets

    def validate_company_context(
        self, snippet: str, company_name: str, ticker: str
    ) -> bool:
        """Validate that the snippet is actually referring to the company.

        This is a simple implementation that could be enhanced with more sophisticated NLP.

        Args:
            snippet: Text snippet
            company_name: Company name
            ticker: Company ticker symbol

        Returns:
            True if the snippet likely refers to the company, False otherwise
        """
        # Convert to lowercase for case-insensitive matching
        snippet_lower = snippet.lower()
        company_lower = company_name.lower()
        ticker_lower = ticker.lower()

        # Check for the company name or ticker
        if company_lower in snippet_lower or ticker_lower in snippet_lower:
            # Additional validation could be added here
            # For example, check for common false positives

            # For "Apple", check it's not referring to the fruit
            if company_lower == "apple" and re.search(
                r"\b(fruit|juice|pie|cider)\b", snippet_lower
            ):
                # Check if it also mentions the ticker or tech-related terms
                if ticker_lower in snippet_lower or re.search(
                    r"\b(tech|iphone|mac|computer)\b", snippet_lower
                ):
                    return True
                return False

            return True

        return False
