"""
LLM Classifier for sentiment analysis of podcast snippets.
"""

import os
import logging
import time
from typing import Dict, Optional, List, Tuple
from config import LLM_CONFIG

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClassifier:
    """Classifier for sentiment analysis using OpenAI's Chat API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ):
        """
        Initialize the LLM classifier.

        Args:
            api_key: API key for OpenAI. If not provided, it will be read from the OPENAI_API_KEY environment variable.
            model: The chat model to use (default is 'gpt-3.5-turbo').
            temperature: Sampling temperature for the completion (default 0.0 for deterministic output).
            max_tokens: Maximum tokens for the response (default 10).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as a parameter."
            )

        # Set API key globally for openai module
        openai.api_key = self.api_key

        self.model = model or LLM_CONFIG["model"]
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"LLMClassifier initialized with model: {self.model}")

    def classify_sentiment(self, snippet: str, company: str) -> str:
        """
        Classify the sentiment of a snippet towards a company.

        Args:
            snippet: The text snippet to classify.
            company: The company name or ticker.

        Returns:
            A sentiment classification as a string: "Buy", "Hold", or "Sell".
        """
        prompt = self._create_prompt(snippet, company)
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analyzer that classifies text as Buy/Outperform, Hold, or Sell/Underperform. Respond with ONLY one word: Buy, Hold, or Sell.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1,
            )
            classification = response.choices[0].message["content"].strip()
            return self._normalize_classification(classification)
        except Exception as e:
            logger.error(f"Failed to classify sentiment: {e}")
            return "Hold"

    def _create_prompt(self, snippet: str, company: str) -> str:
        """
        Create a prompt for the LLM.

        Args:
            snippet: The text snippet.
            company: The company name or ticker.

        Returns:
            A formatted prompt string.
        """
        return f'Classify the sentiment towards {company}. Text: "{snippet}". Respond with only one word: Buy, Hold, or Sell.'

    def _normalize_classification(self, classification: str) -> str:
        """
        Normalize the raw classification output to one of: Buy, Hold, or Sell.

        Args:
            classification: The raw classification response from the LLM.

        Returns:
            A normalized classification string.
        """
        classification_lower = classification.lower().strip()
        if "buy" in classification_lower:
            return "Buy"
        elif "sell" in classification_lower:
            return "Sell"
        else:
            return "Hold"

    def batch_classify(self, snippets: List[Dict], company: str) -> List[Dict]:
        """
        Batch classify multiple snippets to optimize API usage.

        Args:
            snippets: A list of dictionaries where each contains a 'text' key with the snippet.
            company: The company name or ticker.

        Returns:
            A list of dictionaries with an added 'sentiment' key for each snippet.
        """
        classified_snippets = []
        for snippet in snippets:
            # Add a small delay between requests to avoid rate limits
            time.sleep(0.5)
            sentiment = self.classify_sentiment(snippet.get("text", ""), company)
            snippet_with_sentiment = snippet.copy()
            snippet_with_sentiment["sentiment"] = sentiment
            classified_snippets.append(snippet_with_sentiment)
        return classified_snippets
        
    def classify_episode_for_companies(self, transcript: str, companies: List[Dict]) -> List[Dict]:
        """
        Classify the sentiment of an entire episode transcript for multiple companies.
        
        Args:
            transcript: The full episode transcript.
            companies: A list of dictionaries with 'name' and 'ticker' keys.
            
        Returns:
            A list of dictionaries with company information and sentiment classification.
        """
        results = []
        for company in companies:
            company_name = company.get("name", "")
            company_ticker = company.get("ticker", "")
            
            if not company_name:
                logger.warning(f"Skipping company with no name: {company}")
                continue
                
            # Add a small delay between requests to avoid rate limits
            time.sleep(0.5)
            
            # Classify sentiment for this company
            sentiment = self.classify_sentiment(transcript, company_name)
            
            # Create result dictionary
            result = {
                "company_name": company_name,
                "company_ticker": company_ticker,
                "sentiment": sentiment,
                # No snippet_text for episode-level sentiment
            }
            
            results.append(result)
            logger.info(f"Classified episode-level sentiment for {company_name}: {sentiment}")
            
        return results
        
    def analyze_episode_detailed(
        self, 
        podcast_name: str, 
        episode_name: str, 
        episode_description: str, 
        transcript: str, 
        company: str
    ) -> Dict:
        """
        Analyze an episode with detailed sentiment analysis for a specific company.
        
        This method uses a more detailed prompt that extracts sentiment and references
        from the podcast metadata and transcript.
        
        Args:
            podcast_name: Name of the podcast
            episode_name: Title of the episode
            episode_description: Description of the episode
            transcript: Full transcript of the episode
            company: Company name to analyze sentiment for
            
        Returns:
            A dictionary containing sentiment classification and references
        """
        logger.info(f"Analyzing detailed sentiment for {company} in episode: {episode_name}")
        
        # Create a prompt with all the podcast metadata and transcript
        prompt = f"""
Podcast: {podcast_name}
Episode: {episode_name}
Description: {episode_description}

Transcript:
{transcript}
"""
        
        try:
            # Use a higher max_tokens value for the detailed response
            # TODO: handle rate limits: https://cookbook.openai.com/examples/how_to_handle_rate_limits
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"This is a podcast transcript with metadata. Extract the financial sentiment related to {company} stock. Decide between [Buy, Hold, Sell, No sentiment] and provide references from the text.\nIf this isn't a discussion about the company stock but just a mention, return a \"No sentiment\".\nReturn the following format:\n* Sentiment: <...>\n* References:\n    * <...>\n    * <...>",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Slightly higher temperature for more natural responses
                max_tokens=300,   # Allow more tokens for the detailed response with references
                n=1,
            )
            
            result = response.choices[0].message["content"].strip()
            
            # Log the raw response
            # logger.info(f"Raw LLM response for {company}: {result}")
            
            # Parse the result to extract sentiment and references
            sentiment, references = self._parse_detailed_result(result)
            
            # Create response dictionary with raw_response included for DB storage
            response_data = {
                "company": company,
                "sentiment": sentiment,
                "references": references,
                "raw_response": result,
                "model": self.model,
                "timestamp": time.time()
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to analyze detailed sentiment: {e}")
            return {
                "company": company,
                "sentiment": "No sentiment",
                "references": [],
                "raw_response": "",
                "error": str(e),
                "timestamp": time.time()
            }
    
    @staticmethod
    def _parse_detailed_result(result: str) -> Tuple[str, List[str]]:
        """
        Parse the detailed result from the LLM to extract sentiment and references.
        
        Args:
            result: The raw response from the LLM
            
        Returns:
            A tuple containing (sentiment, list of references)
        """
        sentiment = "No sentiment"
        references = []
        
        # Simple parsing logic - can be improved for more robust handling
        lines = result.split('\n')
        for i, line in enumerate(lines):
            if "* Sentiment:" in line or "Sentiment:" in line:
                # Extract sentiment
                sentiment_part = line.split(":", 1)[1].strip() if ":" in line else ""
                if "buy" in sentiment_part.lower():
                    sentiment = "Buy"
                elif "sell" in sentiment_part.lower():
                    sentiment = "Sell"
                elif "hold" in sentiment_part.lower():
                    sentiment = "Hold"
                elif "no sentiment" in sentiment_part.lower():
                    sentiment = "No sentiment"
            
            # Extract references (lines after "* References:" that start with "*" or "-")
            if i > 0 and ("* References:" in lines[i-1] or "References:" in lines[i-1]):
                if line.strip().startswith("*") or line.strip().startswith("-"):
                    ref = line.strip().lstrip("*-").strip()
                    if ref:
                        references.append(ref)
            elif i > 1 and ("* References:" in lines[i-2] or "References:" in lines[i-2]):
                if line.strip().startswith("*") or line.strip().startswith("-"):
                    ref = line.strip().lstrip("*-").strip()
                    if ref:
                        references.append(ref)
        
        return sentiment, references

    def analyze_and_override_sentiment(
        self,
        podcast_name: str,
        episode_name: str,
        episode_description: str,
        transcript: str,
        companies: List[Dict],
    ) -> List[Dict]:
        """
        Analyze episode for multiple companies with detailed sentiment analysis
        and optionally override existing database sentiment.
        
        Args:
            podcast_name: Name of the podcast
            episode_name: Title of the episode
            episode_description: Description of the episode
            transcript: Full transcript of the episode
            companies: List of dictionaries with 'name' and 'ticker' keys
            
        Returns:
            List of dictionaries with detailed sentiment analysis results
        """
        results = []
        
        for company in companies:
            company_name = company.get("name", "")
            company_ticker = company.get("ticker", "")
            
            if not company_name:
                logger.warning(f"Skipping company with no name: {company}")
                continue
                
            # Add a small delay between requests to avoid rate limits
            time.sleep(0.1)
            
            # Get detailed sentiment analysis
            detailed_result = self.analyze_episode_detailed(
                podcast_name=podcast_name,
                episode_name=episode_name,
                episode_description=episode_description,
                transcript=transcript,
                company=company_name
            )
            
            # Add ticker to the result
            detailed_result["company_ticker"] = company_ticker
            
            results.append(detailed_result)
            
        return results


if __name__ == "__main__":
    # Test basic sentiment classification
    test_snippet = "Company X has achieved record-breaking revenue growth last quarter."
    test_company = "Company X"
    classifier = LLMClassifier()
    sentiment = classifier.classify_sentiment(test_snippet, test_company)
    print(f"Basic sentiment for test snippet: {sentiment}")
    
    # Test detailed sentiment analysis
    test_podcast = "Financial Insights"
    test_episode = "Q3 Tech Earnings"
    test_description = "In this episode, we discuss the latest earnings reports from major tech companies."
    test_transcript = """
    Host: Welcome to Financial Insights. Today we're discussing Microsoft's latest earnings report.
    
    Analyst 1: Microsoft's cloud business is showing remarkable growth. Azure revenue is up 27% year-over-year.
    
    Analyst 2: Their guidance was strong too. I think they're positioned well for the AI revolution.
    
    Host: Would you recommend buying Microsoft stock at current levels?
    
    Analyst 1: Absolutely. The valuation is reasonable considering their growth trajectory and dominant position in enterprise.
    
    Analyst 2: I agree. Their diversified revenue streams and strong cash flow make them a solid buy.
    """
    
    test_companies = [{"name": "Microsoft", "ticker": "MSFT"}]
    
    # Test the detailed analysis with override option
    detailed_results = classifier.analyze_and_override_sentiment(
        podcast_name=test_podcast,
        episode_name=test_episode,
        episode_description=test_description,
        transcript=test_transcript,
        companies=test_companies,
    )
    
    # Print detailed results
    for result in detailed_results:
        print(f"\nDetailed analysis for {result['company']}:")
        print(f"Sentiment: {result['sentiment']}")
        print("References:")
        for ref in result['references']:
            print(f"  - {ref}")
