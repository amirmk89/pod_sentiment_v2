"""
Configuration settings for the Podcast Sentiment Analyzer.
"""

# Companies to track (name and ticker pairs)
COMPANIES = [
    {"name": "Microsoft", "ticker": "MSFT"},
    {"name": "Apple", "ticker": "AAPL"},
    {"name": "Tesla", "ticker": "TSLA"},
    {"name": "Amazon", "ticker": "AMZN"},
    {"name": "Google", "ticker": "GOOGL"},
    {"name": "Meta", "ticker": "META"},
    {"name": "Netflix", "ticker": "NFLX"},
    {"name": "Nvidia", "ticker": "NVDA"},
    # Bottom of S&P 500
    {"name": "Newell Brands", "ticker": "NWL"},  # Debug for "Brands" shadowing the name and think of a solution
    # {"name": "Under Armour", "ticker": "UA"},
    # Russell 2000
    {"name": "Rocket Pharmaceuticals", "ticker": "RCKT"},
    {"name": "Axonics", "ticker": "AXNX"},
    {"name": "CryoPort", "ticker": "CYRX"},
    # FTSE
    # {"name": "Diageo", "ticker": "DEO"},
    # {"name": "Unilever", "ticker": "UL"},
    {"name": "AstraZeneca", "ticker": "AZN"},
    # Nikkei
    {"name": "Sony Group", "ticker": "SONY"},
    {"name": "Mitsubishi UFJ", "ticker": "MUFG"},
    {"name": "SoftBank Group", "ticker": "SFTBY"},
]

# API Settings
API_BASE_URL = "https://podscan.fm/api/v1"
API_ENDPOINTS = {
    "search": "/episodes/search",
    "transcript": "/episodes/{episode_id}/transcript",
}

# Rate Limiting
RATE_LIMITS = {
    "requests_per_minute": 10,  # Adjust based on your Podscan plan
    "requests_per_day": 100,  # Adjust based on your Podscan plan
}

# LLM Settings
LLM_CONFIG = {
    "provider": "openai",  # Options: "openai", "huggingface"
    "model": "gpt-4o-mini-2024-07-18",  # For OpenAI
    "temperature": 0.0,  # Lower temperature for more deterministic outputs
    "max_tokens": 50,  # We only need a short response (Buy/Hold/Sell)
}

# Database Settings
DB_CONFIG = {
    "db_path": "podcast_sentiment.db",
    "table_name": "sentiment_results",
}

# Snippet Processing
SNIPPET_CONFIG = {
    "context_window": 200,  # Characters before and after the mention to include
    "min_snippet_length": 50,  # Minimum characters for a valid snippet
    "max_snippet_length": 1000,  # Maximum characters for a valid snippet
}

# Logging
LOG_CONFIG = {
    "log_file": "sentiment_analyzer.log",
    "log_level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
}
