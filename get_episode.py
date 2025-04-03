#!/usr/bin/env python3
"""
Minimal example script to fetch a podcast episode by ID from the Podscan.fm API
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("PODSCAN_API_KEY")
if not API_KEY:
    print("Error: PODSCAN_API_KEY environment variable not set")
    sys.exit(1)

def get_episode(episode_id):
    """
    Fetch a podcast episode by its ID
    
    Args:
        episode_id: The ID of the episode to fetch
        
    Returns:
        The episode data as a dictionary
    """
    # API endpoint for getting a single episode
    url = f"https://podscan.fm/api/v1/episodes/{episode_id}"
    
    # Headers with authentication
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Make the request
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching episode: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            try:
                print(f"Response: {e.response.json()}")
            except:
                print(f"Response text: {e.response.text}")
        sys.exit(1)

def main():
    # Check for episode ID argument
    # if len(sys.argv) < 2:
    #     print("Usage: python get_episode.py <episode_id>")
    #     sys.exit(1)
    
    # episode_id = sys.argv[1]
    episode_id = "ep_q6l7rvgog2k8ra4w"
    print(f"Fetching episode with ID: {episode_id}")
    
    # Get the episode
    episode = get_episode(episode_id)
    
    # Print the result (pretty-printed JSON)
    print(json.dumps(episode, indent=2))

if __name__ == "__main__":
    main() 
    # Dashboard link format: https://podscan.fm/dashboard/podcasts/pd_k42yajr2ovw9p8ow/episode/ep_q6l7rvgog2k8ra4w
    # I.e., https://podscan.fm/dashboard/podcasts/<podcast_id>/episode/<episode_id>