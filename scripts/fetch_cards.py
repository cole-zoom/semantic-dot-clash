#!/usr/bin/env python3
"""
Script to fetch all cards from the Clash Royale API.

This script retrieves the complete list of cards from the official Clash Royale
API and stores them in a JSON file.

Environment Variables:
    CLASH_ROYALE_API_KEY: Your Clash Royale API key from developer.clashroyale.com

Usage:
    python scripts/fetch_cards.py
    python scripts/fetch_cards.py --output data/cards.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# Clash Royale API base URL
API_BASE_URL = "https://api.clashroyale.com/v1"


def get_api_key() -> str:
    """Get the Clash Royale API key from environment variables."""
    api_key = os.environ.get("CLASH_ROYALE_API_KEY")
    if not api_key:
        raise ValueError(
            "CLASH_ROYALE_API_KEY environment variable is required.\n"
            "Get your API key from: https://developer.clashroyale.com/"
        )
    return api_key


def fetch_cards(api_key: str) -> dict:
    """
    Fetch all cards from the Clash Royale API.
    
    Args:
        api_key: The Clash Royale API key
        
    Returns:
        Dictionary containing the cards data
    """
    url = f"{API_BASE_URL}/cards"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch all cards from the Clash Royale API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Fetch cards and save to default location:
    python scripts/fetch_cards.py

  Fetch cards and save to custom location:
    python scripts/fetch_cards.py --output my_cards.json
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/cards.json",
        help="Output file path for the cards JSON (default: data/cards.json)"
    )
    
    args = parser.parse_args()
    
    print("🃏 Clash Royale Card Fetcher")
    print("=" * 50)
    print()
    
    # Get API key
    try:
        api_key = get_api_key()
        print("✅ API key loaded from environment")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Fetch cards
    print("📡 Fetching cards from Clash Royale API...")
    try:
        cards_data = fetch_cards(api_key)
    except requests.exceptions.HTTPError as e:
        print(f"❌ API request failed: {e}")
        if e.response.status_code == 403:
            print("   Your API key may be invalid or not authorized for this IP.")
            print("   Check your key settings at: https://developer.clashroyale.com/")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        sys.exit(1)
    
    # Count items
    items_count = len(cards_data.get("items", []))
    support_items_count = len(cards_data.get("supportItems", []))
    print(f"✅ Retrieved {items_count} cards and {support_items_count} support items")
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    print(f"💾 Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cards_data, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"🎉 Done! Cards saved to {output_path}")
    print()
    
    # Print summary of card rarities
    if items_count > 0:
        print("📊 Card Summary:")
        rarities = {}
        for card in cards_data.get("items", []):
            rarity = card.get("rarity", "Unknown")
            rarities[rarity] = rarities.get(rarity, 0) + 1
        
        for rarity, count in sorted(rarities.items()):
            print(f"   - {rarity}: {count}")


if __name__ == "__main__":
    main()
