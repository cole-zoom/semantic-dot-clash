#!/usr/bin/env python3
"""
Script to scrape card descriptions from the Clash Royale Fandom wiki.

This script fetches card descriptions from the Card Overviews wiki page and
matches them to the cards in my_cards.json, creating an enriched dataset.

Usage:
    # From a local HTML file (recommended - save page from browser first):
    python scripts/scrape_card_descriptions.py --html data/card_overviews.html

    # Try fetching from URL (may fail due to bot protection):
    python scripts/scrape_card_descriptions.py

    # Custom input/output:
    python scripts/scrape_card_descriptions.py --html data/card_overviews.html --input data/my_cards.json --output data/enriched.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Wiki URL for card overviews
WIKI_URL = "https://clashroyale.fandom.com/wiki/Card_Overviews"


def normalize_name(name: str) -> str:
    """
    Normalize a card name for matching.
    
    - Converts to lowercase
    - Removes special characters (periods, apostrophes, etc.)
    - Replaces spaces/underscores with single space
    - Strips whitespace
    
    Args:
        name: The card name to normalize
        
    Returns:
        Normalized name string
    """
    # Convert to lowercase
    name = name.lower()
    # Remove special characters but keep spaces
    name = re.sub(r"[.'_\-]", "", name)
    # Replace multiple spaces with single space
    name = re.sub(r"\s+", " ", name)
    # Strip whitespace
    name = name.strip()
    return name


def scrape_card_descriptions(html_file: str | None = None) -> dict[str, str]:
    """
    Scrape card descriptions from the Clash Royale Fandom wiki.
    
    Args:
        html_file: Optional path to a local HTML file. If not provided,
                   attempts to fetch from the wiki URL (may fail due to bot protection).
    
    Returns:
        Dictionary mapping normalized card names to descriptions
    """
    if html_file:
        # Read from local HTML file
        html_path = Path(html_file)
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_file}")
        
        print(f"Reading from local file: {html_file}")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        # Fetch from URL
        print(f"Fetching wiki page: {WIKI_URL}")
        print("(If this fails with 403, save the page from your browser and use --html)")
        
        # Use comprehensive headers to avoid 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        
        # Use a session for better connection handling
        session = requests.Session()
        response = session.get(WIKI_URL, headers=headers, timeout=30)
        response.raise_for_status()
        html_content = response.text
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    descriptions = {}
    
    # Find all card-overview divs - these contain individual card info
    card_overviews = soup.find_all("div", {"class": "card-overview"})
    
    if not card_overviews:
        # Fallback: try finding mw-parser-output content
        content = soup.find("div", {"class": "mw-parser-output"})
        if content:
            card_overviews = content.find_all("div", {"class": "card-overview"})
    
    print(f"Found {len(card_overviews)} card overview sections")
    
    for card_div in card_overviews:
        # Get card name from h4 > span.mw-headline > a
        h4 = card_div.find("h4")
        if not h4:
            continue
        
        headline = h4.find("span", {"class": "mw-headline"})
        if not headline:
            continue
        
        # Get the card name - prefer the link text if available
        link = headline.find("a")
        if link:
            card_name = link.get_text(strip=True)
        else:
            card_name = headline.get_text(strip=True)
        
        if not card_name:
            continue
        
        # Find the description paragraph
        # It's in a p tag inside the card-overview div
        p_tag = card_div.find("p")
        if not p_tag:
            continue
        
        description = p_tag.get_text(strip=True)
        if not description:
            continue
        
        # Clean up description - remove evolution/hero form parts
        # Split on "For X Cycle(s)" patterns
        main_desc = re.split(
            r"For \d+ Cycles?,",
            description,
            flags=re.IGNORECASE
        )[0].strip()
        
        normalized = normalize_name(card_name)
        descriptions[normalized] = main_desc
    
    return descriptions


def load_cards(input_path: str) -> dict:
    """Load cards from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_cards(
    cards_data: dict,
    descriptions: dict[str, str]
) -> tuple[dict, list[str], list[str]]:
    """
    Match card descriptions to cards from the API.
    
    Args:
        cards_data: The cards data from my_cards.json
        descriptions: Dictionary of normalized names to descriptions
        
    Returns:
        Tuple of (enriched cards data, matched names, unmatched names)
    """
    matched = []
    unmatched = []
    
    # Process both items and supportItems if they exist
    for key in ["items", "supportItems"]:
        if key not in cards_data:
            continue
            
        for card in cards_data[key]:
            card_name = card.get("name", "")
            normalized = normalize_name(card_name)
            
            # Try direct match first
            if normalized in descriptions:
                card["description"] = descriptions[normalized]
                matched.append(card_name)
            else:
                # Try fuzzy matching for special cases
                found = False
                for wiki_name, desc in descriptions.items():
                    # Handle cases like "Mini P.E.K.K.A." vs "Mini PEKKA"
                    if normalized.replace(" ", "") == wiki_name.replace(" ", ""):
                        card["description"] = desc
                        matched.append(card_name)
                        found = True
                        break
                
                if not found:
                    unmatched.append(card_name)
    
    return cards_data, matched, unmatched


def main():
    parser = argparse.ArgumentParser(
        description="Scrape card descriptions from Clash Royale Fandom wiki",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Use a local HTML file (recommended):
    1. Open https://clashroyale.fandom.com/wiki/Card_Overviews in your browser
    2. Save the page as HTML (Cmd+S / Ctrl+S) to data/card_overviews.html
    3. Run: python scripts/scrape_card_descriptions.py --html data/card_overviews.html

  Try fetching from URL (may fail due to bot protection):
    python scripts/scrape_card_descriptions.py

  Use custom input/output paths:
    python scripts/scrape_card_descriptions.py --html data/card_overviews.html --input data/my_cards.json --output data/enriched.json
        """
    )
    
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Path to local HTML file saved from the wiki (recommended)"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/my_cards.json",
        help="Input JSON file with cards (default: data/my_cards.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/cards_with_descriptions.json",
        help="Output JSON file (default: data/cards_with_descriptions.json)"
    )
    
    parser.add_argument(
        "--descriptions-only",
        action="store_true",
        help="Only output the scraped descriptions (for debugging)"
    )
    
    args = parser.parse_args()
    
    print("Clash Royale Card Description Scraper")
    print("=" * 50)
    print()
    
    # Scrape descriptions from wiki or local file
    try:
        descriptions = scrape_card_descriptions(html_file=args.html)
        print(f"Scraped {len(descriptions)} card descriptions")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching wiki page: {e}")
        print()
        print("Tip: Save the page from your browser and use --html option:")
        print("  1. Open https://clashroyale.fandom.com/wiki/Card_Overviews")
        print("  2. Save as HTML to data/card_overviews.html")
        print("  3. Run: python scripts/scrape_card_descriptions.py --html data/card_overviews.html")
        sys.exit(1)
    
    if not descriptions:
        print("No descriptions found!")
        sys.exit(1)
    
    # If descriptions-only mode, just output the raw descriptions
    if args.descriptions_only:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(descriptions)} descriptions to {output_path}")
        return
    
    # Load cards from input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading cards from: {input_path}")
    cards_data = load_cards(input_path)
    
    total_cards = len(cards_data.get("items", [])) + len(cards_data.get("supportItems", []))
    print(f"Loaded {total_cards} cards")
    print()
    
    # Match descriptions to cards
    enriched_data, matched, unmatched = match_cards(cards_data, descriptions)
    
    print(f"Matched: {len(matched)} cards")
    print(f"Unmatched: {len(unmatched)} cards")
    
    if unmatched:
        print()
        print("Unmatched cards:")
        for name in unmatched[:20]:  # Show first 20
            print(f"  - {name}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")
    
    # Save enriched data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"Saved enriched cards to: {output_path}")
    
    # Show a sample of matched cards with descriptions
    print()
    print("Sample matched cards:")
    sample_count = 0
    for key in ["items", "supportItems"]:
        if key not in enriched_data:
            continue
        for card in enriched_data[key]:
            if "description" in card and sample_count < 3:
                desc = card["description"]
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(f"  - {card['name']}: {desc}")
                sample_count += 1


if __name__ == "__main__":
    main()
