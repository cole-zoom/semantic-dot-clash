#!/usr/bin/env python3
"""
Script to load cards with embeddings into the Lance database cards table.

Transforms the cards_with_embeddings.json data to match the Lance schema
and inserts it into the cards table.

Environment Variables:
    LANCE_URI: LanceDB Cloud URI
    LANCE_KEY: LanceDB Cloud API key

Usage:
    python scripts/load_cards_to_lance.py
    python scripts/load_cards_to_lance.py --input data/cards_with_embeddings.json
    python scripts/load_cards_to_lance.py --dry-run
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import lancedb
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# Known card type classifications
# Based on Clash Royale card types: Troop, Spell, Building
SPELL_CARDS = {
    "Arrows", "Fireball", "Rocket", "Zap", "Lightning", "Freeze", "Poison",
    "Tornado", "Earthquake", "Giant Snowball", "Barbarian Barrel", "The Log",
    "Mirror", "Rage", "Clone", "Graveyard", "Goblin Barrel", "Royal Delivery",
    "Void", "Goblin Curse", "Vines",  # New spells
}

BUILDING_CARDS = {
    "Cannon", "Tesla", "Inferno Tower", "Bomb Tower", "Mortar", "X-Bow",
    "Goblin Cage", "Goblin Drill", "Tombstone", "Furnace", "Goblin Hut",
    "Barbarian Hut", "Elixir Collector",
}

# Champion cards (special type)
CHAMPION_CARDS = {
    "Archer Queen", "Golden Knight", "Skeleton King", "Mighty Miner",
    "Monk", "Little Prince",
}

# Tower troops (support items that act on the tower)
TOWER_TROOP_CARDS = {
    "Tower Princess", "Cannoneer", "Dagger Duchess", "Royal Chef",
}


def get_card_type(card: dict) -> str:
    """
    Determine the card type (Troop, Spell, Building, Champion, Tower Troop).
    
    Args:
        card: Card dictionary from JSON
        
    Returns:
        Card type string
    """
    name = card.get("name", "")
    rarity = card.get("rarity", "").lower()
    
    # Check if it's a champion first
    if rarity == "champion" or name in CHAMPION_CARDS:
        return "Champion"
    
    # Check tower troops
    if name in TOWER_TROOP_CARDS:
        return "Tower Troop"
    
    # Check known spells
    if name in SPELL_CARDS:
        return "Spell"
    
    # Check known buildings
    if name in BUILDING_CARDS:
        return "Building"
    
    # Default to Troop (most cards are troops)
    return "Troop"


def download_image(url: str, timeout: int = 30) -> bytes | None:
    """Download an image and return as bytes."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"    Warning: Failed to download image: {e}")
        return None


def convert_crowd_ratings(ratings: dict | None) -> list[dict] | None:
    """
    Convert crowd_ratings from dict to list of structs format.
    
    Schema expects: list of {"key": string, "value": float}
    Input format: {"toxic": 0.73, "annoying": 0.55}
    """
    if not ratings:
        return None
    
    return [{"key": k, "value": float(v)} for k, v in ratings.items()]


def transform_card(card: dict, index: int, total: int, download_images: bool = True) -> dict:
    """
    Transform a card from JSON format to Lance schema format.
    
    Args:
        card: Card dictionary from JSON
        index: Current card index for progress display
        total: Total number of cards
        download_images: Whether to download card images
        
    Returns:
        Transformed card dictionary matching Lance schema
    """
    name = card.get("name", "Unknown")
    print(f"[{index + 1}/{total}] Transforming: {name}...", flush=True)
    
    # Get image bytes if URL exists and download is enabled
    image_bytes = None
    if download_images:
        image_url = card.get("iconUrls", {}).get("medium")
        if image_url:
            image_bytes = download_image(image_url)
    
    # Get embeddings as float32 numpy arrays
    text_embedding = card.get("text_embedding")
    if text_embedding:
        text_embedding = np.array(text_embedding, dtype=np.float32).tolist()
    
    image_embedding = card.get("image_embedding")
    if image_embedding:
        image_embedding = np.array(image_embedding, dtype=np.float32).tolist()
    
    combined_embedding = card.get("combined_embedding")
    if combined_embedding:
        combined_embedding = np.array(combined_embedding, dtype=np.float32).tolist()
    
    # Transform to Lance schema
    transformed = {
        "id": int(card.get("id", 0)),
        "name": name,
        "rarity": card.get("rarity", "unknown"),
        "type": get_card_type(card),
        "elixir": int(card.get("elixirCost", 0)),
        "description": card.get("description"),
        "image": image_bytes,
        "image_embedding": image_embedding,
        "text_embedding": text_embedding,
        "combined_embedding": combined_embedding,
        "role_tags": card.get("role_tags"),
        "vibe_tags": card.get("vibe_tags"),
        "crowd_ratings": convert_crowd_ratings(card.get("crowd_ratings")),
        "llm_vibe_summary": card.get("llm_vibe_summary"),
    }
    
    return transformed


def load_cards(input_path: str) -> list[dict]:
    """Load cards from JSON file and flatten items + supportItems."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_cards = []
    
    # Add items (regular cards)
    if "items" in data:
        all_cards.extend(data["items"])
    
    # Add supportItems if present
    if "supportItems" in data:
        all_cards.extend(data["supportItems"])
    
    return all_cards


def get_db_connection() -> lancedb.DBConnection:
    """Get LanceDB connection using environment variables."""
    uri = os.environ.get("LANCE_URI")
    api_key = os.environ.get("LANCE_KEY")
    
    if not uri:
        raise ValueError("LANCE_URI environment variable is required")
    if not api_key:
        raise ValueError("LANCE_KEY environment variable is required")
    
    return lancedb.connect(
        uri=uri,
        api_key=api_key,
        region="us-east-1"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load cards with embeddings into Lance database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Load cards into Lance database:
    python scripts/load_cards_to_lance.py

  Use custom input file:
    python scripts/load_cards_to_lance.py --input data/cards_with_embeddings.json

  Dry run (transform but don't insert):
    python scripts/load_cards_to_lance.py --dry-run

  Skip image downloads (faster, but no image data):
    python scripts/load_cards_to_lance.py --no-images
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/cards_with_embeddings.json",
        help="Input JSON file with cards and embeddings (default: data/cards_with_embeddings.json)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Transform data but don't insert into database"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip downloading card images (faster)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of cards to insert per batch (default: 25)"
    )
    
    args = parser.parse_args()
    
    print("Clash Royale Cards - Lance Database Loader")
    print("=" * 50)
    print()
    
    # Load cards from JSON
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading cards from: {input_path}")
    cards = load_cards(input_path)
    print(f"Found {len(cards)} cards")
    print()
    
    # Transform cards
    print("Transforming cards to Lance schema...")
    print()
    
    transformed_cards = []
    for i, card in enumerate(cards):
        try:
            transformed = transform_card(
                card, i, len(cards),
                download_images=not args.no_images
            )
            transformed_cards.append(transformed)
        except Exception as e:
            print(f"  Error transforming {card.get('name', 'Unknown')}: {e}")
    
    print()
    print(f"Successfully transformed {len(transformed_cards)} cards")
    print()
    
    # Show sample transformation
    if transformed_cards:
        sample = transformed_cards[0]
        print("Sample transformed card:")
        print(f"  id: {sample['id']}")
        print(f"  name: {sample['name']}")
        print(f"  rarity: {sample['rarity']}")
        print(f"  type: {sample['type']}")
        print(f"  elixir: {sample['elixir']}")
        print(f"  description: {sample['description'][:50]}..." if sample['description'] else "  description: None")
        print(f"  image: {len(sample['image']) if sample['image'] else 0} bytes")
        print(f"  image_embedding: {len(sample['image_embedding']) if sample['image_embedding'] else 0} dims")
        print(f"  text_embedding: {len(sample['text_embedding']) if sample['text_embedding'] else 0} dims")
        print(f"  combined_embedding: {len(sample['combined_embedding']) if sample['combined_embedding'] else 0} dims")
        print(f"  role_tags: {sample['role_tags']}")
        print(f"  vibe_tags: {sample['vibe_tags']}")
        print(f"  crowd_ratings: {sample['crowd_ratings']}")
        print()
    
    # Dry run - stop here
    if args.dry_run:
        print("Dry run complete. No data inserted.")
        return
    
    # Connect to Lance database
    print("Connecting to Lance database...")
    try:
        db = get_db_connection()
        print(f"  Connected to: {os.environ.get('LANCE_URI', '')}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Check if cards table exists
    if "cards" not in db.table_names():
        print("Error: 'cards' table does not exist. Run create_tables.py first.")
        sys.exit(1)
    
    table = db.open_table("cards")
    initial_count = table.count_rows()
    print(f"  Current card count: {initial_count}")
    print()
    
    # Insert cards in batches
    print(f"Inserting {len(transformed_cards)} cards (batch size: {args.batch_size})...")
    
    inserted = 0
    for i in range(0, len(transformed_cards), args.batch_size):
        batch = transformed_cards[i:i + args.batch_size]
        try:
            table.add(batch)
            inserted += len(batch)
            print(f"  Inserted batch {i // args.batch_size + 1}: {len(batch)} cards (total: {inserted})")
        except Exception as e:
            print(f"  Error inserting batch: {e}")
            # Try inserting one by one to identify problematic cards
            for card in batch:
                try:
                    table.add([card])
                    inserted += 1
                except Exception as card_error:
                    print(f"    Failed to insert {card['name']}: {card_error}")
    
    print()
    final_count = table.count_rows()
    print("=" * 50)
    print(f"Done! Inserted {inserted} cards")
    print(f"Table now has {final_count} cards")
    print()
    
    # Show type distribution
    type_counts = {}
    for card in transformed_cards:
        card_type = card["type"]
        type_counts[card_type] = type_counts.get(card_type, 0) + 1
    
    print("Card type distribution:")
    for card_type, count in sorted(type_counts.items()):
        print(f"  {card_type}: {count}")


if __name__ == "__main__":
    main()
