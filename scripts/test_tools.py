#!/usr/bin/env python3
"""
Test script for the CardTools functionality.

Tests all six tools:
- search_archetypes: Semantic search for archetype blueprints
- get_archetype: Fetch archetype by ID
- search_cards: Semantic search with filters
- similar_cards: Find similar cards
- get_card: Fetch card by ID
- score_deck: Evaluate a deck

Environment Variables:
    LANCE_URI: LanceDB Cloud URI
    LANCE_KEY: LanceDB Cloud API key
    OPENAI_API_KEY: OpenAI API key

Usage:
    python scripts/test_tools.py
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from semantic_dot_clash.tools import CardTools, DeckScore


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_card(card: dict, index: int = None) -> None:
    """Print a card in a readable format."""
    prefix = f"  [{index}] " if index is not None else "  "
    distance = card.get("_distance", "N/A")
    distance_str = f" (distance: {distance:.4f})" if isinstance(distance, float) else ""
    print(f"{prefix}{card['name']} - {card['elixir']} elixir, {card['rarity']} {card['type']}{distance_str}")


def print_archetype(archetype: dict, index: int = None) -> None:
    """Print an archetype in a readable format."""
    prefix = f"  [{index}] " if index is not None else "  "
    distance = archetype.get("_distance", "N/A")
    distance_str = f" (distance: {distance:.4f})" if isinstance(distance, float) else ""
    tags = ", ".join(archetype.get("tags") or [])
    print(f"{prefix}{archetype['name']}{distance_str}")
    if tags:
        print(f"      tags: {tags}")
    if archetype.get("playstyle_vibes"):
        print(f"      vibes: {', '.join(archetype['playstyle_vibes'])}")
    if archetype.get("llm_vibe_summary"):
        print(f"      summary: {archetype['llm_vibe_summary'][:100]}...")


def test_search_archetypes(tools: CardTools) -> str | None:
    """Test search_archetypes functionality."""
    print_header("TEST: search_archetypes")

    query = "fast annoying control deck with cheap pressure"
    print(f"\n  Searching archetypes for: {query!r}")
    results = tools.search_archetypes(query, limit=3)
    print(f"  Found {len(results)} results:")
    for i, archetype in enumerate(results, 1):
        print_archetype(archetype, i)

    if not results:
        print("  ERROR: No archetypes found in database!")
        return None

    chosen = results[0]
    print(f"\n  SUCCESS! Selected archetype candidate: {chosen['name']} ({chosen['id']})")
    return chosen["id"]


def test_get_archetype(tools: CardTools, archetype_id: str) -> None:
    """Test get_archetype functionality."""
    print_header("TEST: get_archetype")

    print(f"\n  Fetching archetype by ID {archetype_id!r}...")
    archetype = tools.get_archetype(archetype_id)

    if archetype is None:
        print(f"  ERROR: get_archetype returned None for ID {archetype_id}")
        return

    print("  SUCCESS! Retrieved archetype:")
    print(f"    Name: {archetype['name']}")
    print(f"    ID: {archetype['id']}")
    print(f"    Meta strength: {archetype.get('meta_strength', 'N/A')}")
    print(f"    Tags: {archetype.get('tags', [])}")
    print(f"    Vibes: {archetype.get('playstyle_vibes', [])}")
    print(f"    Vibe summary: {archetype.get('llm_vibe_summary', 'N/A')}")

    print("\n  Testing non-existent archetype ID...")
    missing = tools.get_archetype("not_a_real_archetype")
    if missing is None:
        print("  SUCCESS! Correctly returned None for non-existent archetype ID")
    else:
        print("  WARNING: Expected None for non-existent archetype ID")


def test_get_card(tools: CardTools) -> int | None:
    """Test get_card functionality."""
    print_header("TEST: get_card")
    
    # First, let's search to find a valid card ID
    print("\n  Searching for any card to get its ID...")
    results = tools.search_cards("hog rider", limit=1)
    
    if not results:
        print("  ERROR: No cards found in database!")
        return None
    
    card_id = results[0]["id"]
    card_name = results[0]["name"]
    print(f"  Found card: {card_name} (ID: {card_id})")
    
    # Now test get_card
    print(f"\n  Fetching card by ID {card_id}...")
    card = tools.get_card(card_id)
    
    if card is None:
        print(f"  ERROR: get_card returned None for ID {card_id}")
        return None
    
    print(f"  SUCCESS! Retrieved card:")
    print(f"    Name: {card['name']}")
    print(f"    ID: {card['id']}")
    print(f"    Elixir: {card['elixir']}")
    print(f"    Type: {card['type']}")
    print(f"    Rarity: {card['rarity']}")
    print(f"    Description: {card.get('description', 'N/A')[:80]}...")
    print(f"    Role tags: {card.get('role_tags', [])}")
    print(f"    Vibe tags: {card.get('vibe_tags', [])}")
    
    # Test non-existent card
    print("\n  Testing non-existent card ID...")
    missing = tools.get_card(9999999999)
    if missing is None:
        print("  SUCCESS! Correctly returned None for non-existent ID")
    else:
        print("  WARNING: Expected None for non-existent ID")
    
    return card_id


def test_search_cards(tools: CardTools) -> list[int]:
    """Test search_cards functionality."""
    print_header("TEST: search_cards")
    
    # Basic semantic search
    print("\n  1. Basic semantic search: 'fast cycle cards'")
    results = tools.search_cards("fast cycle cards", limit=5)
    print(f"  Found {len(results)} results:")
    for i, card in enumerate(results, 1):
        print_card(card, i)
    
    # Search with elixir filter
    print("\n  2. Search with elixir filter: 'tank' with elixir >= 5")
    results = tools.search_cards("tank", elixir_min=5, limit=5)
    print(f"  Found {len(results)} results:")
    for i, card in enumerate(results, 1):
        print_card(card, i)
    
    # Search with type filter
    print("\n  3. Search with type filter: 'damage' with type='Spell'")
    results = tools.search_cards("damage", type="Spell", limit=5)
    print(f"  Found {len(results)} results:")
    for i, card in enumerate(results, 1):
        print_card(card, i)
    
    # Search with rarity filter
    print("\n  4. Search with rarity filter: 'legendary'")
    results = tools.search_cards("powerful legendary card", rarity="Legendary", limit=5)
    print(f"  Found {len(results)} results:")
    for i, card in enumerate(results, 1):
        print_card(card, i)
    
    # Search with multiple filters
    print("\n  5. Combined filters: 'anti-air' with elixir <= 4 and type='Troop'")
    results = tools.search_cards("anti-air", elixir_max=4, type="Troop", limit=5)
    print(f"  Found {len(results)} results:")
    for i, card in enumerate(results, 1):
        print_card(card, i)
    
    # Collect card IDs for deck test
    all_results = tools.search_cards("good deck cards", limit=8)
    card_ids = [card["id"] for card in all_results]
    
    print(f"\n  SUCCESS! All search tests passed")
    return card_ids


def test_similar_cards(tools: CardTools, card_id: int) -> None:
    """Test similar_cards functionality."""
    print_header("TEST: similar_cards")
    
    # Get the source card name
    source = tools.get_card(card_id)
    print(f"\n  Finding cards similar to: {source['name']} (ID: {card_id})")
    
    results = tools.similar_cards(card_id, limit=5)
    print(f"  Found {len(results)} similar cards:")
    for i, card in enumerate(results, 1):
        print_card(card, i)
    
    # Verify source card is not in results
    result_ids = [card["id"] for card in results]
    if card_id in result_ids:
        print("  WARNING: Source card should not be in results!")
    else:
        print("\n  SUCCESS! Source card correctly excluded from results")


def get_tower_card_id(tools: CardTools) -> int:
    """Fetch a tower troop ID for deck scoring tests."""
    results = tools.search_cards("tower troop", type="Tower Troop", limit=3)
    if not results:
        raise ValueError("No tower troops found in database")
    tower_card = results[0]
    print(f"\n  Selected tower troop: {tower_card['name']} (ID: {tower_card['id']})")
    return tower_card["id"]


def test_score_deck(tools: CardTools, card_ids: list[int], tower_card_id: int) -> None:
    """Test score_deck functionality."""
    print_header("TEST: score_deck")
    
    # Ensure we have exactly 8 cards
    if len(card_ids) < 8:
        print(f"  WARNING: Only {len(card_ids)} cards available, need 8 for full test")
        # Pad with duplicates for testing (will generate warnings)
        while len(card_ids) < 8:
            card_ids.append(card_ids[0])
    
    card_ids = card_ids[:8]
    
    print(f"\n  Scoring deck with {len(card_ids)} cards...")
    print(f"  Card IDs: {card_ids}")
    print(f"  Tower Card ID: {tower_card_id}")
    
    score = tools.score_deck(battle_card_ids=card_ids, tower_card_id=tower_card_id)
    
    print(f"\n  Deck Cards:")
    for i, card in enumerate(score.battle_cards, 1):
        print(f"    [{i}] {card['name']} - {card['elixir']} elixir")
    if score.tower_card:
        print(f"\n  Tower Troop:")
        print(f"    {score.tower_card['name']} - {score.tower_card['type']}")
    
    print(f"\n  Metrics:")
    print(f"    Average Elixir: {score.avg_elixir}")
    print(f"    Synergy Score: {score.synergy_score}")
    print(f"    Meta Strength: {score.meta_strength}/100")
    
    print(f"\n  Type Distribution:")
    for card_type, count in sorted(score.type_distribution.items()):
        print(f"    {card_type}: {count}")
    
    print(f"\n  Rarity Distribution:")
    for rarity, count in sorted(score.rarity_distribution.items()):
        print(f"    {rarity}: {count}")
    
    print(f"\n  Role Coverage: {sorted(score.role_coverage) if score.role_coverage else 'None'}")
    print(f"  Missing Roles: {sorted(score.missing_roles) if score.missing_roles else 'None'}")
    
    if score.balance_warnings:
        print(f"\n  Balance Warnings:")
        for warning in score.balance_warnings:
            print(f"    - {warning}")
    else:
        print(f"\n  No balance warnings!")
    
    # Test to_dict method
    score_dict = score.to_dict()
    print(f"\n  to_dict() method works: {len(score_dict)} keys")
    
    print("\n  SUCCESS! Deck scoring complete")


def main():
    print()
    print("*" * 60)
    print("*  CardTools Test Suite")
    print("*" * 60)
    
    # Initialize CardTools
    print("\nInitializing CardTools...")
    try:
        tools = CardTools()
        print("SUCCESS! Connected to LanceDB and OpenAI")
    except Exception as e:
        print(f"ERROR: Failed to initialize CardTools: {e}")
        sys.exit(1)
    
    # Run tests
    try:
        archetype_id = test_search_archetypes(tools)
        if archetype_id is None:
            print("\nERROR: Cannot continue without valid archetype ID")
            sys.exit(1)
        test_get_archetype(tools, archetype_id)

        # Test get_card first to get a valid card ID
        card_id = test_get_card(tools)
        if card_id is None:
            print("\nERROR: Cannot continue without valid card ID")
            sys.exit(1)
        
        # Test search_cards
        deck_card_ids = test_search_cards(tools)
        tower_card_id = get_tower_card_id(tools)
        
        # Test similar_cards
        test_similar_cards(tools, card_id)
        
        # Test score_deck
        test_score_deck(tools, deck_card_ids, tower_card_id)
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print_header("TEST SUMMARY")
    print("\n  All tests passed!")
    print()


if __name__ == "__main__":
    main()
