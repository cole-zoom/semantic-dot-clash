#!/usr/bin/env python3
"""
Patch script to update specific card vibe data in the Lance database.

This script updates llm_vibe_summary, role_tags, vibe_tags, crowd_ratings,
and regenerates text_embedding and combined_embedding for specific cards.

Environment Variables:
    LANCE_URI: LanceDB Cloud URI
    LANCE_KEY: LanceDB Cloud API key
    OPENAI_API_KEY: OpenAI API key (for generating missing data and embeddings)

Usage:
    python scripts/patch_card_vibes.py
    python scripts/patch_card_vibes.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

import lancedb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Embedding model settings
TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
TEXT_EMBEDDING_DIMS = 1536
IMAGE_EMBEDDING_DIMS = 512

# Patches to apply
PATCHES = {
    "Dagger Duchess": {
        "llm_vibe_summary": """The community vibe around Dagger Duchess is polarizing and salty.
A lot of players think she's oppressive on defense, especially against cycle decks, and makes ladder feel slower and more rigid.
Others argue she's overhated, saying people just haven't adapted and that she adds strategic depth.
Net result: she's constantly complained about, constantly defended, and always talked about.""",
    },
    "Royal Chef": {
        "llm_vibe_summary": """Royal Chef's community vibe is mixed and chaotic — some players love the novel mechanic of leveling up troops, while others think it doesn't fit well with the game or feels underwhelming for a Legendary card.
Many folks complain about bugs, awkward interactions, and balance issues, turning discussion threads into a blend of frustration and memes.
Others find it situationally fun in beatdown decks but agree it can be weak or inconsistent compared to Princess Tower.
Overall, the sentiment is "cool concept, messy execution," with players either trying to make him work or wishing he'd be reworked or reclassified.""",
        # These will be filled in by LLM
        "role_tags": None,
        "vibe_tags": None,
        "crowd_ratings": None,
    },
}

# JSON schema for LLM to generate Royal Chef data
VIBE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "card_vibe_tags",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "role_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tactical/functional roles the card fulfills"
                },
                "vibe_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Community perception/feeling tags"
                },
                "crowd_ratings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "vibe": {"type": "string"},
                            "score": {"type": "number"}
                        },
                        "required": ["vibe", "score"],
                        "additionalProperties": False
                    },
                    "description": "Quantified intensity scores (0.0-1.0) for relevant vibes"
                }
            },
            "required": ["role_tags", "vibe_tags", "crowd_ratings"],
            "additionalProperties": False
        }
    }
}


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


def generate_royal_chef_tags(client: OpenAI) -> dict:
    """Generate role_tags, vibe_tags, and crowd_ratings for Royal Chef using OpenAI."""
    
    system_prompt = '''You are analyzing player sentiment for Clash Royale cards.

Given a card's community vibe summary, generate structured tags and ratings.

Role tags should be selected from:
"win condition", "secondary win condition", "cycle", "splash", "tank", "mini tank", "anti-air", "building", "spell", "swarm", "DPS", "support", "bridge spam", "bait", "control", "beatdown support", "tank killer", "distraction", "kiting", "spell bait"

Vibe tags should be selected from:
"toxic", "annoying", "spammy", "no-skill", "brain-dead", "respectable", "high-skill", "wholesome", "fun", "tryhard", "troll", "satisfying", "frustrating to face", "underrated", "overrated", "timeless", "gimmicky", "matchup dependent", "coin-flip", "mid-ladder menace", "top-ladder viable", "dead card", "sleeper", "cheese"

Crowd ratings should reflect intensity (0.0-1.0) for the most relevant vibes.'''

    user_prompt = f'''Card: Royal Chef (Tower Troop, Legendary)

Description: Royal Chef is a tower troop that upgrades other troops that pass by him, giving them stat boosts.

Community Vibe Summary:
{PATCHES["Royal Chef"]["llm_vibe_summary"]}

Generate the role_tags, vibe_tags, and crowd_ratings for this card.'''

    print("Generating Royal Chef tags via OpenAI...")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=VIBE_SCHEMA,
        temperature=0.7,
    )
    
    result = json.loads(response.choices[0].message.content)
    print(f"  role_tags: {result['role_tags']}")
    print(f"  vibe_tags: {result['vibe_tags']}")
    print(f"  crowd_ratings: {result['crowd_ratings']}")
    
    return result


def convert_crowd_ratings(ratings: list[dict]) -> list[dict]:
    """Convert crowd_ratings to Lance schema format."""
    return [{"key": r["vibe"], "value": float(r["score"])} for r in ratings]


def build_text_for_embedding(card: dict, patch_data: dict | None = None) -> str:
    """
    Build a rich text representation of a card for embedding.
    
    Uses patch_data to override card fields if provided.
    """
    parts = []
    
    # Card name
    name = card.get("name", "")
    parts.append(f"Card: {name}")
    
    # Rarity
    rarity = card.get("rarity", "")
    if rarity:
        parts.append(f"Rarity: {rarity}")
    
    # Elixir cost
    elixir = card.get("elixir")
    if elixir is not None:
        parts.append(f"Elixir cost: {elixir}")
    
    # Description
    description = card.get("description", "")
    if description:
        parts.append(f"Description: {description}")
    
    # Role tags (use patch if available)
    role_tags = patch_data.get("role_tags") if patch_data else None
    if role_tags is None:
        role_tags = card.get("role_tags", [])
    if role_tags:
        # Handle Lance struct format
        if role_tags and isinstance(role_tags[0], dict):
            role_tags = [r.get("key", r) for r in role_tags]
        parts.append(f"Roles: {', '.join(role_tags)}")
    
    # Vibe tags (use patch if available)
    vibe_tags = patch_data.get("vibe_tags") if patch_data else None
    if vibe_tags is None:
        vibe_tags = card.get("vibe_tags", [])
    if vibe_tags:
        if vibe_tags and isinstance(vibe_tags[0], dict):
            vibe_tags = [v.get("key", v) for v in vibe_tags]
        parts.append(f"Vibes: {', '.join(vibe_tags)}")
    
    # LLM vibe summary (use patch if available)
    vibe_summary = patch_data.get("llm_vibe_summary") if patch_data else None
    if vibe_summary is None:
        vibe_summary = card.get("llm_vibe_summary", "")
    if vibe_summary:
        parts.append(f"Community perception: {vibe_summary}")
    
    return "\n".join(parts)


def get_text_embedding(client: OpenAI, text: str) -> list[float]:
    """Generate OpenAI text embedding."""
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text,
        dimensions=TEXT_EMBEDDING_DIMS,
    )
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def create_combined_embedding(
    text_embedding: list[float],
    image_embedding: list[float] | None,
) -> list[float]:
    """
    Create a combined embedding via normalized concatenation.
    
    If image embedding is missing, pads with zeros.
    Result is normalized to unit length.
    """
    text_arr = np.array(text_embedding, dtype=np.float32)
    
    if image_embedding is None:
        image_arr = np.zeros(IMAGE_EMBEDDING_DIMS, dtype=np.float32)
    else:
        image_arr = np.array(image_embedding, dtype=np.float32)
    
    # Concatenate
    combined = np.concatenate([text_arr, image_arr])
    
    # Normalize to unit length
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Patch specific card vibe data in Lance database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Apply patches to database:
    python scripts/patch_card_vibes.py

  Dry run (show what would be updated):
    python scripts/patch_card_vibes.py --dry-run
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    
    args = parser.parse_args()
    
    print("Card Vibe Patch Script")
    print("=" * 50)
    print()
    
    # Check for OpenAI API key (needed for Royal Chef)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Generate Royal Chef tags
    client = OpenAI(api_key=openai_key)
    royal_chef_data = generate_royal_chef_tags(client)
    
    # Update PATCHES with generated data
    PATCHES["Royal Chef"]["role_tags"] = royal_chef_data["role_tags"]
    PATCHES["Royal Chef"]["vibe_tags"] = royal_chef_data["vibe_tags"]
    PATCHES["Royal Chef"]["crowd_ratings"] = convert_crowd_ratings(royal_chef_data["crowd_ratings"])
    
    print()
    
    # Show patches to apply
    print("Patches to apply:")
    print("-" * 50)
    for card_name, patch_data in PATCHES.items():
        print(f"\n{card_name}:")
        for field, value in patch_data.items():
            if field == "llm_vibe_summary":
                preview = value[:80] + "..." if len(value) > 80 else value
                print(f"  {field}: {preview}")
            else:
                print(f"  {field}: {value}")
    
    print()
    
    if args.dry_run:
        print("Dry run complete. No changes made.")
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
        print("Error: 'cards' table does not exist.")
        sys.exit(1)
    
    table = db.open_table("cards")
    print(f"  Table has {table.count_rows()} cards")
    print()
    
    # Fetch existing card data to get image embeddings and other fields
    print("Fetching existing card data...")
    existing_cards = {}
    for card_name in PATCHES.keys():
        try:
            result = table.search().where(f"name = '{card_name}'").limit(1).to_list()
            if result:
                existing_cards[card_name] = result[0]
                print(f"  Found: {card_name}")
            else:
                print(f"  Not found: {card_name}")
        except Exception as e:
            print(f"  Error fetching {card_name}: {e}")
    
    print()
    
    # Generate new embeddings for each card
    print("Regenerating embeddings...")
    
    for card_name, patch_data in PATCHES.items():
        if card_name not in existing_cards:
            print(f"  Skipping {card_name} (not found in database)")
            continue
        
        existing_card = existing_cards[card_name]
        print(f"\n  {card_name}:")
        
        # Build text for embedding using patched data
        text = build_text_for_embedding(existing_card, patch_data)
        print(f"    Text length: {len(text)} chars")
        
        # Generate new text embedding
        print(f"    Generating text embedding...")
        text_embedding = get_text_embedding(client, text)
        patch_data["text_embedding"] = text_embedding
        print(f"    Text embedding: {len(text_embedding)} dims")
        
        # Get existing image embedding
        image_embedding = existing_card.get("image_embedding")
        if image_embedding is not None:
            print(f"    Existing image embedding: {len(image_embedding)} dims")
        else:
            print(f"    No existing image embedding (will pad with zeros)")
        
        # Create combined embedding
        combined_embedding = create_combined_embedding(text_embedding, image_embedding)
        patch_data["combined_embedding"] = combined_embedding
        print(f"    Combined embedding: {len(combined_embedding)} dims")
    
    print()
    
    # Apply patches using update
    print("Applying patches...")
    
    for card_name, patch_data in PATCHES.items():
        print(f"\nUpdating {card_name}...")
        
        # Build the update values
        update_values = {}
        for field, value in patch_data.items():
            if value is not None:
                update_values[field] = value
        
        try:
            # Use LanceDB update with SQL filter
            table.update(
                where=f"name = '{card_name}'",
                values=update_values
            )
            print(f"  ✓ Updated {card_name} ({len(update_values)} fields)")
        except Exception as e:
            print(f"  ✗ Failed to update {card_name}: {e}")
    
    print()
    print("=" * 50)
    print("Patch complete!")
    
    # Verify updates
    print()
    print("Verifying updates...")
    for card_name in PATCHES.keys():
        try:
            result = table.search().where(f"name = '{card_name}'").limit(1).to_list()
            if result:
                card = result[0]
                print(f"\n{card_name}:")
                print(f"  llm_vibe_summary: {card.get('llm_vibe_summary', '')[:60]}...")
                print(f"  role_tags: {card.get('role_tags', [])}")
                print(f"  vibe_tags: {card.get('vibe_tags', [])}")
                print(f"  crowd_ratings: {card.get('crowd_ratings', [])}")
                text_emb = card.get('text_embedding')
                img_emb = card.get('image_embedding')
                combined_emb = card.get('combined_embedding')
                print(f"  text_embedding: {len(text_emb) if text_emb else 0} dims")
                print(f"  image_embedding: {len(img_emb) if img_emb else 0} dims")
                print(f"  combined_embedding: {len(combined_emb) if combined_emb else 0} dims")
            else:
                print(f"\n{card_name}: Not found in database")
        except Exception as e:
            print(f"\n{card_name}: Error verifying - {e}")


if __name__ == "__main__":
    main()
