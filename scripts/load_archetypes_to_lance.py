#!/usr/bin/env python3
"""
Load a curated archetype seed set into the Lance archetypes table.

This script resolves archetype seed data into the Lance archetypes schema,
reuses precomputed 768-dim embeddings when present, optionally generates any
missing embeddings, and inserts the resulting archetypes into LanceDB Cloud.

Environment Variables:
    LANCE_URI: LanceDB Cloud URI
    LANCE_KEY: LanceDB Cloud API key
    OPENAI_API_KEY: OpenAI API key (required unless --skip-embeddings)

Usage:
    python scripts/load_archetypes_to_lance.py
    python scripts/load_archetypes_to_lance.py --dry-run
    python scripts/load_archetypes_to_lance.py --skip-embeddings
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ARCHETYPE_EMBEDDING_MODEL = "text-embedding-3-small"
ARCHETYPE_EMBEDDING_DIMS = 768


def get_db_connection() -> Any:
    """Get a LanceDB Cloud connection using environment variables."""
    import lancedb

    uri = os.environ.get("LANCE_URI")
    api_key = os.environ.get("LANCE_KEY")

    if not uri:
        raise ValueError("LANCE_URI environment variable is required")
    if not api_key:
        raise ValueError("LANCE_KEY environment variable is required")

    return lancedb.connect(
        uri=uri,
        api_key=api_key,
        region="us-east-1",
    )


def load_cards_lookup(cards_path: Path) -> dict[str, int]:
    """Build a card-name to card-id lookup from local source data."""
    with cards_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    lookup: dict[str, int] = {}
    for key in ("items", "supportItems"):
        for card in data.get(key, []):
            name = card.get("name")
            card_id = card.get("id")
            if name and card_id is not None:
                lookup[name] = int(card_id)
    return lookup


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Generate a normalized 768-dim archetype embedding."""
    response = client.embeddings.create(
        model=ARCHETYPE_EMBEDDING_MODEL,
        input=text,
        dimensions=ARCHETYPE_EMBEDDING_DIMS,
    )
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32).tolist()


def build_text_for_embedding(archetype: dict[str, Any]) -> str:
    """Build the labeled archetype text used for embedding generation."""
    parts = []

    name = archetype.get("name", "")
    if name:
        parts.append(f"Archetype: {name}")

    description = archetype.get("description", "")
    if description:
        parts.append(f"Description: {description}")

    tags = archetype.get("tags", [])
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")

    vibes = archetype.get("playstyle_vibes", [])
    if vibes:
        parts.append(f"Playstyle vibes: {', '.join(vibes)}")

    vibe_summary = archetype.get("llm_vibe_summary", "")
    if vibe_summary:
        parts.append(f"Vibe summary: {vibe_summary}")

    return "\n".join(parts)


def resolve_example_decks(example_decks: list[dict], card_lookup: dict[str, int]) -> list[dict]:
    """Resolve example deck card names or pass through ID-based structures."""
    resolved = []
    for deck in example_decks:
        if "battle_card_ids" in deck and "tower_card_id" in deck:
            battle_card_ids = [int(card_id) for card_id in deck.get("battle_card_ids", [])]
            tower_card_id = int(deck["tower_card_id"])
            if len(battle_card_ids) != 8:
                raise ValueError(
                    f"Example deck must contain exactly 8 battle cards, got {len(battle_card_ids)}"
                )
            resolved.append(
                {
                    "battle_card_ids": battle_card_ids,
                    "tower_card_id": tower_card_id,
                }
            )
            continue

        battle_names = deck.get("battle_cards", [])
        tower_name = deck.get("tower_card")

        battle_card_ids = []
        for name in battle_names:
            if name not in card_lookup:
                raise ValueError(f"Unknown battle card name in archetype seed: {name}")
            battle_card_ids.append(card_lookup[name])

        if len(battle_card_ids) != 8:
            raise ValueError(
                f"Example deck must resolve to exactly 8 battle cards, got {len(battle_card_ids)}"
            )

        if tower_name not in card_lookup:
            raise ValueError(f"Unknown tower card name in archetype seed: {tower_name}")

        resolved.append(
            {
                "battle_card_ids": battle_card_ids,
                "tower_card_id": card_lookup[tower_name],
            }
        )

    return resolved


def transform_archetypes(
    seed_path: Path,
    cards_path: Path,
    skip_embeddings: bool,
) -> list[dict]:
    """Transform the seed JSON into Lance archetype rows."""
    with seed_path.open("r", encoding="utf-8") as f:
        seed_data = json.load(f)

    card_lookup = load_cards_lookup(cards_path)
    client: OpenAI | None = None
    api_key = os.environ.get("OPENAI_API_KEY")

    transformed = []
    for archetype in seed_data:
        description = archetype.get("description", "")
        tags = archetype.get("tags", [])
        vibes = archetype.get("playstyle_vibes", [])
        vibe_summary = archetype.get("llm_vibe_summary")

        embedding = archetype.get("embedding")
        if embedding is None and not skip_embeddings:
            if client is None:
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required when embeddings are missing and --skip-embeddings is not set"
                    )
                client = OpenAI(api_key=api_key)
            embedding_text = build_text_for_embedding(archetype)
            embedding = get_embedding(client, embedding_text)

        transformed.append(
            {
                "id": archetype["id"],
                "name": archetype["name"],
                "description": description,
                "example_decks": resolve_example_decks(
                    archetype.get("example_decks", []),
                    card_lookup,
                ),
                "embedding": embedding,
                "tags": tags,
                "meta_strength": archetype.get("meta_strength"),
                "playstyle_vibes": vibes,
                "llm_vibe_summary": vibe_summary,
            }
        )

    return transformed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load curated archetype seed data into LanceDB",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/archetypes_seed.json",
        help="Path to the archetype seed JSON",
    )
    parser.add_argument(
        "--cards-source",
        default="data/cards_with_descriptions.json",
        help="Path to card source JSON used to resolve names to IDs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Transform data but do not insert anything",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip OpenAI embedding generation and only use embeddings already present in the input JSON",
    )
    args = parser.parse_args()

    seed_path = Path(args.input)
    cards_path = Path(args.cards_source)

    if not seed_path.exists():
        raise SystemExit(f"Seed file not found: {seed_path}")
    if not cards_path.exists():
        raise SystemExit(f"Cards source file not found: {cards_path}")

    archetypes = transform_archetypes(
        seed_path=seed_path,
        cards_path=cards_path,
        skip_embeddings=args.skip_embeddings,
    )

    print(f"Prepared {len(archetypes)} archetypes from {seed_path}")
    for archetype in archetypes:
        print(f"  - {archetype['id']}: {len(archetype['example_decks'])} example deck(s)")

    if args.dry_run:
        print("Dry run complete. No data inserted.")
        return

    db = get_db_connection()
    if "archetypes" not in db.table_names():
        raise SystemExit("The 'archetypes' table does not exist. Run create_tables.py first.")

    table = db.open_table("archetypes")
    table.add(archetypes)
    print(f"Inserted {len(archetypes)} archetype rows into LanceDB.")


if __name__ == "__main__":
    main()
