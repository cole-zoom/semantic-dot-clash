#!/usr/bin/env python3
"""
Add explicit hero/evolution fields to cards_with_vibes.json.

This script enriches an existing cards JSON by copying hero/evolution signals
from a source cards file and writing the following top-level fields per card:
- is_hero
- hero_url
- is_evo
- evo_url

Usage:
    python scripts/add_hero_evo_fields.py
    python scripts/add_hero_evo_fields.py --source data/my_cards.json
    python scripts/add_hero_evo_fields.py --input data/cards_with_vibes.json --output data/cards_with_vibes_enriched.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CARD_SECTIONS = ("items", "supportItems")


def load_cards(path: Path) -> dict:
    """Load a card JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_cards(cards_data: dict):
    """Yield all cards from supported top-level sections."""
    for section in CARD_SECTIONS:
        for card in cards_data.get(section, []):
            yield section, card


def build_source_lookup(cards_data: dict) -> dict[int, dict]:
    """Build an ID-based lookup for source cards."""
    lookup: dict[int, dict] = {}
    for _, card in iter_cards(cards_data):
        card_id = card.get("id")
        if card_id is None:
            continue
        lookup[int(card_id)] = card
    return lookup


def enrich_target_cards(source_data: dict, target_data: dict) -> tuple[dict, int, list[str]]:
    """Copy hero/evo fields from source cards onto target cards."""
    source_lookup = build_source_lookup(source_data)
    matched_count = 0
    unmatched_cards: list[str] = []

    for _, target_card in iter_cards(target_data):
        card_id = target_card.get("id")
        source_card = source_lookup.get(int(card_id)) if card_id is not None else None

        if source_card is None:
            unmatched_cards.append(target_card.get("name", f"Unknown ({card_id})"))
            continue

        icon_urls = source_card.get("iconUrls") or {}
        hero_url = icon_urls.get("heroMedium")
        evo_url = icon_urls.get("evolutionMedium")

        target_card["is_hero"] = bool(hero_url)
        target_card["hero_url"] = hero_url
        target_card["is_evo"] = bool(evo_url)
        target_card["evo_url"] = evo_url
        matched_count += 1

    return target_data, matched_count, unmatched_cards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add explicit hero/evolution fields to cards_with_vibes.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Update cards_with_vibes.json in place:
    python scripts/add_hero_evo_fields.py

  Use a custom source file:
    python scripts/add_hero_evo_fields.py --source data/my_cards.json

  Write to a new output file:
    python scripts/add_hero_evo_fields.py --output data/cards_with_vibes_enriched.json
        """,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/my_cards.json",
        help="Source JSON with hero/evolution iconUrls (default: data/my_cards.json)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/cards_with_vibes.json",
        help="Target JSON to enrich (default: data/cards_with_vibes.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/cards_with_vibes.json",
        help="Output JSON path (default: data/cards_with_vibes.json)",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}")
        sys.exit(1)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("Add Hero/Evo Fields")
    print("=" * 50)
    print(f"Source: {source_path}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    source_data = load_cards(source_path)
    target_data = load_cards(input_path)

    source_count = sum(len(source_data.get(section, [])) for section in CARD_SECTIONS)
    target_count = sum(len(target_data.get(section, [])) for section in CARD_SECTIONS)

    enriched_data, matched_count, unmatched_cards = enrich_target_cards(source_data, target_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    print(f"Source cards scanned: {source_count}")
    print(f"Target cards scanned: {target_count}")
    print(f"Matched by id:        {matched_count}")
    print(f"Unmatched target:     {len(unmatched_cards)}")

    if unmatched_cards:
        print()
        print("Unmatched target cards:")
        for name in unmatched_cards:
            print(f"  - {name}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
