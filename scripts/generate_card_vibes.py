#!/usr/bin/env python3
"""
Script to generate vibe/sentiment data for each Clash Royale card using OpenAI.

This script analyzes player sentiment from content creator transcripts and generates
structured vibe data (role_tags, vibe_tags, crowd_ratings, llm_vibe_summary) for each card.

Uses async/await with concurrency limit for faster processing.

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key

Usage:
    python scripts/generate_card_vibes.py
    python scripts/generate_card_vibes.py --input data/cards_with_descriptions.json --output data/cards_with_vibes.json
    python scripts/generate_card_vibes.py --concurrency 15
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Paths to transcript files
TRANSCRIPT_1_PATH = "data/all_cards_ranked_transcript.txt"
TRANSCRIPT_2_PATH = "data/all_cards_ranked_2.txt"

# OpenAI model to use
MODEL = "gpt-5.2"

# JSON schema for structured output
VIBE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "card_vibe_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "card_name": {
                    "type": "string",
                    "description": "The name of the card being analyzed"
                },
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
                },
                "llm_vibe_summary": {
                    "type": "string",
                    "description": "2-4 sentence natural language summary of community perception"
                }
            },
            "required": ["card_name", "role_tags", "vibe_tags", "crowd_ratings", "llm_vibe_summary"],
            "additionalProperties": False
        }
    }
}


def get_system_prompt(transcript_1: str, transcript_2: str) -> str:
    """Generate the system prompt with transcripts embedded."""
    return f'''You are analyzing player sentiment and perception data for Clash Royale cards based on community opinions from content creators and players.
Source Material Context
You have access to two detailed tier list transcripts from Clash Royale content creators who ranked every card in the game. These transcripts contain:

Emotional reactions and frustration levels toward specific cards
Skill assessments (no-skill vs high-skill)
Meta viability opinions
Personal anecdotes about playing with/against cards
Comparisons between similar cards
Slang and community terminology (e.g., "brain rot", "AIDS", "toxic", "spammy")

Transcript 1 (Top Ladder Player Perspective):
{transcript_1}

Transcript 2 (Jinxy - Graveyard Player Perspective):
{transcript_2}

Your Task
Given a single Clash Royale card name, analyze both transcripts and generate the following structured data representing how the player community perceives this card:
json{{
  "card_name": "<card name>",
  "role_tags": [],
  "vibe_tags": [],
  "crowd_ratings": [{{"vibe": "toxic", "score": 0.7}}, {{"vibe": "annoying", "score": 0.5}}],
  "llm_vibe_summary": ""
}}
Field Definitions
role_tags (list of strings): Tactical/functional roles the card fulfills. Select from:

"win condition", "secondary win condition", "cycle", "splash", "tank", "mini tank", "anti-air", "building", "spell", "swarm", "DPS", "support", "bridge spam", "bait", "control", "beatdown support", "tank killer", "distraction", "kiting", "spell bait","gay"

vibe_tags (list of strings): Community perception/feeling tags. Select from:

"toxic", "annoying", "spammy", "no-skill", "brain-dead", "respectable", "high-skill", "wholesome", "fun", "tryhard", "troll", "satisfying", "frustrating to face", "underrated", "overrated", "timeless", "gimmicky", "matchup dependent", "coin-flip", "mid-ladder menace", "top-ladder viable", "dead card", "sleeper", "cheese"

crowd_ratings (array of {{"vibe": string, "score": float 0.0-1.0}}): Quantified intensity scores for relevant vibes. Only include vibes that are explicitly or strongly implied in the transcripts. Score meaning:

0.0-0.3: Mildly associated
0.4-0.6: Moderately associated
0.7-1.0: Strongly associated

llm_vibe_summary (string): A 2-4 sentence natural language summary capturing how the community actually talks about and feels about this card. Use the tone and language style from the transcripts (casual, sometimes crude, emotionally charged). Do not reference specific opinions from the creators.

Guidelines

Be faithful to the source material - Only include perceptions that are directly stated or strongly implied in the transcripts
Capture emotional intensity - If a creator expresses strong hatred or love, reflect that in the ratings
Note disagreements - If the two creators have different opinions, mention this in the summary
Use community language - The summary should sound like how players actually talk, not corporate PR speak
Context matters - Consider whether opinions are about the base card, its EVO, or both (focus on base card unless EVO is inseparable from perception)
If card is barely mentioned - Still generate output but note limited data in the summary'''


def load_transcripts() -> tuple[str, str]:
    """Load the transcript files."""
    transcript_1_path = Path(TRANSCRIPT_1_PATH)
    transcript_2_path = Path(TRANSCRIPT_2_PATH)
    
    if not transcript_1_path.exists():
        raise FileNotFoundError(f"Transcript 1 not found: {transcript_1_path}")
    if not transcript_2_path.exists():
        raise FileNotFoundError(f"Transcript 2 not found: {transcript_2_path}")
    
    with open(transcript_1_path, "r", encoding="utf-8") as f:
        transcript_1 = f.read()
    
    with open(transcript_2_path, "r", encoding="utf-8") as f:
        transcript_2 = f.read()
    
    return transcript_1, transcript_2


def load_cards(input_path: str) -> dict:
    """Load cards from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def analyze_card(
    client: AsyncOpenAI,
    card_name: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> tuple[int, dict | None, str | None]:
    """
    Analyze a single card using OpenAI with structured output.
    
    Args:
        client: Async OpenAI client
        card_name: Name of the card to analyze
        system_prompt: The system prompt with transcripts
        semaphore: Semaphore to limit concurrency
        index: Card index for progress tracking
        total: Total number of cards
        
    Returns:
        Tuple of (index, vibe_data dict or None, error message or None)
    """
    async with semaphore:
        try:
            print(f"[{index + 1}/{total}] Analyzing: {card_name}...", flush=True)
            
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze the card: {card_name}"}
                ],
                response_format=VIBE_SCHEMA,
                temperature=0.7,
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            print(f"[{index + 1}/{total}] Done: {card_name}", flush=True)
            return (index, result, None)
            
        except Exception as e:
            print(f"[{index + 1}/{total}] ERROR ({card_name}): {e}", flush=True)
            return (index, None, str(e))


async def process_all_cards(
    client: AsyncOpenAI,
    cards_data: dict,
    system_prompt: str,
    concurrency: int,
    output_path: Path,
) -> tuple[int, list]:
    """
    Process all cards concurrently with a semaphore limit.
    
    Args:
        client: Async OpenAI client
        cards_data: The cards data structure
        system_prompt: The system prompt with transcripts
        concurrency: Max concurrent requests
        output_path: Path to save output
        
    Returns:
        Tuple of (processed count, list of errors)
    """
    # Build a flat list with references to original card dicts
    all_cards = []
    for key in ["items", "supportItems"]:
        if key in cards_data:
            for idx, card in enumerate(cards_data[key]):
                all_cards.append((key, idx, card))
    
    total_cards = len(all_cards)
    print(f"Processing {total_cards} cards with concurrency={concurrency}...")
    print()
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create tasks for all cards
    tasks = []
    for i, (key, idx, card) in enumerate(all_cards):
        card_name = card.get("name", "Unknown")
        task = analyze_card(client, card_name, system_prompt, semaphore, i, total_cards)
        tasks.append((key, idx, task))
    
    # Run all tasks concurrently
    results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
    
    # Process results and update cards_data
    processed = 0
    errors = []
    
    for (key, idx, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            card_name = cards_data[key][idx].get("name", "Unknown")
            errors.append((card_name, str(result)))
            continue
            
        index, vibe_data, error = result
        
        if error:
            card_name = cards_data[key][idx].get("name", "Unknown")
            errors.append((card_name, error))
            continue
        
        if vibe_data:
            # Add vibe data directly to the card in cards_data
            cards_data[key][idx]["role_tags"] = vibe_data.get("role_tags", [])
            cards_data[key][idx]["vibe_tags"] = vibe_data.get("vibe_tags", [])
            
            # Convert crowd_ratings from array to dict format
            crowd_ratings_list = vibe_data.get("crowd_ratings", [])
            crowd_ratings_dict = {
                item["vibe"]: item["score"] 
                for item in crowd_ratings_list 
                if "vibe" in item and "score" in item
            }
            cards_data[key][idx]["crowd_ratings"] = crowd_ratings_dict
            
            cards_data[key][idx]["llm_vibe_summary"] = vibe_data.get("llm_vibe_summary", "")
            processed += 1
    
    # Save final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cards_data, f, indent=2, ensure_ascii=False)
    
    return processed, errors


async def main_async(args):
    """Async main function."""
    print("Clash Royale Card Vibe Generator (Async)")
    print("=" * 50)
    print()
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Set it in your .env file or export it in your shell")
        sys.exit(1)
    
    print(f"Using model: {MODEL}")
    print(f"Concurrency: {args.concurrency}")
    print()
    
    # Load transcripts
    print("Loading transcripts...")
    try:
        transcript_1, transcript_2 = load_transcripts()
        print(f"  - Transcript 1: {len(transcript_1)} characters")
        print(f"  - Transcript 2: {len(transcript_2)} characters")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Generate system prompt
    system_prompt = get_system_prompt(transcript_1, transcript_2)
    print(f"  - System prompt: {len(system_prompt)} characters")
    print()
    
    # Load cards
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading cards from: {input_path}")
    cards_data = load_cards(input_path)
    
    total_items = len(cards_data.get("items", []))
    total_support = len(cards_data.get("supportItems", []))
    print(f"Loaded {total_items + total_support} cards")
    print()
    
    # Initialize async OpenAI client
    client = AsyncOpenAI(api_key=api_key)
    
    # Process all cards
    output_path = Path(args.output)
    processed, errors = await process_all_cards(
        client, cards_data, system_prompt, args.concurrency, output_path
    )
    
    print()
    print("=" * 50)
    print(f"Completed! Processed {processed} cards")
    print(f"Output saved to: {output_path}")
    
    if errors:
        print()
        print(f"Errors ({len(errors)}):")
        for card_name, error in errors[:10]:  # Show first 10
            print(f"  - {card_name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    # Show sample output
    print()
    print("Sample output:")
    for key in ["items", "supportItems"]:
        if key not in cards_data:
            continue
        for card in cards_data[key]:
            if "llm_vibe_summary" in card and card["llm_vibe_summary"]:
                print(f"  {card['name']}:")
                print(f"    Role tags: {card.get('role_tags', [])}")
                print(f"    Vibe tags: {card.get('vibe_tags', [])[:5]}...")
                summary = card.get('llm_vibe_summary', '')
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                print(f"    Summary: {summary}")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Generate vibe/sentiment data for Clash Royale cards using OpenAI (async)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate vibes for all cards (10 concurrent requests):
    python scripts/generate_card_vibes.py

  Use higher concurrency:
    python scripts/generate_card_vibes.py --concurrency 15

  Use custom input/output paths:
    python scripts/generate_card_vibes.py --input data/cards_with_descriptions.json --output data/cards_with_vibes.json
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/cards_with_descriptions.json",
        help="Input JSON file with cards (default: data/cards_with_descriptions.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/cards_with_vibes.json",
        help="Output JSON file (default: data/cards_with_vibes.json)"
    )
    
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Number of concurrent API requests (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
