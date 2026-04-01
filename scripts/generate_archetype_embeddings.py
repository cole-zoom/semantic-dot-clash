#!/usr/bin/env python3
"""
Script to generate semantic embeddings for Clash Royale archetypes.

Creates one normalized text embedding per archetype using:
- name
- description
- tags
- playstyle_vibes
- llm_vibe_summary

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key

Usage:
    python scripts/generate_archetype_embeddings.py
    python scripts/generate_archetype_embeddings.py --input data/archetypes_seed.json --output data/archetypes_with_embeddings.json
    python scripts/generate_archetype_embeddings.py --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
TEXT_EMBEDDING_DIMS = 768


def load_archetypes(input_path: Path) -> list[dict]:
    """Load archetypes from a JSON file."""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected archetype input JSON to be a list")

    return data


def build_text_for_embedding(archetype: dict) -> str:
    """Build a labeled text representation of an archetype for embedding."""
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

    playstyle_vibes = archetype.get("playstyle_vibes", [])
    if playstyle_vibes:
        parts.append(f"Playstyle vibes: {', '.join(playstyle_vibes)}")

    vibe_summary = archetype.get("llm_vibe_summary", "")
    if vibe_summary:
        parts.append(f"Vibe summary: {vibe_summary}")

    return "\n".join(parts)


async def get_text_embedding(
    client: AsyncOpenAI,
    text: str,
    semaphore: asyncio.Semaphore,
) -> list[float] | None:
    """Generate a normalized float32 embedding."""
    async with semaphore:
        try:
            response = await client.embeddings.create(
                model=TEXT_EMBEDDING_MODEL,
                input=text,
                dimensions=TEXT_EMBEDDING_DIMS,
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32).tolist()
        except Exception as e:
            print(f"    Warning: Failed to generate embedding: {e}")
            return None


async def process_archetype(
    archetype: dict,
    index: int,
    total: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate an embedding for a single archetype."""
    name = archetype.get("name", "Unknown")
    print(f"[{index + 1}/{total}] Processing: {name}...", flush=True)

    result = dict(archetype)
    text = build_text_for_embedding(archetype)
    embedding = await get_text_embedding(client, text, semaphore)

    if embedding is not None:
        result["embedding"] = embedding
        print(
            f"[{index + 1}/{total}] Done: {name} ({len(embedding)} dims)",
            flush=True,
        )
    else:
        print(f"[{index + 1}/{total}] Done: {name} (no embedding)", flush=True)

    return result


async def process_all_archetypes(
    archetypes: list[dict],
    client: AsyncOpenAI,
    concurrency: int,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Process all archetypes and return updated rows plus errors."""
    semaphore = asyncio.Semaphore(concurrency)
    errors: list[tuple[str, str]] = []

    async def _wrapped_process(index: int, archetype: dict) -> dict:
        try:
            return await process_archetype(
                archetype=archetype,
                index=index,
                total=len(archetypes),
                client=client,
                semaphore=semaphore,
            )
        except Exception as e:
            name = archetype.get("name", "Unknown")
            errors.append((name, str(e)))
            print(f"[{index + 1}/{len(archetypes)}] ERROR ({name}): {e}", flush=True)
            return dict(archetype)

    results = await asyncio.gather(
        *(_wrapped_process(index, archetype) for index, archetype in enumerate(archetypes))
    )
    return results, errors


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    print("Clash Royale Archetype Embedding Generator")
    print("=" * 50)
    print()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Set it in your .env file or export it in your shell")
        sys.exit(1)

    print(f"Embedding model: {TEXT_EMBEDDING_MODEL} ({TEXT_EMBEDDING_DIMS} dims)")
    print(f"Concurrency: {args.concurrency}")
    print()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading archetypes from: {input_path}")
    archetypes = load_archetypes(input_path)
    print(f"Loaded {len(archetypes)} archetypes")
    print()

    client = AsyncOpenAI(api_key=api_key)
    output_path = Path(args.output)
    results, errors = await process_all_archetypes(
        archetypes=archetypes,
        client=client,
        concurrency=args.concurrency,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 50)
    print(f"Completed! Processed {len(results)} archetypes")
    print(f"Output saved to: {output_path}")

    if errors:
        print()
        print(f"Errors ({len(errors)}):")
        for archetype_name, error in errors[:10]:
            print(f"  - {archetype_name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print()
    print("Sample output:")
    for archetype in results:
        embedding = archetype.get("embedding")
        if embedding:
            print(f"  {archetype.get('name', 'Unknown')}: {len(embedding)} dims")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Clash Royale archetypes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate embeddings for all archetypes:
    python scripts/generate_archetype_embeddings.py

  Use custom input/output paths:
    python scripts/generate_archetype_embeddings.py --input data/archetypes_seed.json --output data/archetypes_with_embeddings.json

  Adjust concurrency for embedding API calls:
    python scripts/generate_archetype_embeddings.py --concurrency 15

Embedding Details:
  - Text: OpenAI text-embedding-3-small (768 dimensions)
  - Fields: name, description, tags, playstyle_vibes, llm_vibe_summary
  - Output: normalized float32 embedding stored as `embedding`
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/archetypes_seed.json",
        help="Input JSON file with archetypes (default: data/archetypes_seed.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/archetypes_with_embeddings.json",
        help="Output JSON file (default: data/archetypes_with_embeddings.json)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Number of concurrent embedding API requests (default: 10)",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
