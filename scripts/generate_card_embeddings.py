#!/usr/bin/env python3
"""
Script to generate embeddings for Clash Royale cards.

Creates three types of embeddings for each card:
- Text embeddings: OpenAI text-embedding-3-small (1536 dimensions)
- Image embeddings: CLIP ViT-B/32 (512 dimensions)
- Combined embeddings: Normalized concatenation (2048 dimensions)

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key

Usage:
    python scripts/generate_card_embeddings.py
    python scripts/generate_card_embeddings.py --input data/cards_with_vibes.json --output data/cards_with_embeddings.json
    python scripts/generate_card_embeddings.py --concurrency 10
"""

import argparse
import asyncio
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

load_dotenv()

# OpenAI text embedding model
TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
TEXT_EMBEDDING_DIMS = 1536

# CLIP model for image embeddings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_EMBEDDING_DIMS = 512


def load_clip_model() -> tuple[CLIPModel, CLIPProcessor]:
    """Load the CLIP model and processor."""
    print(f"Loading CLIP model: {CLIP_MODEL_NAME}...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"  CLIP loaded on device: {device}")
    
    return model, processor, device


def download_image(url: str, timeout: int = 30) -> Image.Image | None:
    """Download an image from URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"    Warning: Failed to download image from {url}: {e}")
        return None


def get_image_embedding(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> np.ndarray | None:
    """Generate CLIP image embedding for a PIL Image."""
    try:
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        with torch.no_grad():
            # Use vision_model + visual_projection directly for reliability
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output
            image_features = model.visual_projection(pooled_output)
        
        # Normalize the embedding
        embedding = image_features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    except Exception as e:
        print(f"    Warning: Failed to generate image embedding: {e}")
        return None


def build_text_for_embedding(card: dict) -> str:
    """Build a rich text representation of a card for embedding."""
    parts = []
    
    # Card name
    name = card.get("name", "")
    parts.append(f"Card: {name}")
    
    # Rarity
    rarity = card.get("rarity", "")
    if rarity:
        parts.append(f"Rarity: {rarity}")
    
    # Elixir cost
    elixir = card.get("elixirCost")
    if elixir is not None:
        parts.append(f"Elixir cost: {elixir}")
    
    # Description
    description = card.get("description", "")
    if description:
        parts.append(f"Description: {description}")
    
    # Role tags
    role_tags = card.get("role_tags", [])
    if role_tags:
        parts.append(f"Roles: {', '.join(role_tags)}")
    
    # Vibe tags
    vibe_tags = card.get("vibe_tags", [])
    if vibe_tags:
        parts.append(f"Vibes: {', '.join(vibe_tags)}")
    
    # LLM vibe summary
    vibe_summary = card.get("llm_vibe_summary", "")
    if vibe_summary:
        parts.append(f"Community perception: {vibe_summary}")
    
    return "\n".join(parts)


async def get_text_embedding(
    client: AsyncOpenAI,
    text: str,
    semaphore: asyncio.Semaphore,
) -> np.ndarray | None:
    """Generate OpenAI text embedding."""
    async with semaphore:
        try:
            response = await client.embeddings.create(
                model=TEXT_EMBEDDING_MODEL,
                input=text,
                dimensions=TEXT_EMBEDDING_DIMS,
            )
            embedding = np.array(response.data[0].embedding)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"    Warning: Failed to generate text embedding: {e}")
            return None


def create_combined_embedding(
    text_embedding: np.ndarray | None,
    image_embedding: np.ndarray | None,
) -> np.ndarray | None:
    """
    Create a combined embedding via normalized concatenation.
    
    If one embedding is missing, pads with zeros.
    Result is normalized to unit length.
    """
    if text_embedding is None and image_embedding is None:
        return None
    
    # Handle missing embeddings with zero padding
    if text_embedding is None:
        text_embedding = np.zeros(TEXT_EMBEDDING_DIMS)
    if image_embedding is None:
        image_embedding = np.zeros(IMAGE_EMBEDDING_DIMS)
    
    # Concatenate
    combined = np.concatenate([text_embedding, image_embedding])
    
    # Normalize to unit length
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined


def load_cards(input_path: str) -> dict:
    """Load cards from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def process_card(
    card: dict,
    index: int,
    total: int,
    client: AsyncOpenAI,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Process a single card to generate all embeddings.
    
    Returns dict with text_embedding, image_embedding, combined_embedding.
    """
    card_name = card.get("name", "Unknown")
    print(f"[{index + 1}/{total}] Processing: {card_name}...", flush=True)
    
    result = {
        "text_embedding": None,
        "image_embedding": None,
        "combined_embedding": None,
    }
    
    # Get text embedding
    text = build_text_for_embedding(card)
    text_embedding = await get_text_embedding(client, text, semaphore)
    
    # Get image embedding
    image_url = card.get("iconUrls", {}).get("medium")
    image_embedding = None
    if image_url:
        image = download_image(image_url)
        if image:
            image_embedding = get_image_embedding(image, clip_model, clip_processor, device)
    
    # Create combined embedding
    combined_embedding = create_combined_embedding(text_embedding, image_embedding)
    
    # Convert to lists for JSON serialization
    if text_embedding is not None:
        result["text_embedding"] = text_embedding.tolist()
    if image_embedding is not None:
        result["image_embedding"] = image_embedding.tolist()
    if combined_embedding is not None:
        result["combined_embedding"] = combined_embedding.tolist()
    
    status_parts = []
    if text_embedding is not None:
        status_parts.append("text")
    if image_embedding is not None:
        status_parts.append("image")
    if combined_embedding is not None:
        status_parts.append("combined")
    
    print(f"[{index + 1}/{total}] Done: {card_name} ({', '.join(status_parts)})", flush=True)
    
    return result


async def process_all_cards(
    cards_data: dict,
    client: AsyncOpenAI,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    concurrency: int,
    output_path: Path,
) -> tuple[int, list]:
    """
    Process all cards to generate embeddings.
    
    Note: Image embedding must be done sequentially (CLIP model not async-safe),
    but text embeddings can be batched via the semaphore.
    """
    # Build flat list of all cards
    all_cards = []
    for key in ["items", "supportItems"]:
        if key in cards_data:
            for idx, card in enumerate(cards_data[key]):
                all_cards.append((key, idx, card))
    
    total_cards = len(all_cards)
    print(f"Processing {total_cards} cards...")
    print()
    
    semaphore = asyncio.Semaphore(concurrency)
    
    processed = 0
    errors = []
    
    # Process cards - we need to do this somewhat sequentially due to CLIP
    # but we can still batch text embedding requests
    for i, (key, idx, card) in enumerate(all_cards):
        try:
            embeddings = await process_card(
                card, i, total_cards,
                client, clip_model, clip_processor, device,
                semaphore,
            )
            
            # Add embeddings to card
            if embeddings["text_embedding"]:
                cards_data[key][idx]["text_embedding"] = embeddings["text_embedding"]
            if embeddings["image_embedding"]:
                cards_data[key][idx]["image_embedding"] = embeddings["image_embedding"]
            if embeddings["combined_embedding"]:
                cards_data[key][idx]["combined_embedding"] = embeddings["combined_embedding"]
            
            processed += 1
            
        except Exception as e:
            card_name = card.get("name", "Unknown")
            print(f"[{i + 1}/{total_cards}] ERROR ({card_name}): {e}", flush=True)
            errors.append((card_name, str(e)))
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cards_data, f, indent=2, ensure_ascii=False)
    
    return processed, errors


async def main_async(args):
    """Async main function."""
    print("Clash Royale Card Embedding Generator")
    print("=" * 50)
    print()
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Set it in your .env file or export it in your shell")
        sys.exit(1)
    
    print(f"Text embedding model: {TEXT_EMBEDDING_MODEL} ({TEXT_EMBEDDING_DIMS} dims)")
    print(f"Image embedding model: {CLIP_MODEL_NAME} ({IMAGE_EMBEDDING_DIMS} dims)")
    print(f"Combined embedding dimensions: {TEXT_EMBEDDING_DIMS + IMAGE_EMBEDDING_DIMS}")
    print(f"Concurrency: {args.concurrency}")
    print()
    
    # Load CLIP model
    clip_model, clip_processor, device = load_clip_model()
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
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=api_key)
    
    # Process all cards
    output_path = Path(args.output)
    processed, errors = await process_all_cards(
        cards_data, client, clip_model, clip_processor, device,
        args.concurrency, output_path,
    )
    
    print()
    print("=" * 50)
    print(f"Completed! Processed {processed} cards")
    print(f"Output saved to: {output_path}")
    
    if errors:
        print()
        print(f"Errors ({len(errors)}):")
        for card_name, error in errors[:10]:
            print(f"  - {card_name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    # Show embedding stats
    print()
    print("Embedding dimensions:")
    print(f"  - Text: {TEXT_EMBEDDING_DIMS}")
    print(f"  - Image: {IMAGE_EMBEDDING_DIMS}")
    print(f"  - Combined: {TEXT_EMBEDDING_DIMS + IMAGE_EMBEDDING_DIMS}")
    
    # Sample output
    print()
    print("Sample output:")
    for key in ["items", "supportItems"]:
        if key not in cards_data:
            continue
        for card in cards_data[key]:
            if "combined_embedding" in card:
                print(f"  {card['name']}:")
                text_emb = card.get("text_embedding", [])
                img_emb = card.get("image_embedding", [])
                combined_emb = card.get("combined_embedding", [])
                print(f"    Text embedding: {len(text_emb)} dims")
                print(f"    Image embedding: {len(img_emb)} dims")
                print(f"    Combined embedding: {len(combined_emb)} dims")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Clash Royale cards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate embeddings for all cards:
    python scripts/generate_card_embeddings.py

  Use custom input/output paths:
    python scripts/generate_card_embeddings.py --input data/cards_with_vibes.json --output data/cards_with_embeddings.json

  Adjust concurrency for text embedding API calls:
    python scripts/generate_card_embeddings.py --concurrency 15

Embedding Details:
  - Text: OpenAI text-embedding-3-small (1536 dimensions)
  - Image: CLIP ViT-B/32 (512 dimensions)
  - Combined: Normalized concatenation (2048 dimensions)
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/cards_with_vibes.json",
        help="Input JSON file with cards (default: data/cards_with_vibes.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/cards_with_embeddings.json",
        help="Output JSON file (default: data/cards_with_embeddings.json)"
    )
    
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Number of concurrent text embedding API requests (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
