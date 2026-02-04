# Lance Tables for Semantic Dot Clash

This module defines the Lance database schemas and staging utilities for the Semantic Dot Clash project.

## Overview

The module provides three main tables:

1. **Cards** - Individual cards with multimodal embeddings
2. **Archetypes** - High-level deck styles and meta archetypes
3. **Decks** - Specific 8-card deck combinations

## Quick Start

### Creating Tables

```python
from semantic_dot_clash.tables import create_all_tables

# Create all tables
tables = create_all_tables("./data/lance_db")

# Access individual tables
cards_table = tables["cards"]
archetypes_table = tables["archetypes"]
decks_table = tables["decks"]
```

### Staging Data

```python
from semantic_dot_clash.tables import StagingPipeline

# Initialize the pipeline
pipeline = StagingPipeline("./data/lance_db")

# Stage cards
cards = [
    {
        "id": 26000000,
        "name": "Hog Rider",
        "rarity": "Rare",
        "type": "Troop",
        "elixir": 4,
        "description": "Fast melee troop that targets buildings",
        "role_tags": ["win condition"],
        "vibe_tags": ["annoying", "tryhard"]
    }
]
pipeline.stage_cards(cards)

# Stage from CSV
pipeline.stage_from_csv("cards", "data/cards.csv")

# Stage from API with transformation
api_cards = fetch_from_api()
pipeline.stage_from_api(
    "cards",
    api_cards,
    transform_fn=lambda x: {"id": x["id"], "name": x["name"], ...}
)
```

## Table Schemas

### Cards Table

Stores individual cards with multimodal representations:

- **id**: Unique card ID
- **name**: Card name
- **rarity**: Common/Rare/Epic/Legendary/Champion
- **type**: Troop/Spell/Building
- **elixir**: Elixir cost (1-10)
- **description**: API description
- **image**: Raw image bytes
- **image_embedding**: Visual embedding from CLIP/ViT (512-dim)
- **text_embedding**: Semantic embedding from LLM (768-dim)
- **combined_embedding**: Fused multimodal embedding (1024-dim)
- **role_tags**: Tactical roles (win condition, cycle, splash, etc.)
- **vibe_tags**: Player perception (toxic, annoying, wholesome, etc.)
- **crowd_ratings**: Aggregated vibe ratings
- **llm_vibe_summary**: LLM interpretation of player perception

### Archetypes Table

Stores high-level deck styles:

- **id**: Unique archetype ID (e.g., "hog_cycle")
- **name**: Human-readable name
- **description**: Archetype description
- **example_decks**: List of example deck compositions
- **embedding**: Semantic embedding (768-dim)
- **tags**: Tactical tags (cycle, control, air, aggro)
- **meta_strength**: Win-rate or performance score
- **playstyle_vibes**: Vibe tags for the archetype
- **vibe_embedding**: Embedding of vibe description (768-dim)
- **llm_vibe_summary**: Natural language summary of archetype feel

### Decks Table

Stores specific 8-card decks:

- **id**: Unique deck ID
- **card_ids**: Array of 8 card IDs
- **archetype_id**: Link to archetype
- **average_elixir**: Average elixir cost
- **roles**: Summary of tactical roles
- **deck_embedding**: Combined card embeddings (1024-dim)
- **synergy_embedding**: Embedding of card synergies (768-dim)
- **meta_score**: Performance metric
- **user_labels**: Player-applied labels
- **crowd_ratings**: Aggregated vibe ratings
- **llm_vibe_summary**: LLM classification of deck vibe
- **conflict_tags**: Counter strategies
- **combo_notes**: Key card combinations

## Semantic Search Examples

```python
import lancedb

db = lancedb.connect("./data/lance_db")
cards_table = db.open_table("cards")

# Find cards similar to a query embedding
results = cards_table.search(query_embedding).limit(10).to_list()

# Filter by vibe
toxic_cards = cards_table.search().where(
    "array_contains(vibe_tags, 'toxic')"
).to_list()

# Complex semantic query
results = cards_table.search(query_embedding).where(
    "elixir <= 3 AND array_contains(role_tags, 'cycle')"
).limit(5).to_list()
```

## Utility Functions

### Generate Deck ID

```python
from semantic_dot_clash.tables import generate_deck_id

card_ids = [26000000, 26000001, 26000002, 26000003, 
            26000004, 26000005, 26000006, 26000007]
deck_id = generate_deck_id(card_ids)
# Returns: "deck_a1b2c3d4"
```

### Compute Average Elixir

```python
from semantic_dot_clash.tables import compute_average_elixir

avg_elixir = compute_average_elixir(card_ids, cards_table)
# Returns: 2.875
```

## Data Validation

The staging pipeline includes built-in validation:

- **Cards**: Validates required fields, rarity values, card types, and elixir costs
- **Archetypes**: Validates required fields
- **Decks**: Validates required fields and ensures exactly 8 cards

Disable validation if needed:

```python
pipeline.stage_cards(cards, validate=False)
```

## Batch Processing

All staging methods support batch processing:

```python
# Default batch size: 100 for cards, 50 for archetypes
pipeline.stage_cards(large_card_list, batch_size=500)
```

## Integration with LLMs and Embeddings

The tables are designed to store embeddings from various sources:

- **CLIP/ViT** for image embeddings
- **Sentence transformers** or **OpenAI embeddings** for text
- **Custom fusion models** for combined embeddings

Example workflow:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embedding
text = f"{card['name']}: {card['description']}"
embedding = model.encode(text).tolist()

# Add to card data
card['text_embedding'] = embedding
pipeline.stage_cards([card])
```
