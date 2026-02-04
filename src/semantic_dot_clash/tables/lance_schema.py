"""
Lance database schema definitions for Semantic Dot Clash.

This module defines the table schemas for storing Clash Royale cards, archetypes,
and decks with their multimodal embeddings and semantic annotations.
"""

import pyarrow as pa


def get_cards_schema() -> pa.Schema:
    """
    Schema for the Cards table.
    
    Purpose:
        Store every individual card with its multimodal representation combining
        visual embeddings, text embeddings, and crowd-sourced vibe ratings.
    
    Returns:
        pa.Schema: PyArrow schema for the Cards table
    
    Schema Fields:
        - id (int64): Unique card ID from Clash Royale API
        - name (string): Official card name (e.g., "Hog Rider", "Electro Wizard")
        - rarity (string): Card rarity tier - Common, Rare, Epic, Legendary, or Champion
        - type (string): Card type - Troop, Spell, or Building
        - elixir (int32): Elixir cost to deploy the card (1-10)
        - description (string): Official card description from the API
        - image (binary): Raw image bytes or URL reference for card artwork
        - image_embedding (fixed_size_list<float>[512]): Visual embedding from OpenAI image-embedding-3 model
        - text_embedding (fixed_size_list<float>[1536]): Semantic embedding from OpenAI text-embedding-3-small model
        - combined_embedding (fixed_size_list<float>[2048]): Normalized concatenation of text (1536) + image (512) embeddings
        - role_tags (list<string>): Tactical roles like "win condition", "cycle", "splash", "tank", "anti-air"
        - vibe_tags (list<string>): Player perception tags like "toxic", "annoying", "spammy", "gay", "wholesome", "fun", "tryhard"
        - crowd_ratings (map<string, float>): Aggregated crowd ratings for each vibe (e.g., {"toxic": 0.73, "annoying": 0.55})
        - llm_vibe_summary (string): LLM-generated natural language interpretation of how players perceive the card
    
    Example:
        >>> schema = get_cards_schema()
        >>> # Use schema to create Lance table
        >>> import lancedb
        >>> db = lancedb.connect("./data")
        >>> tbl = db.create_table("cards", schema=schema)
    """
    return pa.schema([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("rarity", pa.string(), nullable=False),
        pa.field("type", pa.string(), nullable=False),
        pa.field("elixir", pa.int32(), nullable=False),
        pa.field("description", pa.string(), nullable=True),
        pa.field("image", pa.binary(), nullable=True),
        pa.field("image_embedding", pa.list_(pa.float32(), 512), nullable=True),
        pa.field("text_embedding", pa.list_(pa.float32(), 1536), nullable=True),
        pa.field("combined_embedding", pa.list_(pa.float32(), 2048), nullable=True),
        pa.field("role_tags", pa.list_(pa.string()), nullable=True),
        pa.field("vibe_tags", pa.list_(pa.string()), nullable=True),
        pa.field("crowd_ratings", pa.list_(pa.struct([
            ("key", pa.string()),
            ("value", pa.float32())
        ])), nullable=True),
        pa.field("llm_vibe_summary", pa.string(), nullable=True),
    ])


def get_archetypes_schema() -> pa.Schema:
    """
    Schema for the Archetypes table.
    
    Purpose:
        Store high-level deck styles and meta archetypes like Hog Cycle, Beatdown,
        Lava Loon, Miner Control, etc., with their semantic embeddings and vibe profiles.
    
    Returns:
        pa.Schema: PyArrow schema for the Archetypes table
    
    Schema Fields:
        - id (string): Unique archetype identifier (e.g., "hog_cycle", "lava_loon", "log_bait")
        - name (string): Human-readable archetype name (e.g., "Hog Cycle", "Lava Loon")
        - description (string): Detailed description of the archetype's playstyle and strategy
        - example_decks (list<list<int64>>): List of example decks as arrays of card IDs
        - embedding (fixed_size_list<float>[768]): Semantic embedding of the archetype description
        - tags (list<string>): Tactical tags like "cycle", "control", "air", "aggro", "beatdown"
        - meta_strength (float): Optional meta performance score or win-rate indicator
        - playstyle_vibes (list<string>): Vibe tags like "annoying", "spammy", "toxic", "off-meta", "wholesome"
        - vibe_embedding (fixed_size_list<float>[768]): Embedding of the vibe description for semantic filtering
        - llm_vibe_summary (string): Natural language summary of the archetype's "feel" and player perception
    
    Example:
        >>> schema = get_archetypes_schema()
        >>> # Create archetype entry
        >>> archetype = {
        ...     "id": "hog_cycle",
        ...     "name": "Hog Cycle",
        ...     "description": "Fast-paced cycle deck focused on Hog Rider pressure",
        ...     "tags": ["cycle", "aggro", "fast"],
        ...     "playstyle_vibes": ["tryhard", "sweaty"]
        ... }
    """
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=True),
        pa.field("example_decks", pa.list_(pa.list_(pa.int64())), nullable=True),
        pa.field("embedding", pa.list_(pa.float32(), 768), nullable=True),
        pa.field("tags", pa.list_(pa.string()), nullable=True),
        pa.field("meta_strength", pa.float32(), nullable=True),
        pa.field("playstyle_vibes", pa.list_(pa.string()), nullable=True),
        pa.field("vibe_embedding", pa.list_(pa.float32(), 768), nullable=True),
        pa.field("llm_vibe_summary", pa.string(), nullable=True),
    ])


def get_decks_schema() -> pa.Schema:
    """
    Schema for the Decks table.
    
    Purpose:
        Store specific 8-card decks including meta decks, historic decks, and
        player-submitted decks with their embeddings, synergies, and vibe ratings.
    
    Returns:
        pa.Schema: PyArrow schema for the Decks table
    
    Schema Fields:
        - id (string): Unique deck identifier (UUID or hash of card composition)
        - card_ids (list<int64>): Array of exactly 8 card IDs that compose the deck
        - archetype_id (string): Foreign key linking to the archetype this deck belongs to
        - average_elixir (float): Auto-computed average elixir cost across all 8 cards
        - roles (list<string>): Summary of tactical roles present in the deck
        - deck_embedding (fixed_size_list<float>[1024]): Combined embedding of all 8 card embeddings
        - synergy_embedding (fixed_size_list<float>[768]): Embedding capturing how the cards work together
        - meta_score (float): Performance metric like win-rate, usage rate, or meta strength score
        - user_labels (list<string>): Player-applied labels like "toxic", "gay", "tryhard", "meme", "casual", "annoying", "sweaty"
        - crowd_ratings (map<string, float>): Aggregated crowd vibe ratings (e.g., {"toxic": 0.82, "meme": 0.12})
        - llm_vibe_summary (string): LLM-generated classification of the deck's vibe and player perception
        - conflict_tags (list<string>): Counter strategies like "hard counter air", "anti-swarm", "structure heavy"
        - combo_notes (string): LLM-generated description of key card combinations and synergies
    
    Example:
        >>> schema = get_decks_schema()
        >>> # Create deck entry
        >>> deck = {
        ...     "id": "deck_abc123",
        ...     "card_ids": [26000000, 26000001, 26000002, 26000003, 26000004, 26000005, 26000006, 26000007],
        ...     "archetype_id": "hog_cycle",
        ...     "average_elixir": 2.9,
        ...     "user_labels": ["tryhard", "cycle", "annoying"]
        ... }
    """
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("card_ids", pa.list_(pa.int64(), 8), nullable=False),
        pa.field("archetype_id", pa.string(), nullable=True),
        pa.field("average_elixir", pa.float32(), nullable=True),
        pa.field("roles", pa.list_(pa.string()), nullable=True),
        pa.field("deck_embedding", pa.list_(pa.float32(), 1024), nullable=True),
        pa.field("synergy_embedding", pa.list_(pa.float32(), 768), nullable=True),
        pa.field("meta_score", pa.float32(), nullable=True),
        pa.field("user_labels", pa.list_(pa.string()), nullable=True),
        pa.field("crowd_ratings", pa.list_(pa.struct([
            ("key", pa.string()),
            ("value", pa.float32())
        ])), nullable=True),
        pa.field("llm_vibe_summary", pa.string(), nullable=True),
        pa.field("conflict_tags", pa.list_(pa.string()), nullable=True),
        pa.field("combo_notes", pa.string(), nullable=True),
    ])


def create_all_tables(db_path: str = "./data/lance_db"):
    """
    Create all Lance tables with the defined schemas.
    
    Args:
        db_path: Path to the Lance database directory
    
    Returns:
        dict: Dictionary containing references to all created tables
        
    Example:
        >>> tables = create_all_tables("./my_data")
        >>> cards_table = tables["cards"]
        >>> archetypes_table = tables["archetypes"]
        >>> decks_table = tables["decks"]
    """
    import lancedb
    
    db = lancedb.connect(db_path)
    
    tables = {}
    
    # Create Cards table
    if "cards" not in db.table_names():
        tables["cards"] = db.create_table("cards", schema=get_cards_schema())
    else:
        tables["cards"] = db.open_table("cards")
    
    # Create Archetypes table
    if "archetypes" not in db.table_names():
        tables["archetypes"] = db.create_table("archetypes", schema=get_archetypes_schema())
    else:
        tables["archetypes"] = db.open_table("archetypes")
    
    # Create Decks table
    if "decks" not in db.table_names():
        tables["decks"] = db.create_table("decks", schema=get_decks_schema())
    else:
        tables["decks"] = db.open_table("decks")
    
    return tables


__all__ = [
    "get_cards_schema",
    "get_archetypes_schema",
    "get_decks_schema",
    "create_all_tables",
]
