"""
Staging utilities for piping data into Lance tables.

This module provides utilities for staging, validating, and inserting data
into Lance database tables from various sources like APIs, CSV files, and
in-memory data structures.
"""

from typing import Any, Dict, List, Optional, Union
import pyarrow as pa
import pandas as pd
from pathlib import Path


class StagingPipeline:
    """
    Pipeline for staging and loading data into Lance tables.
    
    Handles data validation, transformation, and batch insertion into Lance
    database tables with support for multiple data formats.
    """
    
    def __init__(self, db_path: str = "./data/lance_db"):
        """
        Initialize the staging pipeline.
        
        Args:
            db_path: Path to the Lance database directory
        """
        import lancedb
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
    
    def stage_cards(
        self,
        cards_data: Union[List[Dict[str, Any]], pd.DataFrame],
        validate: bool = True,
        batch_size: int = 100
    ) -> int:
        """
        Stage and insert card data into the Cards table.
        
        Args:
            cards_data: List of card dictionaries or pandas DataFrame
            validate: Whether to validate data before insertion
            batch_size: Number of records to insert per batch
        
        Returns:
            int: Number of cards successfully inserted
        
        Raises:
            ValueError: If validation fails
        
        Example:
            >>> pipeline = StagingPipeline()
            >>> cards = [
            ...     {
            ...         "id": 26000000,
            ...         "name": "Hog Rider",
            ...         "rarity": "Rare",
            ...         "type": "Troop",
            ...         "elixir": 4,
            ...         "description": "Fast melee troop...",
            ...         "role_tags": ["win condition"],
            ...         "vibe_tags": ["annoying", "tryhard"]
            ...     }
            ... ]
            >>> count = pipeline.stage_cards(cards)
        """
        if isinstance(cards_data, pd.DataFrame):
            cards_data = cards_data.to_dict(orient="records")
        
        if validate:
            self._validate_cards(cards_data)
        
        table = self.db.open_table("cards")
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(cards_data), batch_size):
            batch = cards_data[i:i + batch_size]
            table.add(batch)
            total_inserted += len(batch)
        
        return total_inserted
    
    def stage_archetypes(
        self,
        archetypes_data: Union[List[Dict[str, Any]], pd.DataFrame],
        validate: bool = True,
        batch_size: int = 50
    ) -> int:
        """
        Stage and insert archetype data into the Archetypes table.
        
        Args:
            archetypes_data: List of archetype dictionaries or pandas DataFrame
            validate: Whether to validate data before insertion
            batch_size: Number of records to insert per batch
        
        Returns:
            int: Number of archetypes successfully inserted
        
        Raises:
            ValueError: If validation fails
        
        Example:
            >>> pipeline = StagingPipeline()
            >>> archetypes = [
            ...     {
            ...         "id": "hog_cycle",
            ...         "name": "Hog Cycle",
            ...         "description": "Fast-paced cycle deck",
            ...         "tags": ["cycle", "aggro", "fast"],
            ...         "playstyle_vibes": ["tryhard", "sweaty"]
            ...     }
            ... ]
            >>> count = pipeline.stage_archetypes(archetypes)
        """
        if isinstance(archetypes_data, pd.DataFrame):
            archetypes_data = archetypes_data.to_dict(orient="records")
        
        if validate:
            self._validate_archetypes(archetypes_data)
        
        table = self.db.open_table("archetypes")
        
        total_inserted = 0
        for i in range(0, len(archetypes_data), batch_size):
            batch = archetypes_data[i:i + batch_size]
            table.add(batch)
            total_inserted += len(batch)
        
        return total_inserted
    
    def stage_decks(
        self,
        decks_data: Union[List[Dict[str, Any]], pd.DataFrame],
        validate: bool = True,
        batch_size: int = 100
    ) -> int:
        """
        Stage and insert deck data into the Decks table.
        
        Args:
            decks_data: List of deck dictionaries or pandas DataFrame
            validate: Whether to validate data before insertion
            batch_size: Number of records to insert per batch
        
        Returns:
            int: Number of decks successfully inserted
        
        Raises:
            ValueError: If validation fails or deck doesn't have exactly 8 battle cards
        
        Example:
            >>> pipeline = StagingPipeline()
            >>> decks = [
            ...     {
            ...         "id": "deck_abc123",
            ...         "battle_card_ids": [26000000, 26000001, 26000002, 26000003,
            ...                      26000004, 26000005, 26000006, 26000007],
            ...         "tower_card_id": 28000000,
            ...         "archetype_id": "hog_cycle",
            ...         "average_elixir": 2.9,
            ...         "user_labels": ["tryhard", "cycle"]
            ...     }
            ... ]
            >>> count = pipeline.stage_decks(decks)
        """
        if isinstance(decks_data, pd.DataFrame):
            decks_data = decks_data.to_dict(orient="records")
        
        if validate:
            self._validate_decks(decks_data)
        
        table = self.db.open_table("decks")
        
        total_inserted = 0
        for i in range(0, len(decks_data), batch_size):
            batch = decks_data[i:i + batch_size]
            table.add(batch)
            total_inserted += len(batch)
        
        return total_inserted
    
    def stage_from_csv(
        self,
        table_name: str,
        csv_path: Union[str, Path],
        validate: bool = True
    ) -> int:
        """
        Stage data from a CSV file into the specified table.
        
        Args:
            table_name: Name of the table ("cards", "archetypes", or "decks")
            csv_path: Path to the CSV file
            validate: Whether to validate data before insertion
        
        Returns:
            int: Number of records successfully inserted
        
        Raises:
            ValueError: If table_name is invalid
            
        Example:
            >>> pipeline = StagingPipeline()
            >>> count = pipeline.stage_from_csv("cards", "data/cards.csv")
        """
        df = pd.read_csv(csv_path)
        
        if table_name == "cards":
            return self.stage_cards(df, validate=validate)
        elif table_name == "archetypes":
            return self.stage_archetypes(df, validate=validate)
        elif table_name == "decks":
            return self.stage_decks(df, validate=validate)
        else:
            raise ValueError(f"Unknown table name: {table_name}")
    
    def stage_from_api(
        self,
        table_name: str,
        api_data: List[Dict[str, Any]],
        transform_fn: Optional[callable] = None,
        validate: bool = True
    ) -> int:
        """
        Stage data from an API response into the specified table.
        
        Args:
            table_name: Name of the table ("cards", "archetypes", or "decks")
            api_data: List of dictionaries from API response
            transform_fn: Optional function to transform API data to table schema
            validate: Whether to validate data before insertion
        
        Returns:
            int: Number of records successfully inserted
        
        Example:
            >>> pipeline = StagingPipeline()
            >>> api_cards = fetch_cards_from_api()
            >>> def transform(card):
            ...     return {
            ...         "id": card["id"],
            ...         "name": card["name"],
            ...         "elixir": card["elixirCost"],
            ...         # ... more transformations
            ...     }
            >>> count = pipeline.stage_from_api("cards", api_cards, transform)
        """
        if transform_fn:
            api_data = [transform_fn(record) for record in api_data]
        
        if table_name == "cards":
            return self.stage_cards(api_data, validate=validate)
        elif table_name == "archetypes":
            return self.stage_archetypes(api_data, validate=validate)
        elif table_name == "decks":
            return self.stage_decks(api_data, validate=validate)
        else:
            raise ValueError(f"Unknown table name: {table_name}")
    
    def _validate_cards(self, cards: List[Dict[str, Any]]) -> None:
        """
        Validate card data before insertion.
        
        Args:
            cards: List of card dictionaries
        
        Raises:
            ValueError: If validation fails
        """
        required_fields = ["id", "name", "rarity", "type", "elixir"]
        valid_rarities = ["Common", "Rare", "Epic", "Legendary", "Champion"]
        valid_types = ["Troop", "Spell", "Building", "Champion", "Tower Troop"]
        
        for i, card in enumerate(cards):
            # Check required fields
            for field in required_fields:
                if field not in card:
                    raise ValueError(f"Card {i}: Missing required field '{field}'")
            
            # Validate rarity
            if card["rarity"] not in valid_rarities:
                raise ValueError(f"Card {i}: Invalid rarity '{card['rarity']}'")
            
            # Validate type
            if card["type"] not in valid_types:
                raise ValueError(f"Card {i}: Invalid type '{card['type']}'")
            
            # Validate elixir cost
            if not isinstance(card["elixir"], int) or card["elixir"] < 0 or card["elixir"] > 10:
                raise ValueError(f"Card {i}: Invalid elixir cost '{card['elixir']}'")
    
    def _validate_archetypes(self, archetypes: List[Dict[str, Any]]) -> None:
        """
        Validate archetype data before insertion.
        
        Args:
            archetypes: List of archetype dictionaries
        
        Raises:
            ValueError: If validation fails
        """
        required_fields = ["id", "name"]
        
        for i, archetype in enumerate(archetypes):
            for field in required_fields:
                if field not in archetype:
                    raise ValueError(f"Archetype {i}: Missing required field '{field}'")
    
    def _validate_decks(self, decks: List[Dict[str, Any]]) -> None:
        """
        Validate deck data before insertion.
        
        Args:
            decks: List of deck dictionaries
        
        Raises:
            ValueError: If validation fails or deck doesn't have exactly 8 battle cards
        """
        required_fields = ["id", "battle_card_ids", "tower_card_id"]
        
        for i, deck in enumerate(decks):
            # Check required fields
            for field in required_fields:
                if field not in deck:
                    raise ValueError(f"Deck {i}: Missing required field '{field}'")
            
            # Validate card count
            if len(deck["battle_card_ids"]) != 8:
                raise ValueError(
                    f"Deck {i}: Must have exactly 8 battle cards, got {len(deck['battle_card_ids'])}"
                )
            
            # Validate all card IDs are integers
            if not all(isinstance(card_id, int) for card_id in deck["battle_card_ids"]):
                raise ValueError(f"Deck {i}: All battle_card_ids must be integers")
            if not isinstance(deck["tower_card_id"], int):
                raise ValueError(f"Deck {i}: tower_card_id must be an integer")


def compute_average_elixir(battle_card_ids: List[int], cards_table) -> float:
    """
    Compute average elixir cost for a deck.
    
    Args:
        battle_card_ids: List of 8 battle card IDs
        cards_table: Lance table containing card data
    
    Returns:
        float: Average elixir cost
    
    Example:
        >>> from semantic_dot_clash.tables import create_all_tables
        >>> tables = create_all_tables()
        >>> avg = compute_average_elixir([26000000, 26000001, ...], tables["cards"])
    """
    total_elixir = 0
    for card_id in battle_card_ids:
        result = cards_table.search().where(f"id = {card_id}").limit(1).to_list()
        if result:
            total_elixir += result[0]["elixir"]
    
    return total_elixir / len(battle_card_ids)


def generate_deck_id(battle_card_ids: List[int], tower_card_id: int) -> str:
    """
    Generate a unique deck ID from card IDs.
    
    Args:
        battle_card_ids: List of 8 battle card IDs
        tower_card_id: Tower troop card ID
    
    Returns:
        str: Unique deck identifier
    
    Example:
        >>> deck_id = generate_deck_id([26000000, 26000001, 26000002, ...], 28000000)
        >>> print(deck_id)  # "deck_a1b2c3d4"
    """
    import hashlib
    
    # Sort to ensure consistent ID regardless of card order
    sorted_ids = sorted(battle_card_ids)
    id_string = "_".join(str(cid) for cid in sorted_ids + [tower_card_id])
    
    # Generate hash
    hash_obj = hashlib.md5(id_string.encode())
    hash_hex = hash_obj.hexdigest()[:8]
    
    return f"deck_{hash_hex}"


__all__ = [
    "StagingPipeline",
    "compute_average_elixir",
    "generate_deck_id",
]
