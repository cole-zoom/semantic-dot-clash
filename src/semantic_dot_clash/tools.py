"""
Tools for interacting with the Lance database cards and archetypes tables.

Provides semantic search, similarity search, card lookup, archetype retrieval,
and deck scoring functionality using LanceDB Cloud.

Environment Variables:
    LANCE_URI: LanceDB Cloud URI
    LANCE_KEY: LanceDB Cloud API key
    OPENAI_API_KEY: OpenAI API key for query embedding

Usage:
    from semantic_dot_clash.tools import CardTools

    tools = CardTools()
    cards = tools.search_cards("fast cycle cards", elixir_max=3)
    archetypes = tools.search_archetypes("annoying low-cost control")
    ranked_archetypes = tools.select_archetype_for_core(
        user_request="annoying deck",
        core_card_ids=[26000032],
    )
    complementary = tools.search_complementary_cards(
        user_request="annoying deck",
        archetype_id="log_bait",
        core_card_ids=[26000032],
        role_hint="cheap anti-air support",
    )
    similar = tools.similar_cards(card_id=26000000)
    card = tools.get_card(card_id=26000000)
    score = tools.score_deck(battle_card_ids=[...], tower_card_id=28000000)
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any

import lancedb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

# OpenAI text embedding model (same as used in generate_card_embeddings.py)
TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
TEXT_EMBEDDING_DIMS = 1536
COMBINED_EMBEDDING_DIMS = 2048
ARCHETYPE_EMBEDDING_DIMS = 768

# Required roles for a balanced deck
REQUIRED_ROLES = {"win condition", "anti-air", "splash", "tank", "cycle"}

# Elixir thresholds for deck balance
EXPENSIVE_DECK_THRESHOLD = 4.2
CHEAP_DECK_THRESHOLD = 2.8

# Deck size
DECK_SIZE = 8


@dataclass
class DeckScore:
    """
    Comprehensive deck evaluation result.
    
    Attributes:
        avg_elixir: Average elixir cost of the 8 battle cards
        type_distribution: Count of cards by type (Troop, Spell, Building, etc.)
        rarity_distribution: Count of cards by rarity
        role_coverage: Set of roles covered by the deck
        missing_roles: Set of important roles not covered
        synergy_score: Average pairwise embedding similarity (0-1)
        balance_warnings: List of balance issues detected
        meta_strength: Heuristic score (0-100) based on role balance + synergy
        battle_cards: List of the 8 battle cards in the deck
        tower_card: Tower troop card data for the deck
    """
    avg_elixir: float
    type_distribution: dict[str, int]
    rarity_distribution: dict[str, int]
    role_coverage: set[str]
    missing_roles: set[str]
    synergy_score: float
    balance_warnings: list[str]
    meta_strength: float
    battle_cards: list[dict] = field(default_factory=list)
    tower_card: dict | None = None

    @property
    def cards(self) -> list[dict]:
        """Backward-compatible alias for the 8 battle cards."""
        return self.battle_cards
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "avg_elixir": self.avg_elixir,
            "type_distribution": self.type_distribution,
            "rarity_distribution": self.rarity_distribution,
            "role_coverage": list(self.role_coverage),
            "missing_roles": list(self.missing_roles),
            "synergy_score": self.synergy_score,
            "balance_warnings": self.balance_warnings,
            "meta_strength": self.meta_strength,
            "battle_cards": [
                {"id": c["id"], "name": c["name"], "elixir": c["elixir"]}
                for c in self.battle_cards
            ],
            "cards": [
                {"id": c["id"], "name": c["name"], "elixir": c["elixir"]}
                for c in self.battle_cards
            ],
            "tower_card": (
                {
                    "id": self.tower_card["id"],
                    "name": self.tower_card["name"],
                    "elixir": self.tower_card["elixir"],
                }
                if self.tower_card
                else None
            ),
        }


class CardTools:
    """
    Client for interacting with the Lance database cards and archetypes tables.

    Provides semantic search, similarity search, card lookup, archetype lookup,
    and deck scoring.
    
    Args:
        lance_uri: LanceDB Cloud URI (defaults to LANCE_URI env var)
        lance_key: LanceDB Cloud API key (defaults to LANCE_KEY env var)
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
    
    Example:
        >>> tools = CardTools()
        >>> results = tools.search_cards("fast cycle cards", elixir_max=3)
        >>> for card in results:
        ...     print(f"{card['name']}: {card['_distance']:.3f}")
    """
    
    def __init__(
        self,
        lance_uri: str | None = None,
        lance_key: str | None = None,
        openai_api_key: str | None = None,
    ):
        """Initialize CardTools with database and API connections."""
        # Get credentials from env vars if not provided
        self._lance_uri = lance_uri or os.environ.get("LANCE_URI")
        self._lance_key = lance_key or os.environ.get("LANCE_KEY")
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self._lance_uri:
            raise ValueError("LANCE_URI environment variable or lance_uri parameter is required")
        if not self._lance_key:
            raise ValueError("LANCE_KEY environment variable or lance_key parameter is required")
        if not self._openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable or openai_api_key parameter is required")
        
        # Connect to LanceDB
        self._db = lancedb.connect(
            uri=self._lance_uri,
            api_key=self._lance_key,
            region="us-east-1",
        )
        
        # Open the cards table
        if "cards" not in self._db.table_names():
            raise ValueError("Cards table does not exist. Run create_tables.py and load_cards_to_lance.py first.")
        self._cards_table = self._db.open_table("cards")

        if "archetypes" not in self._db.table_names():
            raise ValueError(
                "Archetypes table does not exist. Run create_tables.py and load_archetypes_to_lance.py first."
            )
        self._archetypes_table = self._db.open_table("archetypes")
        
        # Initialize OpenAI client
        self._openai = OpenAI(api_key=self._openai_api_key)
    
    def _embed_query(self, query: str) -> list[float]:
        """
        Embed a query string using OpenAI text-embedding-3-small.
        
        The embedding is padded with zeros to match the combined_embedding
        dimension (2048) used in the cards table.
        
        Args:
            query: The search query text
            
        Returns:
            List of floats representing the padded embedding (2048 dims)
        """
        response = self._openai.embeddings.create(
            model=TEXT_EMBEDDING_MODEL,
            input=query,
        )
        
        # Get the text embedding (1536 dims)
        text_embedding = response.data[0].embedding
        
        # Pad with zeros to match combined_embedding dimension (2048)
        # The combined_embedding is [text_embedding (1536) | image_embedding (512)]
        # For text-only queries, we pad the image portion with zeros
        padded_embedding = text_embedding + [0.0] * (COMBINED_EMBEDDING_DIMS - TEXT_EMBEDDING_DIMS)
        
        return padded_embedding

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        """L2-normalize an embedding vector."""
        array = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm == 0:
            return array.tolist()
        return (array / norm).tolist()

    def _embed_archetype_query(self, query: str) -> list[float]:
        """
        Embed a query string for the archetypes table.

        Archetype rows use 768-dim normalized text embeddings, so the query
        path must match that format instead of the 2048-dim card search path.
        """
        response = self._openai.embeddings.create(
            model=TEXT_EMBEDDING_MODEL,
            input=query,
            dimensions=ARCHETYPE_EMBEDDING_DIMS,
        )
        return self._normalize_embedding(response.data[0].embedding)

    def _escape_sql_string(self, value: str) -> str:
        """Escape single quotes for Lance string filters."""
        return value.replace("'", "''")
    
    def _infer_image_mime(self, image_bytes: bytes | None) -> str | None:
        """Infer the MIME type for raw image bytes."""
        if not image_bytes:
            return None
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                fmt = (img.format or "").lower()
        except Exception:
            return "image/png"
        if fmt == "jpeg":
            return "image/jpeg"
        if fmt:
            return f"image/{fmt}"
        return "image/png"

    def _encode_image_data_url(self, image_bytes: bytes | None) -> str | None:
        """Encode raw image bytes as a data URL string."""
        if not image_bytes:
            return None
        mime = self._infer_image_mime(image_bytes) or "image/png"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _clean_card_result(self, card: dict, include_image: bool = False) -> dict:
        """
        Clean a card result by removing embedding fields for readability.
        
        Args:
            card: Raw card dictionary from LanceDB
            
        Returns:
            Cleaned card dictionary without embedding fields
        """
        # Fields to exclude from results
        embedding_fields = {"text_embedding", "image_embedding", "combined_embedding", "image"}

        cleaned = {k: v for k, v in card.items() if k not in embedding_fields}
        if include_image:
            cleaned["image_data_url"] = self._encode_image_data_url(card.get("image"))
        return cleaned

    def _clean_archetype_result(self, archetype: dict) -> dict:
        """Clean an archetype row and expose only its descriptive fields."""
        cleaned = {
            key: value
            for key, value in archetype.items()
            if key not in {"embedding", "example_decks"}
        }
        cleaned["example_deck_count"] = len(archetype.get("example_decks") or [])
        return cleaned

    def _clean_archetype_result_with_examples(self, archetype: dict) -> dict:
        """Clean an archetype row while preserving compact example deck data."""
        cleaned = self._clean_archetype_result(archetype)
        cleaned["example_decks"] = archetype.get("example_decks") or []
        return cleaned

    def _normalize_card_ids(self, card_ids: list[int] | None) -> list[int]:
        """Deduplicate card IDs while preserving order."""
        if not card_ids:
            return []
        normalized: list[int] = []
        seen: set[int] = set()
        for card_id in card_ids:
            normalized_id = int(card_id)
            if normalized_id in seen:
                continue
            seen.add(normalized_id)
            normalized.append(normalized_id)
        return normalized

    def _get_required_cards(
        self,
        card_ids: list[int],
        *,
        include_embeddings: bool = False,
    ) -> list[dict]:
        """Fetch cards by ID and raise if any requested card is missing."""
        cards: list[dict] = []
        for card_id in self._normalize_card_ids(card_ids):
            card = self.get_card(card_id, include_embeddings=include_embeddings)
            if card is None:
                raise ValueError(f"Card with ID {card_id} not found")
            cards.append(card)
        if not cards:
            raise ValueError("At least one core card ID is required")
        return cards

    def _extract_tag_values(self, cards: list[dict], key: str) -> set[str]:
        """Collect normalized list-valued tags from a card set."""
        values: set[str] = set()
        for card in cards:
            for value in card.get(key) or []:
                if value:
                    values.add(str(value).lower())
        return values

    def _search_archetype_rows(self, query: str, limit: int) -> list[dict]:
        """Return raw archetype rows for internal reranking workflows."""
        query_embedding = self._embed_archetype_query(query)
        return (
            self._archetypes_table
            .search(query_embedding, vector_column_name="embedding")
            .where("embedding IS NOT NULL")
            .limit(limit)
            .to_list()
        )

    def _compose_core_context(self, cards: list[dict]) -> str:
        """Summarize core cards into a compact text description."""
        segments = []
        for card in cards:
            roles = ", ".join(card.get("role_tags") or [])
            vibes = ", ".join(card.get("vibe_tags") or [])
            details = [card["name"]]
            if card.get("type"):
                details.append(str(card["type"]))
            if card.get("elixir") is not None:
                details.append(f"{card['elixir']} elixir")
            if roles:
                details.append(f"roles: {roles}")
            if vibes:
                details.append(f"vibes: {vibes}")
            segments.append("; ".join(details))
        return " | ".join(segments)

    def _distance_to_similarity(self, distance: float | None) -> float:
        """Convert a Lance distance into a bounded similarity-like score."""
        if distance is None:
            return 0.0
        return 1.0 / (1.0 + max(float(distance), 0.0))

    def _average_similarity_to_core(
        self,
        candidate_embedding: list[float] | None,
        core_cards: list[dict],
    ) -> float:
        """Average cosine similarity between one candidate and the current core."""
        if not candidate_embedding:
            return 0.0
        core_embeddings = [
            card["combined_embedding"]
            for card in core_cards
            if card.get("combined_embedding")
        ]
        if not core_embeddings:
            return 0.0
        candidate = np.array(candidate_embedding, dtype=np.float32)
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm == 0:
            return 0.0
        candidate = candidate / candidate_norm
        similarities: list[float] = []
        for embedding in core_embeddings:
            core = np.array(embedding, dtype=np.float32)
            core_norm = float(np.linalg.norm(core))
            if core_norm == 0:
                continue
            similarities.append(float(np.dot(candidate, core / core_norm)))
        if not similarities:
            return 0.0
        return float(np.mean(similarities))
    
    def search_cards(
        self,
        query: str,
        elixir_min: int | None = None,
        elixir_max: int | None = None,
        type: str | None = None,
        rarity: str | None = None,
        exclude_card_ids: list[int] | None = None,
        limit: int = 10,
        include_images: bool = False,
    ) -> list[dict]:
        """
        Search cards semantically with optional filters.
        
        Args:
            query: Natural language search query
            elixir_min: Minimum elixir cost filter
            elixir_max: Maximum elixir cost filter
            type: Card type filter (Troop, Spell, Building, Champion, Tower Troop)
            rarity: Card rarity filter (Common, Rare, Epic, Legendary, Champion)
            exclude_card_ids: Card IDs to exclude from the results
            limit: Maximum number of results to return (default: 10)
            
        Returns:
            List of matching cards with similarity scores (_distance field)
            
        Example:
            >>> tools.search_cards("fast cycle cards that counter air", elixir_max=4)
            [{'name': 'Bats', 'elixir': 2, '_distance': 0.123}, ...]
        """
        # Embed the query
        query_embedding = self._embed_query(query)
        
        # Start vector search
        search = self._cards_table.search(
            query_embedding,
            vector_column_name="combined_embedding",
        )
        
        # Build filter conditions
        # Note: type and rarity use LOWER() for case-insensitive matching
        filters = []
        if elixir_min is not None:
            filters.append(f"elixir >= {elixir_min}")
        if elixir_max is not None:
            filters.append(f"elixir <= {elixir_max}")
        if type is not None:
            filters.append(f"LOWER(type) = '{type.lower()}'")
        if rarity is not None:
            filters.append(f"LOWER(rarity) = '{rarity.lower()}'")
        normalized_excludes = self._normalize_card_ids(exclude_card_ids)
        if normalized_excludes:
            excluded_ids = ", ".join(str(card_id) for card_id in normalized_excludes)
            filters.append(f"id NOT IN ({excluded_ids})")
        
        # Apply filters if any
        if filters:
            where_clause = " AND ".join(filters)
            search = search.where(where_clause)
        
        # Execute search
        results = search.limit(limit).to_list()
        
        # Clean results
        return [self._clean_card_result(card, include_image=include_images) for card in results]

    def search_archetypes(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Search archetypes semantically using the archetype embedding space.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return (default: 5)

        Returns:
            List of matching archetypes with descriptive fields and similarity
            scores (`_distance` field).
        """
        results = self._search_archetype_rows(query=query, limit=limit)
        return [self._clean_archetype_result(archetype) for archetype in results]

    def get_archetype(self, archetype_id: str) -> dict | None:
        """
        Fetch an archetype by ID without exposing example deck lists.

        Args:
            archetype_id: Unique archetype identifier

        Returns:
            Archetype dictionary or None if not found
        """
        escaped_archetype_id = self._escape_sql_string(archetype_id)
        zero_vector = [0.0] * ARCHETYPE_EMBEDDING_DIMS
        results = (
            self._archetypes_table
            .search(zero_vector, vector_column_name="embedding")
            .where(f"id = '{escaped_archetype_id}'")
            .limit(1)
            .to_list()
        )
        if not results:
            return None
        return self._clean_archetype_result(results[0])

    def select_archetype_for_core(
        self,
        user_request: str,
        core_card_ids: list[int],
        limit: int = 5,
    ) -> list[dict]:
        """
        Rank archetypes using both the user's request and the current core cards.

        This lets the agent commit to the strongest candidate cards first and
        then select the archetype shell that best fits those cards.
        """
        core_cards = self._get_required_cards(core_card_ids)
        core_roles = self._extract_roles(core_cards)
        core_vibes = self._extract_tag_values(core_cards, "vibe_tags")
        core_card_id_set = set(self._normalize_card_ids(core_card_ids))
        fused_query = (
            f"User request: {user_request}\n"
            f"Current core cards: {self._compose_core_context(core_cards)}\n"
            "Choose the Clash Royale archetype that best supports these cards."
        )
        raw_results = self._search_archetype_rows(
            query=fused_query,
            limit=max(limit * 3, limit),
        )

        ranked_results: list[dict] = []
        for archetype in raw_results:
            cleaned = self._clean_archetype_result(archetype)
            example_decks = archetype.get("example_decks") or []
            archetype_tags = {tag.lower() for tag in archetype.get("tags") or []}
            archetype_vibes = {
                vibe.lower() for vibe in archetype.get("playstyle_vibes") or []
            }

            semantic_score = self._distance_to_similarity(archetype.get("_distance"))
            vibe_overlap = len(core_vibes & archetype_vibes) / max(
                len(core_vibes | archetype_vibes),
                1,
            )

            example_overlap = 0.0
            if example_decks:
                example_overlap = max(
                    len(core_card_id_set & set(deck.get("battle_card_ids") or []))
                    / max(len(core_card_id_set), 1)
                    for deck in example_decks
                )

            role_alignment = 0.0
            if core_roles:
                matched_tags = {
                    role for role in core_roles
                    if any(role_part in archetype_tags for role_part in role.split())
                }
                role_alignment = len(matched_tags) / len(core_roles)

            fit_score = (
                semantic_score * 0.55
                + example_overlap * 0.3
                + vibe_overlap * 0.1
                + role_alignment * 0.05
            )

            reasons: list[str] = []
            if semantic_score > 0:
                reasons.append("semantic match to the request and core cards")
            if example_overlap > 0:
                reasons.append("example deck overlap with the chosen core")
            if vibe_overlap > 0:
                reasons.append("shared vibe tags between the core and archetype")
            if role_alignment > 0:
                reasons.append("tag alignment with current core roles")

            cleaned["fit_score"] = round(fit_score, 3)
            cleaned["fit_reasons"] = reasons
            ranked_results.append(cleaned)

        ranked_results.sort(
            key=lambda result: (
                result.get("fit_score", 0.0),
                result.get("meta_strength") or 0.0,
            ),
            reverse=True,
        )
        return ranked_results[:limit]

    def search_complementary_cards(
        self,
        user_request: str,
        archetype_id: str,
        core_card_ids: list[int],
        role_hint: str | None = None,
        elixir_min: int | None = None,
        elixir_max: int | None = None,
        type: str | None = None,
        rarity: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search for cards that complement the current core inside a chosen shell.

        Results are semantically searched first, then reranked to reward missing
        role coverage, similarity to the current core, and archetype vibe fit.
        """
        core_cards = self._get_required_cards(core_card_ids, include_embeddings=True)
        archetype = self.get_archetype(archetype_id=archetype_id)
        if archetype is None:
            raise ValueError(f"Archetype with ID {archetype_id} not found")

        core_roles = self._extract_roles(core_cards)
        missing_roles = REQUIRED_ROLES - core_roles
        core_vibes = self._extract_tag_values(core_cards, "vibe_tags")
        archetype_vibes = {
            vibe.lower() for vibe in archetype.get("playstyle_vibes") or []
        }
        query_parts = [
            f"user request: {user_request}",
            f"archetype: {archetype['name']}",
            f"archetype description: {archetype.get('description', '')}",
            f"current core cards: {self._compose_core_context(core_cards)}",
        ]
        if role_hint:
            query_parts.append(f"needed role: {role_hint}")
        if missing_roles:
            query_parts.append(
                "missing roles: " + ", ".join(sorted(missing_roles))
            )
        query_parts.append(
            "Find the best complementary Clash Royale card for this deck core."
        )
        search_query = ". ".join(part for part in query_parts if part)

        candidates = self.search_cards(
            query=search_query,
            elixir_min=elixir_min,
            elixir_max=elixir_max,
            type=type,
            rarity=rarity,
            exclude_card_ids=core_card_ids,
            limit=max(limit * 4, limit),
        )

        reranked: list[dict] = []
        role_hint_lower = (role_hint or "").lower()
        core_has_spell = any(card.get("type") == "Spell" for card in core_cards)

        for candidate in candidates:
            full_candidate = self.get_card(
                candidate["id"],
                include_embeddings=True,
            )
            if full_candidate is None:
                continue

            candidate_roles = {
                role.lower() for role in full_candidate.get("role_tags") or []
            }
            candidate_vibes = {
                vibe.lower() for vibe in full_candidate.get("vibe_tags") or []
            }
            semantic_score = self._distance_to_similarity(candidate.get("_distance"))
            core_similarity = self._average_similarity_to_core(
                full_candidate.get("combined_embedding"),
                core_cards,
            )
            missing_role_score = len(candidate_roles & missing_roles) / max(
                len(missing_roles),
                1,
            )
            archetype_vibe_score = len(candidate_vibes & (archetype_vibes | core_vibes)) / max(
                len(candidate_vibes | archetype_vibes | core_vibes),
                1,
            )

            role_hint_score = 0.0
            if role_hint_lower:
                searchable_text = " ".join(
                    [
                        full_candidate.get("name", "").lower(),
                        full_candidate.get("description", "").lower(),
                        " ".join(candidate_roles),
                        " ".join(candidate_vibes),
                        str(full_candidate.get("type", "")).lower(),
                    ]
                )
                if role_hint_lower in searchable_text:
                    role_hint_score = 1.0
                else:
                    hint_words = {
                        word for word in role_hint_lower.split() if len(word) > 2
                    }
                    if hint_words:
                        role_hint_score = len(
                            [word for word in hint_words if word in searchable_text]
                        ) / len(hint_words)

            utility_bonus = 0.0
            if not core_has_spell and full_candidate.get("type") == "Spell":
                utility_bonus += 0.6
            if "anti-air" in missing_roles and "anti-air" in candidate_roles:
                utility_bonus += 0.4

            fit_score = (
                semantic_score * 0.4
                + core_similarity * 0.25
                + missing_role_score * 0.2
                + archetype_vibe_score * 0.1
                + role_hint_score * 0.05
                + utility_bonus * 0.05
            )

            fit_reasons: list[str] = []
            if semantic_score > 0:
                fit_reasons.append("semantic match to the request and archetype")
            if core_similarity > 0.6:
                fit_reasons.append("strong embedding synergy with the current core")
            if missing_role_score > 0:
                fit_reasons.append("helps cover currently missing required roles")
            if archetype_vibe_score > 0:
                fit_reasons.append("matches the archetype or core vibe")
            if role_hint_score > 0:
                fit_reasons.append(f"fits the requested role hint: {role_hint}")

            reranked_candidate = self._clean_card_result(full_candidate)
            reranked_candidate["_distance"] = candidate.get("_distance")
            reranked_candidate["fit_score"] = round(fit_score, 3)
            reranked_candidate["fit_reasons"] = fit_reasons
            reranked.append(reranked_candidate)

        reranked.sort(
            key=lambda card: (card.get("fit_score", 0.0), -(card.get("_distance") or 0.0)),
            reverse=True,
        )
        return reranked[:limit]
    
    def similar_cards(
        self,
        card_id: int,
        limit: int = 5,
        include_images: bool = False,
    ) -> list[dict]:
        """
        Find cards similar to a given card.
        
        Args:
            card_id: The ID of the card to find similar cards for
            limit: Maximum number of similar cards to return (default: 5)
            
        Returns:
            List of similar cards with similarity scores (_distance field)
            
        Raises:
            ValueError: If the card_id is not found
            
        Example:
            >>> tools.similar_cards(card_id=26000000, limit=5)
            [{'name': 'Battle Ram', 'elixir': 4, '_distance': 0.089}, ...]
        """
        # First, get the source card to retrieve its embedding
        source_card = self.get_card(card_id, include_embeddings=True)
        
        if source_card is None:
            raise ValueError(f"Card with ID {card_id} not found")
        
        # Get the combined embedding
        embedding = source_card.get("combined_embedding")
        if embedding is None:
            raise ValueError(f"Card {card_id} does not have a combined_embedding")
        
        # Search for similar cards (request limit+1 to account for excluding self)
        results = self._cards_table.search(
            embedding,
            vector_column_name="combined_embedding",
        ).limit(limit + 1).to_list()
        
        # Filter out the source card and clean results
        similar = []
        for card in results:
            if card["id"] != card_id:
                similar.append(self._clean_card_result(card, include_image=include_images))
                if len(similar) >= limit:
                    break
        
        return similar
    
    def get_card(
        self,
        card_id: int,
        include_embeddings: bool = False,
        include_image: bool = False,
    ) -> dict | None:
        """
        Fetch a card by ID.
        
        Args:
            card_id: The unique card ID
            include_embeddings: Whether to include embedding fields (default: False)
            
        Returns:
            Card dictionary or None if not found
            
        Example:
            >>> tools.get_card(card_id=26000000)
            {'id': 26000000, 'name': 'Hog Rider', 'elixir': 4, ...}
        """
        # Query by ID using vector search with a zero vector and ID filter
        # LanceDB Cloud requires vector queries, so we use a dummy vector
        # and rely entirely on the filter to get the exact card
        zero_vector = [0.0] * COMBINED_EMBEDDING_DIMS
        
        results = (
            self._cards_table
            .search(zero_vector, vector_column_name="combined_embedding")
            .where(f"id = {card_id}")
            .limit(1)
            .to_list()
        )
        
        if not results:
            return None
        
        card = results[0]
        
        if include_embeddings:
            return card
        else:
            return self._clean_card_result(card, include_image=include_image)
    
    def _calculate_synergy_score(self, embeddings: list[list[float]]) -> float:
        """
        Calculate average pairwise cosine similarity between card embeddings.
        
        Higher scores indicate cards that semantically relate well together.
        
        Args:
            embeddings: List of card embeddings
            
        Returns:
            Average pairwise similarity score (0-1)
        """
        if len(embeddings) < 2:
            return 1.0
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings_array / norms
        
        # Calculate pairwise cosine similarities
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Get upper triangle (excluding diagonal)
        n = len(embeddings)
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_similarities = similarity_matrix[upper_triangle_indices]
        
        # Return average similarity
        return float(np.mean(pairwise_similarities))
    
    def _extract_roles(self, cards: list[dict]) -> set[str]:
        """
        Extract all roles covered by a set of cards.
        
        Args:
            cards: List of card dictionaries
            
        Returns:
            Set of role strings
        """
        roles = set()
        for card in cards:
            role_tags = card.get("role_tags") or []
            for role in role_tags:
                if role:
                    roles.add(role.lower())
        return roles
    
    def _calculate_meta_strength(
        self,
        avg_elixir: float,
        role_coverage: set[str],
        missing_roles: set[str],
        synergy_score: float,
        type_distribution: dict[str, int],
    ) -> float:
        """
        Calculate a heuristic meta strength score (0-100).
        
        Components:
        - Role coverage: 40 points max
        - Synergy score: 30 points max
        - Elixir balance: 20 points max
        - Type diversity: 10 points max
        
        Args:
            avg_elixir: Average elixir cost
            role_coverage: Set of roles covered
            missing_roles: Set of missing important roles
            synergy_score: Pairwise embedding similarity
            type_distribution: Count of cards by type
            
        Returns:
            Meta strength score (0-100)
        """
        score = 0.0
        
        # Role coverage (40 points)
        # Each required role covered adds points
        role_points = (len(REQUIRED_ROLES) - len(missing_roles)) / len(REQUIRED_ROLES) * 40
        score += role_points
        
        # Synergy score (30 points)
        # Synergy is already 0-1, scale to 30 points
        score += synergy_score * 30
        
        # Elixir balance (20 points)
        # Optimal is around 3.0-3.5 average elixir
        optimal_elixir = 3.25
        elixir_deviation = abs(avg_elixir - optimal_elixir)
        elixir_points = max(0, 20 - (elixir_deviation * 10))
        score += elixir_points
        
        # Type diversity (10 points)
        # Having a mix of troops, spells, and buildings is good
        num_types = len([t for t, count in type_distribution.items() if count > 0])
        type_points = min(num_types / 3 * 10, 10)  # Max 10 points for 3+ types
        score += type_points
        
        return min(100, max(0, score))
    
    def score_deck(
        self,
        battle_card_ids: list[int],
        tower_card_id: int | None = None,
        include_images: bool = False,
    ) -> DeckScore:
        """
        Evaluate a deck for balance and constraints.
        
        Analyzes the deck across multiple dimensions:
        - Average elixir cost
        - Type distribution (Troop, Spell, Building, etc.)
        - Rarity distribution
        - Role coverage (win condition, anti-air, splash, tank, cycle)
        - Synergy score (embedding-based)
        - Meta strength heuristic
        
        Args:
            battle_card_ids: List of battle card IDs in the deck (should be 8)
            tower_card_id: Tower troop card ID for the deck
            
        Returns:
            DeckScore object with comprehensive evaluation
            
        Raises:
            ValueError: If any card_id is not found
            
        Example:
            >>> score = tools.score_deck(
            ...     battle_card_ids=[26000000, 26000001, ...],
            ...     tower_card_id=28000000,
            ... )
            >>> print(f"Avg elixir: {score.avg_elixir}")
            >>> print(f"Meta strength: {score.meta_strength}")
            >>> for warning in score.balance_warnings:
            ...     print(f"Warning: {warning}")
        """
        # Fetch all cards
        battle_cards = []
        embeddings = []
        
        for card_id in battle_card_ids:
            card = self.get_card(card_id, include_embeddings=True)
            if card is None:
                raise ValueError(f"Card with ID {card_id} not found")
            battle_cards.append(card)
            
            # Collect embeddings for synergy calculation
            if card.get("combined_embedding"):
                embeddings.append(card["combined_embedding"])

        tower_card = None
        if tower_card_id is not None:
            tower_card = self.get_card(tower_card_id, include_embeddings=True)
            if tower_card is None:
                raise ValueError(f"Card with ID {tower_card_id} not found")
        
        # Calculate average elixir from the 8 battle cards only.
        total_elixir = sum(card.get("elixir", 0) for card in battle_cards)
        avg_elixir = total_elixir / len(battle_cards) if battle_cards else 0
        
        # Calculate type distribution
        type_distribution: dict[str, int] = {}
        for card in battle_cards:
            card_type = card.get("type", "Unknown")
            type_distribution[card_type] = type_distribution.get(card_type, 0) + 1
        
        # Calculate rarity distribution
        rarity_distribution: dict[str, int] = {}
        for card in battle_cards:
            rarity = card.get("rarity", "Unknown")
            rarity_distribution[rarity] = rarity_distribution.get(rarity, 0) + 1
        
        # Extract roles
        role_coverage = self._extract_roles(battle_cards)
        missing_roles = REQUIRED_ROLES - role_coverage
        
        # Calculate synergy score
        synergy_score = self._calculate_synergy_score(embeddings) if embeddings else 0.0
        
        # Generate balance warnings
        balance_warnings = []
        
        # Deck size warning
        if len(battle_card_ids) != DECK_SIZE:
            balance_warnings.append(
                f"Deck has {len(battle_card_ids)} battle cards (expected {DECK_SIZE})"
            )
        
        # Duplicate cards warning
        if len(battle_card_ids) != len(set(battle_card_ids)):
            balance_warnings.append("Deck contains duplicate cards")

        if tower_card_id is None:
            balance_warnings.append("Deck is missing a tower troop")
        elif tower_card_id in set(battle_card_ids):
            balance_warnings.append("Tower troop is duplicated in battle cards")

        if any(card.get("type") == "Tower Troop" for card in battle_cards):
            balance_warnings.append("Battle cards should not include tower troops")

        if tower_card is not None and tower_card.get("type") != "Tower Troop":
            balance_warnings.append(
                f"Selected tower card is not a tower troop ({tower_card.get('type', 'Unknown')})"
            )
        
        # Elixir warnings
        if avg_elixir > EXPENSIVE_DECK_THRESHOLD:
            balance_warnings.append(f"Deck is expensive (avg {avg_elixir:.1f} elixir)")
        elif avg_elixir < CHEAP_DECK_THRESHOLD:
            balance_warnings.append(f"Deck may lack damage potential (avg {avg_elixir:.1f} elixir)")
        
        # Missing role warnings
        for role in missing_roles:
            balance_warnings.append(f"Missing role: {role}")
        
        # No spells warning
        if type_distribution.get("Spell", 0) == 0:
            balance_warnings.append("No spells in deck")
        
        # Multiple champions warning (only 1 allowed in Clash Royale)
        if type_distribution.get("Champion", 0) > 1:
            balance_warnings.append("Multiple champions (only 1 allowed)")
        
        # Calculate meta strength
        meta_strength = self._calculate_meta_strength(
            avg_elixir=avg_elixir,
            role_coverage=role_coverage,
            missing_roles=missing_roles,
            synergy_score=synergy_score,
            type_distribution=type_distribution,
        )
        
        # Clean cards for output (remove embeddings)
        cleaned_battle_cards = [
            self._clean_card_result(card, include_image=include_images) for card in battle_cards
        ]
        cleaned_tower_card = (
            self._clean_card_result(tower_card, include_image=include_images)
            if tower_card is not None
            else None
        )
        
        return DeckScore(
            avg_elixir=round(avg_elixir, 2),
            type_distribution=type_distribution,
            rarity_distribution=rarity_distribution,
            role_coverage=role_coverage,
            missing_roles=missing_roles,
            synergy_score=round(synergy_score, 3),
            balance_warnings=balance_warnings,
            meta_strength=round(meta_strength, 1),
            battle_cards=cleaned_battle_cards,
            tower_card=cleaned_tower_card,
        )


__all__ = [
    "CardTools",
    "DeckScore",
]
