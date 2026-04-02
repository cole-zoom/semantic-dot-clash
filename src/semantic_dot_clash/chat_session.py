"""Ephemeral chat session models and storage."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class DeckSnapshot:
    """Structured snapshot of the current validated deck."""

    battle_cards: list[dict] = field(default_factory=list)
    tower_card: dict | None = None
    avg_elixir: float = 0.0

    @classmethod
    def from_score_payload(cls, payload: dict) -> "DeckSnapshot":
        """Build a deck snapshot from a ``score_deck`` tool payload."""
        return cls(
            battle_cards=[
                {
                    "id": card.get("id"),
                    "name": card.get("name"),
                    "elixir": card.get("elixir"),
                }
                for card in payload.get("battle_cards") or payload.get("cards") or []
                if card.get("id") is not None
            ],
            tower_card=payload.get("tower_card"),
            avg_elixir=float(payload.get("avg_elixir") or 0.0),
        )

    @property
    def battle_card_ids(self) -> list[int]:
        """Return the battle card IDs from the snapshot."""
        return [
            int(card["id"])
            for card in self.battle_cards
            if card.get("id") is not None
        ]


@dataclass
class ChatTurn:
    """Single user/assistant exchange in a chat session."""

    user_message: str
    assistant_message: str
    deck_snapshot: DeckSnapshot | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    """Ephemeral chat session used by the web API and CLI chat."""

    session_id: str
    turns: list[ChatTurn] = field(default_factory=list)
    rolling_summary: str = ""
    preference_notes: list[str] = field(default_factory=list)
    latest_deck_snapshot: DeckSnapshot | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Refresh the session activity timestamp."""
        self.updated_at = time.time()

    @property
    def turn_count(self) -> int:
        """Return the number of completed turns."""
        return len(self.turns)


class InMemoryChatSessionStore:
    """Thread-safe in-memory session store with simple TTL eviction."""

    def __init__(self, ttl_seconds: int = 60 * 60 * 4) -> None:
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def _prune_expired_locked(self) -> None:
        now = time.time()
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.updated_at > self._ttl_seconds
        ]
        for session_id in expired_ids:
            self._sessions.pop(session_id, None)

    def get_or_create(self, session_id: str) -> ChatSession:
        """Return the session, creating it if needed."""
        with self._lock:
            self._prune_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                session = ChatSession(session_id=session_id)
                self._sessions[session_id] = session
            session.touch()
            return session

    def get(self, session_id: str) -> ChatSession | None:
        """Return a session without creating it."""
        with self._lock:
            self._prune_expired_locked()
            session = self._sessions.get(session_id)
            if session is not None:
                session.touch()
            return session

    def reset(self, session_id: str) -> None:
        """Delete a session if it exists."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def prune_expired(self) -> None:
        """Remove all expired sessions."""
        with self._lock:
            self._prune_expired_locked()
