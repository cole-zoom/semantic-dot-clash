"""Helpers for building and shrinking chat memory."""

from __future__ import annotations

from semantic_dot_clash.chat_session import ChatSession, ChatTurn, DeckSnapshot

MAX_RECENT_TURNS = 3
MAX_PREFERENCE_NOTES = 6
MAX_NOTE_LENGTH = 180
MAX_SUMMARY_CHANGES = 4


def _clip(text: str, max_length: int = MAX_NOTE_LENGTH) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def update_preference_notes(session: ChatSession, user_message: str) -> None:
    """Persist a bounded list of user preference notes."""
    note = _clip(user_message)
    if not note:
        return
    if note in session.preference_notes:
        session.preference_notes.remove(note)
    session.preference_notes.append(note)
    session.preference_notes = session.preference_notes[-MAX_PREFERENCE_NOTES:]


def build_rolling_summary(session: ChatSession) -> str:
    """Build a compact summary from the current session state."""
    parts: list[str] = []

    if session.preference_notes:
        parts.append("User preferences and requested changes:")
        parts.extend(f"- {note}" for note in session.preference_notes)

    if session.latest_deck_snapshot and session.latest_deck_snapshot.battle_cards:
        deck_cards = ", ".join(
            card.get("name", "Unknown")
            for card in session.latest_deck_snapshot.battle_cards
        )
        tower_name = (
            session.latest_deck_snapshot.tower_card.get("name", "Unknown")
            if session.latest_deck_snapshot.tower_card
            else "Unknown"
        )
        parts.append(
            "Current validated deck: "
            f"{deck_cards}. Tower troop: {tower_name}. "
            f"Average elixir: {session.latest_deck_snapshot.avg_elixir:.2f}."
        )

    if session.turns:
        parts.append("Most recent accepted changes:")
        for turn in session.turns[-MAX_SUMMARY_CHANGES:]:
            parts.append(f"- {_clip(turn.user_message)}")

    return "\n".join(parts).strip()


def append_turn(
    session: ChatSession,
    user_message: str,
    assistant_message: str,
    deck_snapshot: DeckSnapshot | None,
) -> dict[str, object]:
    """Add a completed turn to the session and refresh its summary."""
    update_preference_notes(session, user_message)
    if deck_snapshot is not None:
        session.latest_deck_snapshot = deck_snapshot

    session.turns.append(
        ChatTurn(
            user_message=user_message,
            assistant_message=assistant_message,
            deck_snapshot=deck_snapshot,
        )
    )
    session.turns = session.turns[-MAX_RECENT_TURNS:]
    session.rolling_summary = build_rolling_summary(session)
    session.touch()
    return {
        "summary": session.rolling_summary,
        "turn_index": session.turn_count,
    }


def build_chat_messages(
    *,
    system_prompt: str,
    session: ChatSession,
    user_message: str,
) -> tuple[list[dict], bool]:
    """Build bounded prompt messages for the next chat turn."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    used_summary = bool(session.rolling_summary or session.latest_deck_snapshot)

    if used_summary:
        summary_lines = [
            "Conversation context for this thread.",
            "Use it to preserve continuity while still following the newest user request.",
        ]
        if session.rolling_summary:
            summary_lines.extend(["", session.rolling_summary])
        if session.latest_deck_snapshot and session.latest_deck_snapshot.battle_cards:
            battle_cards = "\n".join(
                f"- {card.get('name', 'Unknown')} ({card.get('elixir', '?')} elixir)"
                for card in session.latest_deck_snapshot.battle_cards
            )
            tower_name = (
                session.latest_deck_snapshot.tower_card.get("name", "Unknown")
                if session.latest_deck_snapshot.tower_card
                else "Unknown"
            )
            summary_lines.extend(
                [
                    "",
                    "Current validated deck snapshot:",
                    battle_cards,
                    f"Tower troop: {tower_name}",
                    f"Average elixir: {session.latest_deck_snapshot.avg_elixir:.2f}",
                ]
            )
        messages.append({"role": "system", "content": "\n".join(summary_lines).strip()})

    for turn in session.turns[-MAX_RECENT_TURNS:]:
        messages.append({"role": "user", "content": turn.user_message})
        messages.append({"role": "assistant", "content": turn.assistant_message})

    messages.append({"role": "user", "content": user_message})
    return messages, used_summary
