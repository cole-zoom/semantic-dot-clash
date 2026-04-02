"""
LLM Agent for building Clash Royale decks.

Implements an agentic loop that uses OpenAI function calling to orchestrate
CardTools methods for semantic card search, archetype retrieval, similarity
matching, and deck scoring.

Environment Variables:
    LANCE_URI: LanceDB Cloud URI
    LANCE_KEY: LanceDB Cloud API key
    OPENAI_API_KEY: OpenAI API key

Usage:
    from semantic_dot_clash import DeckAgent

    agent = DeckAgent()
    result = agent.build("Make me a fast cycle deck under 3.0 elixir")
    print(result)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from semantic_dot_clash.chat_session import ChatSession, DeckSnapshot
from semantic_dot_clash.memory import append_turn, build_chat_messages
from semantic_dot_clash.tools import CardTools

load_dotenv()

# Default model for deck building
DEFAULT_MODEL = "gpt-5.2"

# Maximum iterations to prevent infinite loops
DEFAULT_MAX_ITERATIONS = 10

# Tool schemas for OpenAI function calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_archetypes",
            "description": "Search archetypes semantically using the archetype embedding space. Use this first to find the archetype whose playstyle, vibe, and feel best match the user's request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing the desired archetype (e.g., 'fast annoying control deck', 'air beatdown', 'bridge spam with pressure')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_archetype",
            "description": "Fetch full details for a specific archetype by ID. Use this when you want to inspect the chosen archetype's description, tags, and vibes more closely before assembling the final deck.",
            "parameters": {
                "type": "object",
                "properties": {
                    "archetype_id": {
                        "type": "string",
                        "description": "The unique archetype ID"
                    }
                },
                "required": ["archetype_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_cards",
            "description": "Search for cards semantically with optional filters. Use this to find cards matching a role, playstyle, or archetype.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'cheap cycle cards', 'anti-air splash damage', 'fast aggressive win condition')"
                    },
                    "elixir_min": {
                        "type": "integer",
                        "description": "Minimum elixir cost filter"
                    },
                    "elixir_max": {
                        "type": "integer",
                        "description": "Maximum elixir cost filter"
                    },
                    "type": {
                        "type": "string",
                        "description": "Card type filter",
                        "enum": ["Troop", "Spell", "Building", "Champion", "Tower Troop"]
                    },
                    "rarity": {
                        "type": "string",
                        "description": "Card rarity filter",
                        "enum": ["Common", "Rare", "Epic", "Legendary", "Champion"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "similar_cards",
            "description": "Find cards similar to a given card. Use this to find alternatives when swapping cards (e.g., cheaper version, different rarity, same role).",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "integer",
                        "description": "The ID of the reference card to find similar cards for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of similar cards to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["card_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_card",
            "description": "Fetch complete details for a specific card by ID. Use this to verify exact stats before including a card, or to explain a card's role.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "integer",
                        "description": "The unique card ID"
                    }
                },
                "required": ["card_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "score_deck",
            "description": "Evaluate a deck for balance, legality, and constraint satisfaction. You MUST call this before finalizing any deck.",
            "parameters": {
                "type": "object",
                "properties": {
                    "battle_card_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of exactly 8 battle card IDs representing the deck"
                    },
                    "tower_card_id": {
                        "type": "integer",
                        "description": "Tower troop card ID for the deck"
                    }
                },
                "required": ["battle_card_ids", "tower_card_id"]
            }
        }
    }
]


@dataclass
class DeckResult:
    """
    Result of a deck building request.
    
    Attributes:
        success: Whether a valid deck was built
        deck: List of the 8 battle card dictionaries in the final deck
        tower_card: Tower troop dictionary in the final deck
        avg_elixir: Average elixir cost of the 8 battle cards
        response: Full text response from the agent
        iterations: Number of loop iterations used
        error: Error message if success is False
    deck_snapshot: Structured deck snapshot for the current turn
    used_summary: Whether summarized history was injected into the prompt
    turn_index: Completed turn count for chat sessions
    memory_updates: Structured memory metadata for chat sessions
    """
    success: bool
    deck: list[dict] = field(default_factory=list)
    tower_card: dict | None = None
    avg_elixir: float = 0.0
    response: str = ""
    iterations: int = 0
    error: str | None = None
    deck_snapshot: DeckSnapshot | None = None
    used_summary: bool = False
    turn_index: int | None = None
    memory_updates: dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if not self.success:
            return f"Failed to build deck: {self.error}"
        return self.response


class DeckAgent:
    """
    LLM agent for building Clash Royale decks.
    
    Uses an agentic loop with OpenAI function calling to:
    1. Search for archetypes semantically
    2. Search for cards semantically
    3. Find similar cards for swaps
    4. Validate decks with score_deck
    5. Present final deck with strategy
    
    Args:
        model: OpenAI model to use (default: gpt-4o)
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        lance_uri: LanceDB Cloud URI (defaults to LANCE_URI env var)
        lance_key: LanceDB Cloud API key (defaults to LANCE_KEY env var)
    
    Example:
        >>> agent = DeckAgent()
        >>> result = agent.build("Make me a fast Hog Rider cycle deck")
        >>> print(result)
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        openai_api_key: str | None = None,
        lance_uri: str | None = None,
        lance_key: str | None = None,
        verbose: bool = False,
    ):
        """Initialize the deck building agent."""
        self.model = model
        self.verbose = verbose
        
        # Get OpenAI API key
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable or openai_api_key parameter is required")
        
        # Initialize OpenAI client
        self._openai = OpenAI(api_key=api_key)
        
        # Initialize CardTools for tool execution
        self._tools = CardTools(
            lance_uri=lance_uri,
            lance_key=lance_key,
            openai_api_key=api_key,
        )
        
        # Load system prompt
        self._system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """
        Load the system prompt from system_prompt.txt.
        
        Returns:
            System prompt string
        """
        # Look for system_prompt.txt in several locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "system_prompt.txt",  # Project root
            Path(__file__).parent / "system_prompt.txt",  # Package directory
            Path.cwd() / "system_prompt.txt",  # Current working directory
        ]
        
        for path in possible_paths:
            if path.exists():
                return path.read_text()
        
        # Fallback to embedded minimal prompt
        return """You are a Clash Royale deck-building agent. Build exactly 8 battle cards plus 1 tower troop using only the tools provided.

Rules:
- Only use cards returned by tools - never invent cards
- Always call score_deck before presenting a final deck
- If constraints are violated, revise and try again
- Present the final deck with card names, elixir costs, and strategy"""
    
    def _execute_tool(self, name: str, arguments: str) -> Any:
        """
        Execute a tool by name with the given arguments.
        
        Args:
            name: Tool name (search_archetypes, get_archetype, search_cards,
                similar_cards, get_card, score_deck)
            arguments: JSON string of arguments
            
        Returns:
            Tool result (dict, list, or DeckScore)
        """
        args = json.loads(arguments)
        
        if name == "search_archetypes":
            return self._tools.search_archetypes(
                query=args["query"],
                limit=args.get("limit", 5),
            )

        elif name == "get_archetype":
            archetype = self._tools.get_archetype(archetype_id=args["archetype_id"])
            if archetype is None:
                return {"error": f"Archetype with ID {args['archetype_id']} not found"}
            return archetype

        elif name == "search_cards":
            return self._tools.search_cards(
                query=args["query"],
                elixir_min=args.get("elixir_min"),
                elixir_max=args.get("elixir_max"),
                type=args.get("type"),
                rarity=args.get("rarity"),
                limit=args.get("limit", 10),
            )
        
        elif name == "similar_cards":
            return self._tools.similar_cards(
                card_id=args["card_id"],
                limit=args.get("limit", 5),
            )
        
        elif name == "get_card":
            card = self._tools.get_card(card_id=args["card_id"])
            if card is None:
                return {"error": f"Card with ID {args['card_id']} not found"}
            return card
        
        elif name == "score_deck":
            try:
                score = self._tools.score_deck(
                    battle_card_ids=args["battle_card_ids"],
                    tower_card_id=args["tower_card_id"],
                )
                return score.to_dict()
            except ValueError as e:
                return {"error": str(e)}
        
        else:
            return {"error": f"Unknown tool: {name}"}
    
    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[DeckAgent] {message}")

    def _snapshot_from_tool_result(
        self,
        tool_name: str,
        tool_result: Any,
    ) -> DeckSnapshot | None:
        """Return a structured deck snapshot for a valid ``score_deck`` result."""
        if tool_name != "score_deck" or not isinstance(tool_result, dict):
            return None
        if tool_result.get("error") or "avg_elixir" not in tool_result:
            return None
        return DeckSnapshot.from_score_payload(tool_result)

    def _build_result(
        self,
        *,
        success: bool,
        response: str = "",
        iterations: int = 0,
        error: str | None = None,
        deck_snapshot: DeckSnapshot | None = None,
        used_summary: bool = False,
        turn_index: int | None = None,
        memory_updates: dict[str, Any] | None = None,
    ) -> DeckResult:
        """Create a ``DeckResult`` with backward-compatible fields populated."""
        return DeckResult(
            success=success,
            deck=list(deck_snapshot.battle_cards) if deck_snapshot else [],
            tower_card=deck_snapshot.tower_card if deck_snapshot else None,
            avg_elixir=deck_snapshot.avg_elixir if deck_snapshot else 0.0,
            response=response,
            iterations=iterations,
            error=error,
            deck_snapshot=deck_snapshot,
            used_summary=used_summary,
            turn_index=turn_index,
            memory_updates=memory_updates or {},
        )

    def _run_agent_loop(
        self,
        *,
        messages: list[dict],
        max_iterations: int,
        used_summary: bool = False,
    ) -> DeckResult:
        """Run the shared OpenAI tool-calling loop against a prepared message list."""
        latest_deck_snapshot: DeckSnapshot | None = None

        for step in range(max_iterations):
            self._log(f"Iteration {step + 1}/{max_iterations}")

            try:
                response = self._openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except Exception as exc:
                return self._build_result(
                    success=False,
                    error=f"OpenAI API error: {exc}",
                    iterations=step + 1,
                    deck_snapshot=latest_deck_snapshot,
                    used_summary=used_summary,
                )

            msg = response.choices[0].message

            if msg.tool_calls:
                self._log(f"Tool calls: {[tc.function.name for tc in msg.tool_calls]}")

                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    self._log(f"Executing {tool_name}...")

                    try:
                        tool_result = self._execute_tool(tool_name, tool_args)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        tool_result = {"error": str(exc)}

                    snapshot = self._snapshot_from_tool_result(tool_name, tool_result)
                    if snapshot is not None:
                        latest_deck_snapshot = snapshot

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, default=str),
                        }
                    )

                continue

            self._log("Final answer received")
            return self._build_result(
                success=True,
                response=msg.content or "",
                iterations=step + 1,
                deck_snapshot=latest_deck_snapshot,
                used_summary=used_summary,
            )

        self._log("Max iterations reached")

        last_content = ""
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                last_content = message["content"]
                break

        return self._build_result(
            success=False,
            response=last_content,
            error=f"Max iterations ({max_iterations}) reached without final answer",
            iterations=max_iterations,
            deck_snapshot=latest_deck_snapshot,
            used_summary=used_summary,
        )

    def build(
        self,
        user_request: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> DeckResult:
        """
        Build a deck based on the user's request.

        This is the main agentic loop that:
        1. Sends the request to the LLM with tool schemas
        2. Executes any tool calls the LLM requests
        3. Continues until the LLM provides a final answer or max iterations reached

        Args:
            user_request: Natural language deck request (e.g., "Make me a fast cycle deck")
            max_iterations: Maximum number of loop iterations (default: 10)

        Returns:
            DeckResult with the built deck or error information
        """
        self._log(f"Starting deck build: {user_request[:50]}...")
        return self._run_agent_loop(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_request},
            ],
            max_iterations=max_iterations,
        )

    def chat(
        self,
        *,
        session: ChatSession,
        user_message: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> DeckResult:
        """Run a single chat turn against an ephemeral session."""
        self._log(
            f"Starting chat turn {session.turn_count + 1} for session {session.session_id}"
        )
        messages, used_summary = build_chat_messages(
            system_prompt=self._system_prompt,
            session=session,
            user_message=user_message,
        )
        result = self._run_agent_loop(
            messages=messages,
            max_iterations=max_iterations,
            used_summary=used_summary,
        )
        if not result.success:
            return result

        memory_updates = append_turn(
            session,
            user_message=user_message,
            assistant_message=result.response,
            deck_snapshot=result.deck_snapshot,
        )
        result.turn_index = session.turn_count
        result.memory_updates = memory_updates
        return result

    def hydrate_deck_snapshot(
        self,
        deck_snapshot: DeckSnapshot | None,
        *,
        include_image: bool = True,
    ) -> tuple[list[dict], dict | None]:
        """Hydrate a structured deck snapshot back to full card payloads."""
        if deck_snapshot is None:
            return [], None

        cards: list[dict] = []
        for card_id in deck_snapshot.battle_card_ids:
            card = self._tools.get_card(card_id=card_id, include_image=include_image)
            if card:
                cards.append(card)

        tower_card = None
        if deck_snapshot.tower_card and deck_snapshot.tower_card.get("id") is not None:
            tower_card = self._tools.get_card(
                card_id=deck_snapshot.tower_card["id"],
                include_image=include_image,
            )
        return cards, tower_card


__all__ = [
    "DeckAgent",
    "DeckResult",
    "TOOL_SCHEMAS",
]
