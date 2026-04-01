"""
CLI entry point for the Semantic Dot Clash deck builder.

Usage:
    python -m semantic_dot_clash "Make me a fast cycle deck"
    python -m semantic_dot_clash --chat
    python -m semantic_dot_clash --verbose "Build a Golem beatdown deck"
    python -m semantic_dot_clash --model gpt-4o-mini "Quick aggressive deck"

Environment Variables:
    LANCE_URI: LanceDB Cloud URI
    LANCE_KEY: LanceDB Cloud API key
    OPENAI_API_KEY: OpenAI API key
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from semantic_dot_clash.chat_session import ChatSession

load_dotenv()


def _print_snapshot(session: ChatSession) -> None:
    snapshot = session.latest_deck_snapshot
    if snapshot is None or not snapshot.battle_cards:
        print("[CLI] No validated deck in the current session yet.")
        return

    print("## Current Deck")
    for card in snapshot.battle_cards:
        print(f"- {card.get('name', 'Unknown')} ({card.get('elixir', '?')} elixir)")

    tower_name = (
        snapshot.tower_card.get("name", "Unknown")
        if snapshot.tower_card
        else "Unknown"
    )
    print(f"Tower troop: {tower_name}")
    print(f"Average elixir: {snapshot.avg_elixir:.2f}")


def _run_chat_loop(*, agent, max_iterations: int, verbose: bool) -> int:
    session = ChatSession(session_id="cli-chat")

    print("Semantic Dot Clash chat mode")
    print("Commands: restart, show deck, exit")

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nInterrupted by user", file=sys.stderr)
            return 130

        if not user_input:
            continue

        command = user_input.lower()
        if command in {"exit", "quit"}:
            return 0
        if command == "restart":
            session = ChatSession(session_id="cli-chat")
            print("[CLI] Chat session restarted.")
            continue
        if command == "show deck":
            _print_snapshot(session)
            continue

        if verbose:
            print(f"[CLI] Sending turn {session.turn_count + 1}")
            print("-" * 60)

        result = agent.chat(
            session=session,
            user_message=user_input,
            max_iterations=max_iterations,
        )

        if verbose:
            print("-" * 60)
            print(f"[CLI] Completed in {result.iterations} iterations")
            if result.used_summary:
                print("[CLI] Included summarized session context")
            print()

        print(f"Agent> {result}")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="semantic_dot_clash",
        description="Build Clash Royale decks using AI-powered semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m semantic_dot_clash "Make me a fast Hog Rider cycle deck"
  python -m semantic_dot_clash --chat
  python -m semantic_dot_clash --verbose "Build a defensive deck under 3.5 elixir"
  python -m semantic_dot_clash --model gpt-4o-mini "Quick aggressive deck"
        """,
    )
    
    parser.add_argument(
        "request",
        type=str,
        nargs="?",
        help="Natural language deck request (e.g., 'Make me a fast cycle deck')",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive ephemeral chat session",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-4o)",
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of agent loop iterations (default: 10)",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output showing agent reasoning",
    )
    
    args = parser.parse_args()
    if not args.chat and not args.request:
        parser.error("request is required unless --chat is used")
    
    # Import here to avoid slow startup for --help
    from semantic_dot_clash.agent import DeckAgent
    
    try:
        # Initialize the agent
        if args.verbose:
            print(f"[CLI] Initializing DeckAgent with model={args.model}")
        
        agent = DeckAgent(
            model=args.model,
            verbose=args.verbose,
        )

        if args.chat:
            return _run_chat_loop(
                agent=agent,
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        
        if args.verbose:
            print(f"[CLI] Building deck for: {args.request}")
            print("-" * 60)
        
        # Build the deck
        result = agent.build(
            user_request=args.request,
            max_iterations=args.max_iterations,
        )
        
        if args.verbose:
            print("-" * 60)
            print(f"[CLI] Completed in {result.iterations} iterations")
            print()
        
        # Print the result
        print(result)
        
        return 0 if result.success else 1
        
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print("Make sure LANCE_URI, LANCE_KEY, and OPENAI_API_KEY are set.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
