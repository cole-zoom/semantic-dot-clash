"""
CLI entry point for the Semantic Dot Clash deck builder.

Usage:
    python -m semantic_dot_clash "Make me a fast cycle deck"
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

load_dotenv()


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="semantic_dot_clash",
        description="Build Clash Royale decks using AI-powered semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m semantic_dot_clash "Make me a fast Hog Rider cycle deck"
  python -m semantic_dot_clash --verbose "Build a defensive deck under 3.5 elixir"
  python -m semantic_dot_clash --model gpt-4o-mini "Quick aggressive deck"
        """,
    )
    
    parser.add_argument(
        "request",
        type=str,
        help="Natural language deck request (e.g., 'Make me a fast cycle deck')",
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
