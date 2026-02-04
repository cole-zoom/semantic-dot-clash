#!/usr/bin/env python3
"""
Script to create all Lance database tables for Semantic Dot Clash.

This script initializes the Cards, Archetypes, and Decks tables with their
defined schemas. If the tables already exist, it will open them without
overwriting existing data.

Environment Variables:
    LANCE_URI: LanceDB Cloud URI (e.g., db://semantic-clash-royale-deck-90twk0)
    LANCE_KEY: LanceDB Cloud API key

Usage:
    python scripts/create_tables.py
    python scripts/create_tables.py --force
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

import lancedb

load_dotenv()

# Add src to path for local development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from semantic_dot_clash.tables import (
    get_cards_schema,
    get_archetypes_schema,
    get_decks_schema,
)


def get_db_connection() -> lancedb.DBConnection:
    """Get LanceDB connection using environment variables."""
    uri = os.environ.get("LANCE_URI")
    api_key = os.environ.get("LANCE_KEY")
    
    if not uri:
        raise ValueError("LANCE_URI environment variable is required")
    if not api_key:
        raise ValueError("LANCE_KEY environment variable is required")
    
    return lancedb.connect(
        uri=uri,
        api_key=api_key,
        region="us-east-1"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create all Lance database tables for Semantic Dot Clash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create tables (requires LANCE_URI and LANCE_KEY env vars):
    python scripts/create_tables.py

  Force recreate tables (WARNING: deletes existing data):
    python scripts/create_tables.py --force
        """
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate tables, deleting existing data (use with caution)"
    )
    
    args = parser.parse_args()
    
    uri = os.environ.get("LANCE_URI", "")
    
    print(f"🎯 Semantic Dot Clash - Table Initialization")
    print(f"{'='*60}")
    print(f"Database URI: {uri}")
    print()
    
    try:
        db = get_db_connection()
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Warning for force mode
    if args.force:
        print("⚠️  WARNING: --force flag detected!")
        print("   This will DELETE all existing data in the tables.")
        response = input("   Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Aborted.")
            sys.exit(0)
        print()
        
        # Delete existing tables
        for table_name in ["cards", "archetypes", "decks"]:
            try:
                if table_name in db.table_names():
                    db.drop_table(table_name)
                    print(f"🗑️  Dropped existing table: {table_name}")
            except Exception as e:
                print(f"⚠️  Could not drop table {table_name}: {e}")
        print()
    
    # Create tables
    print("Creating tables...")
    print()
    
    try:
        schemas = {
            "cards": get_cards_schema(),
            "archetypes": get_archetypes_schema(),
            "decks": get_decks_schema(),
        }
        
        tables = {}
        for table_name, schema in schemas.items():
            if table_name not in db.table_names():
                tables[table_name] = db.create_table(table_name, schema=schema)
                print(f"   ✅ Created table: {table_name}")
            else:
                tables[table_name] = db.open_table(table_name)
                print(f"   📂 Opened existing table: {table_name}")
        
        print()
        print("✅ Tables ready!")
        print()
        
        print("📈 Table Statistics:")
        for table_name in ["cards", "archetypes", "decks"]:
            table = db.open_table(table_name)
            count = table.count_rows()
            print(f"   - {table_name}: {count} records")
        
        print()
        print("🎉 All done! Your Lance database is ready.")
        print()
        print("Next steps:")
        print("  1. Use the StagingPipeline to load data:")
        print("     from semantic_dot_clash.tables import StagingPipeline")
        print("     pipeline = StagingPipeline()")
        print("     pipeline.stage_cards(your_cards_data)")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
