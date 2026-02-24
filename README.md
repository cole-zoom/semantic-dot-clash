# Semantic Dot Clash

**An AI deck builder for Clash Royale that actually understands what you mean.**

Ask for "a fast cycle deck that counters Golem beatdown" and it'll figure out you need cheap cycle cards, air defense, and something to punish slow pushes. No more scrolling through tier lists or memorizing the meta.

---

## What the hell is this?

An LLM agent with tools. You give it a natural language request, it uses semantic search to find cards that match your vibe, validates the deck for balance/legality, and spits out an 8-card deck with strategy tips.

The secret sauce: every card has been embedded with both text descriptions AND image embeddings (via CLIP), so the search actually understands card identity beyond just keywords. Ask for "swarm control" and it knows you probably want Fireball or Arrows, not the Giant.

### The Architecture (for nerds)

```
┌─────────────────────────────────────────────────────────────┐
│                        DeckAgent                            │
│  (OpenAI function calling loop, gpt-5.2 by default)         │
└─────────────────────────────────────────────────────────────┘
                              │
                    tool calls│
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        CardTools                            │
│  search_cards()   similar_cards()   get_card()   score_deck()│
└─────────────────────────────────────────────────────────────┘
                              │
                   vector search│
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     LanceDB Cloud                           │
│  cards table with combined_embedding (text 1536 + img 512)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Clone it
git clone https://github.com/coledumanski/semantic-dot-clash.git
cd semantic-dot-clash

# Install with uv (recommended) or pip
uv sync
# or
pip install -e .
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
LANCE_URI=db://your-lancedb-cloud-uri
LANCE_KEY=your-lancedb-api-key
OPENAI_API_KEY=sk-your-openai-key
```

You need:
- **LanceDB Cloud account** - the cards are stored there with their embeddings
- **OpenAI API key** - for query embedding and the agent's brain

---

## Usage

### CLI (the easy way)

```bash
# Basic usage
python -m semantic_dot_clash "Make me a Hog Rider cycle deck"

# With options
python -m semantic_dot_clash --verbose "Build a defensive deck under 3.5 elixir"
python -m semantic_dot_clash --model gpt-4o-mini "Quick aggressive deck"
```

### Web App (chat UI)

This repo includes a Next.js chat UI and a FastAPI backend that wraps the CLI agent.

**Start the API server (Python):**

```bash
python -m semantic_dot_clash.web_api
```

**Start the frontend (Next.js):**

```bash
cd web
npm install
npm run dev
```

The UI expects the API at `http://localhost:8000` by default. To change it:

```bash
export NEXT_PUBLIC_API_URL="http://localhost:8000"
```

For CORS, you can customize:

```bash
export SDC_CORS_ORIGINS="http://localhost:3000"
```

### Python API (the flexible way)

```python
from semantic_dot_clash import DeckAgent

agent = DeckAgent()
result = agent.build("Fast bridge spam deck with Battle Ram")

if result.success:
    print(result)  # Formatted deck + strategy
    print(f"Average elixir: {result.avg_elixir}")
else:
    print(f"Failed: {result.error}")
```

### Just the tools (no agent)

```python
from semantic_dot_clash import CardTools

tools = CardTools()

# Semantic search
cards = tools.search_cards("cheap cycle cards", elixir_max=3)
for card in cards:
    print(f"{card['name']}: {card['elixir']} elixir")

# Find alternatives
similar = tools.similar_cards(card_id=26000000)  # Cards like Hog Rider

# Get card details
hog = tools.get_card(card_id=26000000)

# Score a deck
score = tools.score_deck(card_ids=[...])  # 8 card IDs
print(f"Meta strength: {score.meta_strength}/100")
print(f"Missing roles: {score.missing_roles}")
```

---

## How the Agent Works

1. **You ask** for a deck in plain English
2. **Agent plans** which roles to fill (win con, support, spells, defense, cycle)
3. **Searches semantically** for cards matching each role
4. **Assembles draft deck** from search results
5. **Validates** with `score_deck()` - checks balance, legality, role coverage
6. **Iterates** if validation fails (up to 3 attempts)
7. **Presents** final deck with strategy, synergies, and gameplan

The agent is NOT allowed to make up cards. Every card must come from a tool call. This prevents hallucinated "Super Mega Dragon" type nonsense.

---

## Deck Scoring

The `score_deck()` function evaluates your deck across multiple dimensions:

| Metric | What it checks |
|--------|----------------|
| `avg_elixir` | Average elixir cost |
| `role_coverage` | Which roles are filled (win con, anti-air, splash, tank, cycle) |
| `missing_roles` | Important roles you're missing |
| `synergy_score` | How well cards work together (embedding similarity) |
| `meta_strength` | Overall score 0-100 based on balance |
| `balance_warnings` | Specific issues (no spells, too expensive, duplicates, etc.) |

---

## Lance Schema

Three tables, because why stop at one!

### Cards Table

The main event. Every card in the game with multimodal embeddings and vibe annotations.

| Field | Type | Description |
|-------|------|-------------|
| `id` | int64 | Unique card ID from Clash Royale API |
| `name` | string | Official card name |
| `rarity` | string | Common, Rare, Epic, Legendary, Champion |
| `type` | string | Troop, Spell, Building |
| `elixir` | int32 | Elixir cost (1-10) |
| `description` | string | Official card description |
| `image` | binary | Raw image bytes for card artwork |
| `image_embedding` | float[512] | Visual embedding from CLIP |
| `text_embedding` | float[1536] | Semantic embedding from text-embedding-3-small |
| `combined_embedding` | float[2048] | Concatenation of text + image embeddings (this is what we search) |
| `role_tags` | list\<string\> | Tactical roles: "win condition", "cycle", "splash", "tank", "anti-air" |
| `vibe_tags` | list\<string\> | Player perception: "toxic", "annoying", "spammy", "wholesome", "tryhard" |
| `crowd_ratings` | map\<string, float\> | Aggregated vibe ratings (e.g., `{"toxic": 0.73}`) |
| `llm_vibe_summary` | string | GPT-generated summary of how players perceive the card |

### Archetypes Table

High-level deck styles and meta archetypes (Hog Cycle, Lava Loon, Log Bait, etc.).

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique archetype ID (e.g., "hog_cycle") |
| `name` | string | Human-readable name |
| `description` | string | Playstyle and strategy description |
| `example_decks` | list\<list\<int64\>\> | Example decks as arrays of card IDs |
| `embedding` | float[768] | Semantic embedding of the description |
| `tags` | list\<string\> | Tactical tags: "cycle", "control", "air", "aggro", "beatdown" |
| `meta_strength` | float | Performance score / win-rate indicator |
| `playstyle_vibes` | list\<string\> | Vibe tags: "annoying", "spammy", "toxic", "off-meta" |
| `vibe_embedding` | float[768] | Embedding for semantic vibe filtering |
| `llm_vibe_summary` | string | Natural language summary of the archetype's "feel" |

### Decks Table

Specific 8-card decks with synergy analysis and vibe ratings.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique deck ID (UUID or hash) |
| `card_ids` | int64[8] | The 8 card IDs that compose the deck |
| `archetype_id` | string | Foreign key to archetype |
| `average_elixir` | float | Auto-computed average elixir |
| `roles` | list\<string\> | Summary of tactical roles in the deck |
| `deck_embedding` | float[1024] | Combined embedding of all 8 cards |
| `synergy_embedding` | float[768] | Embedding capturing how cards work together |
| `meta_score` | float | Win-rate / usage rate / meta strength |
| `user_labels` | list\<string\> | Player labels: "toxic", "tryhard", "meme", "casual" |
| `crowd_ratings` | map\<string, float\> | Aggregated vibe ratings |
| `llm_vibe_summary` | string | GPT classification of the deck's vibe |
| `conflict_tags` | list\<string\> | Counter strategies: "hard counter air", "anti-swarm" |
| `combo_notes` | string | GPT-generated description of key synergies |

### Why all the vibe stuff?

Because Clash Royale players have *feelings* about cards. Asking for "a deck that isn't toxic" is a legitimate request, and the vibe embeddings let you search by that. The `crowd_ratings` and `llm_vibe_summary` fields capture community perception so you can build decks that match your desired energy level.

---

## Scripts

The `scripts/` folder has utilities for building the database:

| Script | What it does |
|--------|--------------|
| `fetch_cards.py` | Pull card data from the official Clash Royale API |
| `scrape_card_descriptions.py` | Scrape descriptions from the wiki |
| `generate_card_vibes.py` | Use GPT to generate "vibe" descriptions |
| `generate_card_embeddings.py` | Create text + image embeddings |
| `load_cards_to_lance.py` | Upload everything to LanceDB Cloud |
| `create_tables.py` | Initialize the LanceDB schema |
| `test_tools.py` | Sanity check the tools work |

---

## Project Structure

```
semantic_dot_clash/
├── src/semantic_dot_clash/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # CLI entry point
│   ├── agent.py             # DeckAgent (agentic loop)
│   ├── tools.py             # CardTools (search, score, etc.)
│   └── tables/              # LanceDB schema stuff
├── scripts/                 # Data pipeline scripts
├── data/                    # Card JSON, embeddings, etc.
├── system_prompt.txt        # The agent's instructions
└── pyproject.toml           # Package config
```

## Limitations

- **Requires cloud services** - LanceDB Cloud + OpenAI API (not free)
- **Card data might lag behind updates** - need to re-run pipeline when new cards drop
- **Meta knowledge is static** - the "meta strength" score is heuristic, not based on live ladder data
- **Tower troops** - the agent knows about them but scoring treats them separately from the main 8

---

## License

MIT. Do whatever you want.

---


*Built because I got tired of losing to meta decks and wanted a robot to think for me.*
