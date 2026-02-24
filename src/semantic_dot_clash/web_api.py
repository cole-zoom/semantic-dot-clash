"""FastAPI server exposing the deck builder for the web app."""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from semantic_dot_clash.agent import DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL, DeckAgent


def _split_origins(raw: str | None) -> List[str]:
    if not raw:
        return ["http://localhost:3000"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Semantic Dot Clash API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_split_origins(os.environ.get("SDC_CORS_ORIGINS")),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DeckRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User deck request")
    model: str | None = Field(default=None, description="OpenAI model override")
    max_iterations: int | None = Field(
        default=None, description="Max agent iterations before abort"
    )


class CardOut(BaseModel):
    id: int
    name: str
    rarity: str | None = None
    type: str | None = None
    elixir: int | None = None
    description: str | None = None
    role_tags: list[str] | None = None
    vibe_tags: list[str] | None = None
    image_data_url: str | None = None


class DeckResponse(BaseModel):
    prompt: str
    response_markdown: str
    avg_elixir: float
    cards: list[CardOut]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/deck", response_model=DeckResponse)
def build_deck(req: DeckRequest) -> DeckResponse:
    try:
        agent = DeckAgent(model=req.model or DEFAULT_MODEL)
        result = agent.build(
            user_request=req.prompt,
            max_iterations=req.max_iterations or DEFAULT_MAX_ITERATIONS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfacing unexpected errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Deck build failed")

    tools = agent._tools
    card_ids = [card.get("id") for card in result.deck if card.get("id") is not None]
    cards: list[dict] = []
    for card_id in card_ids:
        card = tools.get_card(card_id=card_id, include_image=True)
        if card:
            cards.append(card)

    return DeckResponse(
        prompt=req.prompt,
        response_markdown=result.response,
        avg_elixir=result.avg_elixir,
        cards=cards,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "semantic_dot_clash.web_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
