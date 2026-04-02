"""FastAPI server exposing the deck builder for the web app."""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from semantic_dot_clash.agent import DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL, DeckAgent
from semantic_dot_clash.chat_session import InMemoryChatSessionStore


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

SESSION_STORE = InMemoryChatSessionStore()


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
    has_evolution: bool | None = None
    max_evolution_level: int | None = None
    evolution_image_url: str | None = None
    has_hero: bool | None = None
    hero_image_url: str | None = None
    image_data_url: str | None = None


class DeckResponse(BaseModel):
    prompt: str
    response_markdown: str
    avg_elixir: float
    cards: list[CardOut]
    tower_card: CardOut | None = None


class ChatMessageRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Ephemeral session ID")
    message: str = Field(..., min_length=1, description="Next user chat message")
    model: str | None = Field(default=None, description="OpenAI model override")
    max_iterations: int | None = Field(
        default=None, description="Max agent iterations before abort"
    )


class ChatMessageResponse(BaseModel):
    session_id: str
    response_markdown: str
    avg_elixir: float
    cards: list[CardOut]
    tower_card: CardOut | None = None
    turn_index: int
    used_summary: bool = False


class ChatResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Ephemeral session ID")


class ChatSessionStateResponse(BaseModel):
    session_id: str
    avg_elixir: float
    cards: list[CardOut]
    tower_card: CardOut | None = None
    turn_count: int


def _hydrate_result(agent: DeckAgent, *, include_image: bool, deck_snapshot) -> tuple[list[dict], dict | None]:
    return agent.hydrate_deck_snapshot(deck_snapshot, include_image=include_image)


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

    cards, tower_card = _hydrate_result(
        agent,
        include_image=True,
        deck_snapshot=result.deck_snapshot,
    )

    return DeckResponse(
        prompt=req.prompt,
        response_markdown=result.response,
        avg_elixir=result.avg_elixir,
        cards=cards,
        tower_card=tower_card,
    )


@app.post("/api/chat/message", response_model=ChatMessageResponse)
def chat_message(req: ChatMessageRequest) -> ChatMessageResponse:
    session = SESSION_STORE.get_or_create(req.session_id)

    try:
        agent = DeckAgent(model=req.model or DEFAULT_MODEL)
        result = agent.chat(
            session=session,
            user_message=req.message,
            max_iterations=req.max_iterations or DEFAULT_MAX_ITERATIONS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfacing unexpected errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Chat turn failed")

    cards, tower_card = _hydrate_result(
        agent,
        include_image=True,
        deck_snapshot=result.deck_snapshot,
    )

    return ChatMessageResponse(
        session_id=req.session_id,
        response_markdown=result.response,
        avg_elixir=result.avg_elixir,
        cards=cards,
        tower_card=tower_card,
        turn_index=result.turn_index or session.turn_count,
        used_summary=result.used_summary,
    )


@app.get("/api/chat/session/{session_id}", response_model=ChatSessionStateResponse)
def get_chat_session(session_id: str) -> ChatSessionStateResponse:
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        agent = DeckAgent(model=DEFAULT_MODEL)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfacing unexpected errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cards, tower_card = _hydrate_result(
        agent,
        include_image=True,
        deck_snapshot=session.latest_deck_snapshot,
    )

    return ChatSessionStateResponse(
        session_id=session_id,
        avg_elixir=(
            session.latest_deck_snapshot.avg_elixir
            if session.latest_deck_snapshot
            else 0.0
        ),
        cards=cards,
        tower_card=tower_card,
        turn_count=session.turn_count,
    )


@app.post("/api/chat/reset")
def reset_chat(req: ChatResetRequest) -> dict:
    SESSION_STORE.reset(req.session_id)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "semantic_dot_clash.web_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
