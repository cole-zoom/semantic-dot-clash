import json

from semantic_dot_clash.agent import DeckAgent, DeckResult
from semantic_dot_clash.chat_session import ChatSession, DeckSnapshot, InMemoryChatSessionStore


class StubFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class StubToolCall:
    def __init__(self, tool_id: str, name: str, arguments: str) -> None:
        self.id = tool_id
        self.function = StubFunction(name=name, arguments=arguments)


class StubMessage:
    def __init__(self, content: str | None, tool_calls=None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class StubChoice:
    def __init__(self, message: StubMessage) -> None:
        self.message = message


class StubResponse:
    def __init__(self, message: StubMessage) -> None:
        self.choices = [StubChoice(message)]


class StubCompletions:
    def __init__(self, responses: list[StubResponse]) -> None:
        self._responses = list(responses)

    def create(self, **_kwargs):
        return self._responses.pop(0)


class StubChatClient:
    def __init__(self, responses: list[StubResponse]) -> None:
        self.completions = StubCompletions(responses)


class StubOpenAIClient:
    def __init__(self, responses: list[StubResponse]) -> None:
        self.chat = StubChatClient(responses)


def make_lightweight_agent() -> DeckAgent:
    agent = DeckAgent.__new__(DeckAgent)
    agent.model = "test-model"
    agent.verbose = False
    agent._system_prompt = "system prompt"
    return agent


def test_run_agent_loop_tracks_latest_score_deck_snapshot():
    score_payload = {
        "avg_elixir": 3.1,
        "battle_cards": [
            {"id": 1, "name": "Knight", "elixir": 3},
            {"id": 2, "name": "Archers", "elixir": 3},
        ],
        "tower_card": {"id": 99, "name": "Princess Tower", "elixir": 0},
    }
    agent = make_lightweight_agent()
    agent._openai = StubOpenAIClient(
        [
            StubResponse(
                StubMessage(
                    content=None,
                    tool_calls=[
                        StubToolCall(
                            "tool-1",
                            "score_deck",
                            json.dumps(
                                {
                                    "battle_card_ids": [1, 2],
                                    "tower_card_id": 99,
                                }
                            ),
                        )
                    ],
                )
            ),
            StubResponse(StubMessage(content="Final deck ready.")),
        ]
    )
    agent._execute_tool = lambda _name, _arguments: score_payload

    result = agent.build("Build me a deck", max_iterations=2)

    assert result.success is True
    assert result.response == "Final deck ready."
    assert result.avg_elixir == 3.1
    assert result.deck_snapshot is not None
    assert result.deck_snapshot.battle_card_ids == [1, 2]
    assert result.tower_card == {"id": 99, "name": "Princess Tower", "elixir": 0}


def test_chat_updates_session_and_reuses_summary():
    calls: list[dict] = []

    def fake_run_agent_loop(*, messages, max_iterations: int, used_summary: bool = False):
        calls.append(
            {
                "messages": messages,
                "max_iterations": max_iterations,
                "used_summary": used_summary,
            }
        )
        return DeckResult(
            success=True,
            response="Updated deck.",
            iterations=1,
            deck_snapshot=DeckSnapshot(
                battle_cards=[{"id": 10, "name": "Miner", "elixir": 3}],
                tower_card={"id": 99, "name": "Princess Tower", "elixir": 0},
                avg_elixir=3.0,
            ),
            used_summary=used_summary,
        )

    agent = make_lightweight_agent()
    agent._run_agent_loop = fake_run_agent_loop

    session = ChatSession(session_id="abc")

    first = agent.chat(session=session, user_message="Build me a miner deck")
    second = agent.chat(session=session, user_message="Make it cheaper")

    assert first.turn_index == 1
    assert second.turn_index == 2
    assert session.turn_count == 2
    assert session.latest_deck_snapshot is not None
    assert "Make it cheaper" in session.rolling_summary
    assert calls[0]["used_summary"] is False
    assert calls[1]["used_summary"] is True
    assert any(
        message["role"] == "assistant" and message["content"] == "Updated deck."
        for message in calls[1]["messages"]
    )


def test_in_memory_session_store_can_reset_sessions():
    store = InMemoryChatSessionStore()

    session = store.get_or_create("session-1")
    session.preference_notes.append("fast cycle")
    assert store.get("session-1") is session

    store.reset("session-1")

    assert store.get("session-1") is None
