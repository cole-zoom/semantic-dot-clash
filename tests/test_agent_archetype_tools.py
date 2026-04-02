import json

from semantic_dot_clash.agent import DeckAgent, TOOL_SCHEMAS


class StubTools:
    def __init__(self) -> None:
        self.calls = []

    def search_cards(
        self,
        query: str,
        elixir_min=None,
        elixir_max=None,
        type=None,
        rarity=None,
        exclude_card_ids=None,
        limit: int = 10,
    ):
        self.calls.append(
            (
                "search_cards",
                query,
                elixir_min,
                elixir_max,
                type,
                rarity,
                exclude_card_ids,
                limit,
            )
        )
        return [{"id": 26000000, "name": "Knight"}]

    def search_archetypes(self, query: str, limit: int = 5):
        self.calls.append(("search_archetypes", query, limit))
        return [{"id": "cycle", "name": "Cycle"}]

    def select_archetype_for_core(
        self,
        user_request: str,
        core_card_ids: list[int],
        limit: int = 5,
    ):
        self.calls.append(
            ("select_archetype_for_core", user_request, core_card_ids, limit)
        )
        return [{"id": "log_bait", "name": "Log Bait", "fit_score": 0.93}]

    def search_complementary_cards(
        self,
        user_request: str,
        archetype_id: str,
        core_card_ids: list[int],
        role_hint=None,
        elixir_min=None,
        elixir_max=None,
        type=None,
        rarity=None,
        limit: int = 10,
    ):
        self.calls.append(
            (
                "search_complementary_cards",
                user_request,
                archetype_id,
                core_card_ids,
                role_hint,
                elixir_min,
                elixir_max,
                type,
                rarity,
                limit,
            )
        )
        return [{"id": 26000032, "name": "Goblin Barrel", "fit_score": 0.91}]

    def get_archetype(self, archetype_id: str):
        self.calls.append(("get_archetype", archetype_id))
        if archetype_id == "missing":
            return None
        return {"id": archetype_id, "name": "Cycle"}


def make_agent_with_stubbed_tools() -> tuple[DeckAgent, StubTools]:
    agent = DeckAgent.__new__(DeckAgent)
    tools = StubTools()
    agent._tools = tools
    return agent, tools


def test_tool_schemas_include_archetype_tools():
    tool_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert "search_cards" in tool_names
    assert "search_archetypes" in tool_names
    assert "get_archetype" in tool_names
    assert "select_archetype_for_core" in tool_names
    assert "search_complementary_cards" in tool_names


def test_execute_tool_routes_search_cards_with_excludes():
    agent, tools = make_agent_with_stubbed_tools()

    result = agent._execute_tool(
        "search_cards",
        json.dumps(
            {
                "query": "annoying pressure cards",
                "exclude_card_ids": [26000032],
                "limit": 4,
            }
        ),
    )

    assert result == [{"id": 26000000, "name": "Knight"}]
    assert tools.calls == [
        (
            "search_cards",
            "annoying pressure cards",
            None,
            None,
            None,
            None,
            [26000032],
            4,
        )
    ]


def test_execute_tool_routes_search_archetypes():
    agent, tools = make_agent_with_stubbed_tools()

    result = agent._execute_tool(
        "search_archetypes",
        json.dumps({"query": "fast annoying control", "limit": 3}),
    )

    assert result == [{"id": "cycle", "name": "Cycle"}]
    assert tools.calls == [("search_archetypes", "fast annoying control", 3)]


def test_execute_tool_routes_select_archetype_for_core():
    agent, tools = make_agent_with_stubbed_tools()

    result = agent._execute_tool(
        "select_archetype_for_core",
        json.dumps(
            {
                "user_request": "annoying deck",
                "core_card_ids": [26000032, 26000041],
                "limit": 2,
            }
        ),
    )

    assert result == [{"id": "log_bait", "name": "Log Bait", "fit_score": 0.93}]
    assert tools.calls == [
        (
            "select_archetype_for_core",
            "annoying deck",
            [26000032, 26000041],
            2,
        )
    ]


def test_execute_tool_routes_search_complementary_cards():
    agent, tools = make_agent_with_stubbed_tools()

    result = agent._execute_tool(
        "search_complementary_cards",
        json.dumps(
            {
                "user_request": "annoying deck",
                "archetype_id": "log_bait",
                "core_card_ids": [26000032],
                "role_hint": "cheap anti-air support",
                "limit": 3,
            }
        ),
    )

    assert result == [{"id": 26000032, "name": "Goblin Barrel", "fit_score": 0.91}]
    assert tools.calls == [
        (
            "search_complementary_cards",
            "annoying deck",
            "log_bait",
            [26000032],
            "cheap anti-air support",
            None,
            None,
            None,
            None,
            3,
        )
    ]


def test_execute_tool_routes_get_archetype():
    agent, tools = make_agent_with_stubbed_tools()

    result = agent._execute_tool(
        "get_archetype",
        json.dumps({"archetype_id": "cycle"}),
    )

    assert result == {"id": "cycle", "name": "Cycle"}
    assert tools.calls == [("get_archetype", "cycle")]


def test_execute_tool_returns_error_for_missing_archetype():
    agent, _tools = make_agent_with_stubbed_tools()

    result = agent._execute_tool(
        "get_archetype",
        json.dumps({"archetype_id": "missing"}),
    )

    assert result == {"error": "Archetype with ID missing not found"}
