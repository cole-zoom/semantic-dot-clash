import json

from semantic_dot_clash.agent import DeckAgent, TOOL_SCHEMAS


class StubTools:
    def __init__(self) -> None:
        self.calls = []

    def search_archetypes(self, query: str, limit: int = 5):
        self.calls.append(("search_archetypes", query, limit))
        return [{"id": "cycle", "name": "Cycle"}]

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
    assert "search_archetypes" in tool_names
    assert "get_archetype" in tool_names


def test_execute_tool_routes_search_archetypes():
    agent, tools = make_agent_with_stubbed_tools()

    result = agent._execute_tool(
        "search_archetypes",
        json.dumps({"query": "fast annoying control", "limit": 3}),
    )

    assert result == [{"id": "cycle", "name": "Cycle"}]
    assert tools.calls == [("search_archetypes", "fast annoying control", 3)]


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
