"""Tests for Chroma tracer (TracedCollection) and available pool rendering."""

from unittest.mock import MagicMock

from context_engineering_dashboard import trace_chroma
from context_engineering_dashboard.core.context_window import ContextWindow
from context_engineering_dashboard.core.trace import (
    ComponentType,
    ContextComponent,
)
from context_engineering_dashboard.tracers.chroma_tracer import (
    TracedCollection,
    count_tokens,
)


class _MockCollection:
    """Mock chromadb Collection."""

    def __init__(self, name: str = "test_docs"):
        self._name = name

    @property
    def name(self):
        return self._name

    def query(self, **kwargs):
        n = kwargs.get("n_results", 3)
        ids = [f"doc_{i}" for i in range(n)]
        documents = [f"Content of document {i}." for i in range(n)]
        distances = [0.1 * (i + 1) for i in range(n)]
        metadatas = [{"source": f"file_{i}.md"} for i in range(n)]
        return {
            "ids": [ids],
            "documents": [documents],
            "distances": [distances],
            "metadatas": [metadatas],
        }


def test_trace_chroma_convenience():
    coll = _MockCollection()
    traced = trace_chroma(coll)
    assert isinstance(traced, TracedCollection)
    assert traced.name == "test_docs"


def test_query_captures_results():
    coll = _MockCollection()
    traced = TracedCollection(coll)
    results = traced.query(query_texts=["test query"], n_results=3)

    assert results is not None
    assert len(results["ids"][0]) == 3

    trace = traced.get_trace()
    assert len(trace.chroma_queries) == 1
    query = trace.chroma_queries[0]
    assert query.query_text == "test query"
    assert len(query.results) == 3
    # Scores should be between 0 and 1
    for r in query.results:
        assert 0 <= r.score <= 1


def test_mark_selected():
    coll = _MockCollection()
    traced = TracedCollection(coll)
    traced.query(query_texts=["test"], n_results=3)

    traced.mark_selected(["doc_0", "doc_1"])

    trace = traced.get_trace()
    query = trace.chroma_queries[0]
    selected = [r for r in query.results if r.selected]
    unselected = [r for r in query.results if not r.selected]
    assert len(selected) == 2
    assert len(unselected) == 1
    assert unselected[0].id == "doc_2"


def test_get_trace_builds_context_trace():
    coll = _MockCollection()
    traced = TracedCollection(coll, context_limit=64000)
    traced.query(query_texts=["test"], n_results=3)
    traced.mark_selected(["doc_0"])
    traced.add_system_prompt("You are helpful.")
    traced.add_user_message("Question?")

    trace = traced.get_trace()

    assert trace.context_limit == 64000
    assert len(trace.components) == 3  # system + user + 1 RAG
    types = [c.type for c in trace.components]
    assert ComponentType.SYSTEM_PROMPT in types
    assert ComponentType.USER_MESSAGE in types
    assert ComponentType.RAG in types
    assert trace.total_tokens > 0
    assert trace.session_id != ""
    assert trace.timestamp != ""


def test_add_component():
    coll = _MockCollection()
    traced = TracedCollection(coll)
    comp = ContextComponent("custom_1", ComponentType.SCRATCHPAD, "Remember this.", 10)
    traced.add_component(comp)

    trace = traced.get_trace()
    assert len(trace.components) == 1
    assert trace.components[0].type == ComponentType.SCRATCHPAD


def test_count_tokens():
    text = "Hello, world!"
    tokens = count_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_reset():
    coll = _MockCollection()
    traced = TracedCollection(coll)
    traced.query(query_texts=["test"], n_results=3)
    traced.mark_selected(["doc_0"])
    traced.add_system_prompt("System prompt")
    traced.reset()

    trace = traced.get_trace()
    assert len(trace.chroma_queries) == 0
    assert len(trace.components) == 0


def test_delegation():
    coll = _MockCollection()
    coll.count = MagicMock(return_value=42)
    traced = TracedCollection(coll)
    assert traced.count() == 42
    coll.count.assert_called_once()


def test_available_pool_html():
    coll = _MockCollection()
    traced = TracedCollection(coll, context_limit=128000)
    traced.query(query_texts=["test"], n_results=3)
    traced.mark_selected(["doc_0", "doc_1"])
    traced.add_system_prompt("System prompt here.")

    trace = traced.get_trace()
    ctx = ContextWindow(trace=trace, show_available_pool=True)
    h = ctx.to_html()

    assert "ced-two-panel" in h
    assert "ced-panel" in h
    assert "ced-scissors" in h
    assert "ced-selected" in h
    assert "ced-unselected" in h
    assert "doc_0" in h
    assert "doc_1" in h
    assert "doc_2" in h


def test_available_pool_without_queries_falls_back():
    from context_engineering_dashboard.core.trace import ContextTrace

    trace = ContextTrace(
        context_limit=1000,
        components=[ContextComponent("s1", ComponentType.SYSTEM_PROMPT, "Hi", 5)],
        total_tokens=5,
    )
    ctx = ContextWindow(trace=trace, show_available_pool=True)
    h = ctx.to_html()
    # Falls back to normal context window since there are no chroma queries
    assert "ced-context-window" in h
    # No actual two-panel div rendered (only CSS rule exists)
    assert '<div class="ced-two-panel">' not in h


def test_score_conversion_l2():
    """L2 distance of 0 should give score 1.0, larger distances give smaller scores."""
    coll = MagicMock()
    coll.name = "test"
    coll.query.return_value = {
        "ids": [["d1", "d2"]],
        "documents": [["close doc", "far doc"]],
        "distances": [[0.0, 2.0]],
        "metadatas": [[{}, {}]],
    }
    traced = TracedCollection(coll)
    traced.query(query_texts=["test"], n_results=2)

    trace = traced.get_trace()
    results = trace.chroma_queries[0].results
    assert results[0].score == 1.0  # distance 0
    assert results[1].score < results[0].score  # distance 2.0 -> lower score
