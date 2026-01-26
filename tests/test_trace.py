"""Tests for core trace data structures."""

import json
import tempfile
from pathlib import Path

from context_engineering_dashboard.core.trace import (
    ChromaQuery,
    ChromaRetrievalResult,
    ComponentType,
    ContextComponent,
    ContextTrace,
    EmbeddingTrace,
    LLMTrace,
    ToolCall,
)

# --- Roundtrip serialization tests ---


def test_component_type_values():
    assert ComponentType.SYSTEM_PROMPT.value == "system_prompt"
    assert ComponentType.USER_MESSAGE.value == "user_message"
    assert ComponentType.CHAT_HISTORY.value == "chat_history"
    assert ComponentType.RAG_DOCUMENT.value == "rag_document"
    assert ComponentType.TOOL.value == "tool"
    assert ComponentType.FEW_SHOT.value == "few_shot"
    assert ComponentType.MEMORY.value == "memory"


def test_context_component_roundtrip():
    comp = ContextComponent(
        id="sys_1",
        type=ComponentType.SYSTEM_PROMPT,
        content="You are helpful.",
        token_count=5,
        metadata={"source": "manual"},
    )
    d = comp.to_dict()
    restored = ContextComponent.from_dict(d)
    assert restored.id == comp.id
    assert restored.type == comp.type
    assert restored.content == comp.content
    assert restored.token_count == comp.token_count
    assert restored.metadata == comp.metadata


def test_context_component_default_metadata():
    d = {
        "id": "x",
        "type": "user_message",
        "content": "hi",
        "token_count": 1,
    }
    comp = ContextComponent.from_dict(d)
    assert comp.metadata == {}


def test_chroma_retrieval_result_roundtrip():
    result = ChromaRetrievalResult(
        id="doc_1",
        content="Some document content",
        token_count=10,
        score=0.92,
        selected=True,
        metadata={"source": "test.md"},
        collection="docs",
        embedding_model="text-embedding-3-small",
    )
    d = result.to_dict()
    restored = ChromaRetrievalResult.from_dict(d)
    assert restored.id == result.id
    assert restored.score == result.score
    assert restored.selected == result.selected
    assert restored.collection == result.collection
    assert restored.embedding_model == result.embedding_model


def test_chroma_retrieval_result_defaults():
    d = {
        "id": "d1",
        "content": "text",
        "token_count": 5,
        "score": 0.5,
        "selected": False,
    }
    r = ChromaRetrievalResult.from_dict(d)
    assert r.metadata == {}
    assert r.collection == ""
    assert r.embedding_model == ""


def test_chroma_query_roundtrip():
    results = [
        ChromaRetrievalResult("d1", "text1", 5, 0.9, True),
        ChromaRetrievalResult("d2", "text2", 10, 0.7, False),
    ]
    query = ChromaQuery(
        collection="test_collection",
        query_text="what is chroma?",
        query_embedding=[0.1, 0.2, 0.3],
        n_results=10,
        where_filter={"source": "docs"},
        results=results,
    )
    d = query.to_dict()
    restored = ChromaQuery.from_dict(d)
    assert restored.collection == query.collection
    assert restored.query_text == query.query_text
    assert restored.query_embedding == query.query_embedding
    assert restored.n_results == query.n_results
    assert restored.where_filter == query.where_filter
    assert len(restored.results) == 2
    assert restored.results[0].id == "d1"


def test_chroma_query_selected_unselected():
    results = [
        ChromaRetrievalResult("d1", "text1", 5, 0.9, True),
        ChromaRetrievalResult("d2", "text2", 10, 0.7, False),
        ChromaRetrievalResult("d3", "text3", 8, 0.8, True),
    ]
    query = ChromaQuery(collection="c", query_text="q", results=results)
    assert len(query.selected_results) == 2
    assert len(query.unselected_results) == 1
    assert query.unselected_results[0].id == "d2"


def test_tool_call_roundtrip():
    tc = ToolCall(name="search", arguments={"query": "test"}, result="found it")
    d = tc.to_dict()
    restored = ToolCall.from_dict(d)
    assert restored.name == tc.name
    assert restored.arguments == tc.arguments
    assert restored.result == tc.result


def test_tool_call_optional_result():
    d = {"name": "fn", "arguments": {"x": 1}}
    tc = ToolCall.from_dict(d)
    assert tc.result is None


def test_llm_trace_roundtrip():
    trace = LLMTrace(
        provider="openai",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        response="Hello!",
        tool_calls=[ToolCall("fn", {"x": 1}, "result")],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        latency_ms=123.4,
    )
    d = trace.to_dict()
    restored = LLMTrace.from_dict(d)
    assert restored.provider == trace.provider
    assert restored.model == trace.model
    assert restored.messages == trace.messages
    assert restored.response == trace.response
    assert len(restored.tool_calls) == 1
    assert restored.tool_calls[0].name == "fn"
    assert restored.usage == trace.usage
    assert restored.latency_ms == trace.latency_ms


def test_llm_trace_defaults():
    d = {
        "provider": "openai",
        "model": "gpt-4o",
        "messages": [],
        "response": "ok",
    }
    t = LLMTrace.from_dict(d)
    assert t.tool_calls == []
    assert t.usage == {}
    assert t.latency_ms == 0.0


def test_embedding_trace_roundtrip():
    trace = EmbeddingTrace(
        provider="openai",
        model="text-embedding-3-small",
        input_text="hello world",
        embedding=[0.1, 0.2, 0.3],
        latency_ms=50.0,
    )
    d = trace.to_dict()
    restored = EmbeddingTrace.from_dict(d)
    assert restored.provider == trace.provider
    assert restored.model == trace.model
    assert restored.input_text == trace.input_text
    assert restored.embedding == trace.embedding
    assert restored.latency_ms == trace.latency_ms


def test_context_trace_roundtrip():
    components = [
        ContextComponent("sys_1", ComponentType.SYSTEM_PROMPT, "System prompt", 100),
        ContextComponent("user_1", ComponentType.USER_MESSAGE, "User msg", 50),
    ]
    queries = [
        ChromaQuery(
            collection="docs",
            query_text="test",
            results=[ChromaRetrievalResult("d1", "doc", 20, 0.9, True)],
        )
    ]
    llm = LLMTrace("openai", "gpt-4o", [], "response")
    embeddings = [EmbeddingTrace("openai", "text-embedding-3-small", "hi", [0.1], 10.0)]

    trace = ContextTrace(
        context_limit=128000,
        components=components,
        total_tokens=150,
        chroma_queries=queries,
        llm_trace=llm,
        embedding_traces=embeddings,
        timestamp="2025-01-01T00:00:00Z",
        session_id="sess_123",
        tags=["test"],
    )
    d = trace.to_dict()
    restored = ContextTrace.from_dict(d)

    assert restored.context_limit == trace.context_limit
    assert restored.total_tokens == trace.total_tokens
    assert len(restored.components) == 2
    assert restored.components[0].type == ComponentType.SYSTEM_PROMPT
    assert len(restored.chroma_queries) == 1
    assert restored.llm_trace is not None
    assert restored.llm_trace.provider == "openai"
    assert len(restored.embedding_traces) == 1
    assert restored.timestamp == trace.timestamp
    assert restored.session_id == trace.session_id
    assert restored.tags == ["test"]


def test_context_trace_no_optional_fields():
    d = {
        "context_limit": 4096,
        "total_tokens": 100,
        "components": [{"id": "s", "type": "system_prompt", "content": "hi", "token_count": 100}],
    }
    trace = ContextTrace.from_dict(d)
    assert trace.chroma_queries == []
    assert trace.llm_trace is None
    assert trace.embedding_traces == []
    assert trace.timestamp == ""
    assert trace.session_id == ""
    assert trace.tags == []


# --- Computed properties ---


def test_unused_tokens():
    trace = ContextTrace(context_limit=1000, components=[], total_tokens=300)
    assert trace.unused_tokens == 700


def test_utilization():
    trace = ContextTrace(context_limit=1000, components=[], total_tokens=250)
    assert trace.utilization == 25.0


def test_utilization_full():
    trace = ContextTrace(context_limit=1000, components=[], total_tokens=1000)
    assert trace.utilization == 100.0


def test_get_components_by_type():
    components = [
        ContextComponent("s1", ComponentType.SYSTEM_PROMPT, "sys", 100),
        ContextComponent("r1", ComponentType.RAG_DOCUMENT, "rag1", 200),
        ContextComponent("r2", ComponentType.RAG_DOCUMENT, "rag2", 300),
        ContextComponent("u1", ComponentType.USER_MESSAGE, "user", 50),
    ]
    trace = ContextTrace(context_limit=10000, components=components, total_tokens=650)
    rag_comps = trace.get_components_by_type(ComponentType.RAG_DOCUMENT)
    assert len(rag_comps) == 2
    assert all(c.type == ComponentType.RAG_DOCUMENT for c in rag_comps)

    sys_comps = trace.get_components_by_type(ComponentType.SYSTEM_PROMPT)
    assert len(sys_comps) == 1

    tool_comps = trace.get_components_by_type(ComponentType.TOOL)
    assert len(tool_comps) == 0


# --- File I/O ---


def test_to_json_from_json():
    components = [
        ContextComponent("s1", ComponentType.SYSTEM_PROMPT, "System prompt", 100),
    ]
    trace = ContextTrace(
        context_limit=128000,
        components=components,
        total_tokens=100,
        timestamp="2025-01-01T00:00:00Z",
        session_id="sess_abc",
        tags=["io_test"],
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        trace.to_json(path)
        restored = ContextTrace.from_json(path)
        assert restored.context_limit == 128000
        assert len(restored.components) == 1
        assert restored.components[0].id == "s1"
        assert restored.tags == ["io_test"]

        # Verify file is valid JSON
        with open(path) as fh:
            data = json.load(fh)
        assert data["context_limit"] == 128000
    finally:
        path.unlink(missing_ok=True)


def test_to_json_from_json_string_path():
    trace = ContextTrace(context_limit=1000, components=[], total_tokens=0)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        trace.to_json(path)
        restored = ContextTrace.from_json(path)
        assert restored.context_limit == 1000
    finally:
        Path(path).unlink(missing_ok=True)
