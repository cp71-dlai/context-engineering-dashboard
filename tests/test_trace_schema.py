"""Tests for validating trace data against JSON schema."""

import json
from pathlib import Path

import pytest

from context_engineering_dashboard.core.trace import (
    ChromaQuery,
    ChromaRetrievalResult,
    ComponentType,
    ContextComponent,
    ContextTrace,
    LLMTrace,
    ToolCall,
)

SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "trace-schema.json"


@pytest.fixture
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture
def validator(schema):
    import jsonschema

    return jsonschema.Draft7Validator(schema)


def _make_full_trace():
    """Build a full trace with all fields populated."""
    components = [
        ContextComponent(
            id="sys_001",
            type=ComponentType.SYSTEM_PROMPT,
            content="You are a helpful assistant.",
            token_count=2000,
            metadata={},
        ),
        ContextComponent(
            id="rag_001",
            type=ComponentType.RAG_DOCUMENT,
            content="Document content here.",
            token_count=8000,
            metadata={"source": "test.md", "chroma_score": 0.92},
        ),
        ContextComponent(
            id="user_001",
            type=ComponentType.USER_MESSAGE,
            content="How do I use Chroma?",
            token_count=15,
            metadata={},
        ),
    ]
    queries = [
        ChromaQuery(
            collection="documentation",
            query_text="How do I use Chroma?",
            n_results=10,
            results=[
                ChromaRetrievalResult(
                    id="doc_14",
                    content="Getting started content...",
                    token_count=8000,
                    score=0.92,
                    selected=True,
                    metadata={"source": "docs/getting-started.md"},
                    collection="documentation",
                    embedding_model="text-embedding-3-small",
                ),
                ChromaRetrievalResult(
                    id="doc_7",
                    content="Advanced content...",
                    token_count=5200,
                    score=0.87,
                    selected=False,
                    metadata={"source": "docs/advanced.md"},
                    collection="documentation",
                    embedding_model="text-embedding-3-small",
                ),
            ],
        ),
    ]
    llm = LLMTrace(
        provider="openai",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I use Chroma?"},
        ],
        response="To use Chroma, start by...",
        tool_calls=[ToolCall("search", {"query": "test"}, "result")],
        usage={"prompt_tokens": 10015, "completion_tokens": 256, "total_tokens": 10271},
        latency_ms=1234.5,
    )
    return ContextTrace(
        context_limit=128000,
        components=components,
        total_tokens=10015,
        chroma_queries=queries,
        llm_trace=llm,
        embedding_traces=[],
        timestamp="2025-01-26T10:30:00Z",
        session_id="sess_abc123",
        tags=["test", "schema-validation"],
    )


def test_full_trace_validates(validator):
    trace = _make_full_trace()
    d = trace.to_dict()
    errors = list(validator.iter_errors(d))
    assert errors == [], f"Validation errors: {[e.message for e in errors]}"


def test_minimal_trace_validates(validator):
    trace = ContextTrace(
        context_limit=4096,
        components=[
            ContextComponent("s1", ComponentType.SYSTEM_PROMPT, "Hello", 5),
        ],
        total_tokens=5,
    )
    d = trace.to_dict()
    errors = list(validator.iter_errors(d))
    assert errors == [], f"Validation errors: {[e.message for e in errors]}"


def test_invalid_missing_required_fails(validator):
    # Missing context_limit
    d = {"total_tokens": 100, "components": []}
    errors = list(validator.iter_errors(d))
    assert len(errors) > 0


def test_invalid_negative_tokens_fails(validator):
    d = {
        "context_limit": 1000,
        "total_tokens": -1,
        "components": [],
    }
    errors = list(validator.iter_errors(d))
    assert len(errors) > 0


def test_invalid_component_type_fails(validator):
    d = {
        "context_limit": 1000,
        "total_tokens": 100,
        "components": [
            {
                "id": "x",
                "type": "invalid_type",
                "content": "hi",
                "token_count": 10,
            }
        ],
    }
    errors = list(validator.iter_errors(d))
    assert len(errors) > 0


def test_all_component_types_valid(validator):
    """Every ComponentType enum value should be valid in the schema."""
    for ct in ComponentType:
        d = {
            "context_limit": 1000,
            "total_tokens": 10,
            "components": [
                {
                    "id": f"test_{ct.value}",
                    "type": ct.value,
                    "content": "test",
                    "token_count": 10,
                }
            ],
        }
        errors = list(validator.iter_errors(d))
        assert errors == [], f"ComponentType {ct.value} failed validation: {errors}"
