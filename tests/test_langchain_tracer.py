"""Tests for LangChain tracer (mocked callbacks)."""

from unittest.mock import patch

import pytest

from context_engineering_dashboard.core.trace import ComponentType
from context_engineering_dashboard.tracers.langchain_tracer import LangChainTracer


class _MockDocument:
    """Mock LangChain Document."""

    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _MockGeneration:
    def __init__(self, text: str):
        self.text = text


class _MockLLMResult:
    def __init__(self, generations):
        self.generations = generations


class TestLangChainTracer:
    def test_retriever_docs_become_rag_components(self):
        """Retrieved documents should become RAG components."""
        tracer = LangChainTracer()
        with tracer:
            handler = tracer.handler

            # Simulate retriever callback
            docs = [
                _MockDocument("Document about Chroma", {"source": "chroma.md"}),
                _MockDocument("Document about RAG", {"source": "rag.md"}),
            ]
            handler.on_retriever_end(docs)

        trace = tracer.result
        assert trace is not None
        rag_comps = [c for c in trace.components if c.type == ComponentType.RAG]
        assert len(rag_comps) == 2
        assert rag_comps[0].content == "Document about Chroma"
        assert rag_comps[0].metadata == {"source": "chroma.md"}

    def test_llm_calls_captured(self):
        """LLM start/end events should be captured."""
        tracer = LangChainTracer()
        with tracer:
            handler = tracer.handler

            handler.on_llm_start(
                {"name": "ChatOpenAI"},
                ["You are helpful.\nHi there!"],
            )
            handler.on_llm_end(_MockLLMResult([[_MockGeneration("Hello! How can I help?")]]))

        trace = tracer.result
        assert trace is not None
        assert trace.llm_trace is not None
        assert trace.llm_trace.response == "Hello! How can I help?"
        assert trace.llm_trace.model == "ChatOpenAI"

    def test_handler_property(self):
        """Handler should be accessible as a property."""
        tracer = LangChainTracer()
        with tracer:
            handler = tracer.handler
            assert handler is not None
            # Should have callback methods
            assert hasattr(handler, "on_llm_start")
            assert hasattr(handler, "on_llm_end")
            assert hasattr(handler, "on_retriever_start")
            assert hasattr(handler, "on_retriever_end")

    def test_context_limit_passed(self):
        """Custom context limit should be used."""
        tracer = LangChainTracer(context_limit=50_000)
        with tracer:
            handler = tracer.handler
            docs = [_MockDocument("test doc")]
            handler.on_retriever_end(docs)

        trace = tracer.result
        assert trace is not None
        assert trace.context_limit == 50_000

    def test_result_none_before_exit(self):
        """Result should be None before exiting context manager."""
        tracer = LangChainTracer()
        assert tracer.result is None

    def test_timestamp_and_session_id(self):
        """Trace should have timestamp and session_id."""
        tracer = LangChainTracer()
        with tracer:
            handler = tracer.handler
            handler.on_retriever_end([_MockDocument("test")])

        trace = tracer.result
        assert trace is not None
        assert trace.timestamp != ""
        assert trace.session_id != ""

    def test_total_tokens_calculated(self):
        """Total tokens should be sum of component tokens."""
        tracer = LangChainTracer()
        with tracer:
            handler = tracer.handler
            handler.on_retriever_end(
                [
                    _MockDocument("Hello world"),
                    _MockDocument("Another document"),
                ]
            )

        trace = tracer.result
        assert trace is not None
        assert trace.total_tokens > 0
        assert trace.total_tokens == sum(c.token_count for c in trace.components)

    def test_import_error_helpful(self):
        """ImportError should have a helpful message."""
        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.callbacks": None}):
            tracer = LangChainTracer()
            with pytest.raises(ImportError, match="langchain-core is required"):
                tracer.__enter__()
