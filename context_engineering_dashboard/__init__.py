"""
Context Engineering Dashboard — Visualize and debug LLM context windows in Jupyter notebooks.
"""

__version__ = "0.1.0"

from context_engineering_dashboard.core.context_diff import ContextDiff
from context_engineering_dashboard.core.context_window import ContextWindow
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


def trace_chroma(collection, **kwargs):  # type: ignore[no-untyped-def]
    """Convenience function to wrap a Chroma collection for tracing."""
    from context_engineering_dashboard.tracers.chroma_tracer import TracedCollection

    return TracedCollection(collection, **kwargs)


def trace_openai(**kwargs):  # type: ignore[no-untyped-def]
    """Convenience function to create an OpenAI tracer context manager."""
    from context_engineering_dashboard.tracers.openai_tracer import OpenAITracer

    return OpenAITracer(**kwargs)


def trace_langchain(**kwargs):  # type: ignore[no-untyped-def]
    """Convenience function to create a LangChain tracer context manager."""
    from context_engineering_dashboard.tracers.langchain_tracer import LangChainTracer

    return LangChainTracer(**kwargs)


__all__ = [
    "ComponentType",
    "ContextComponent",
    "ContextTrace",
    "ChromaQuery",
    "ChromaRetrievalResult",
    "LLMTrace",
    "EmbeddingTrace",
    "ToolCall",
    "ContextWindow",
    "ContextDiff",
    "trace_chroma",
    "trace_openai",
    "trace_langchain",
    "__version__",
]
