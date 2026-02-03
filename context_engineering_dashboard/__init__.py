"""
Context Engineering Dashboard — Visualize and debug LLM context windows in Jupyter notebooks.
"""

__version__ = "0.1.0"

from context_engineering_dashboard.core.context_diff import ContextDiff
from context_engineering_dashboard.core.context_window import ContextBuilder, ContextWindow
from context_engineering_dashboard.core.resource import (
    ContextResource,
    ResourceItem,
    ResourceType,
)
from context_engineering_dashboard.core.trace import (
    ComponentType,
    ContextComponent,
    ContextTrace,
    EmbeddingTrace,
    ToolCall,
    Trace,
)


def trace_openai(**kwargs):  # type: ignore[no-untyped-def]
    """Convenience function to create an OpenAI tracer context manager."""
    from context_engineering_dashboard.tracers.openai_tracer import OpenAITracer

    return OpenAITracer(**kwargs)


def trace_langchain(**kwargs):  # type: ignore[no-untyped-def]
    """Convenience function to create a LangChain tracer context manager."""
    from context_engineering_dashboard.tracers.langchain_tracer import LangChainTracer

    return LangChainTracer(**kwargs)


def trace_litellm(**kwargs):  # type: ignore[no-untyped-def]
    """Convenience function to create a LiteLLM tracer context manager."""
    from context_engineering_dashboard.tracers.litellm_tracer import LiteLLMTracer

    return LiteLLMTracer(**kwargs)


__all__ = [
    "ComponentType",
    "ContextComponent",
    "ContextTrace",
    "Trace",
    "EmbeddingTrace",
    "ToolCall",
    "ContextResource",
    "ResourceType",
    "ResourceItem",
    "ContextBuilder",
    "ContextWindow",  # Deprecated alias for ContextBuilder
    "ContextDiff",
    "trace_openai",
    "trace_langchain",
    "trace_litellm",
    "__version__",
]
