"""LangChain tracer — callback handler integrating with LangChain's callback system."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from context_engineering_dashboard.core.trace import (
    ComponentType,
    ContextComponent,
    ContextTrace,
    LLMTrace,
)
from context_engineering_dashboard.tracers.base_tracer import BaseTracer


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken with fallback."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def _make_handler_class() -> type:
    """Create the callback handler class with deferred imports."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        raise ImportError(
            "langchain-core is required for LangChainTracer. "
            "Install it with: pip install 'context-engineering-dashboard[langchain]'"
        )

    class _TracerCallbackHandler(BaseCallbackHandler):
        """LangChain callback handler that captures LLM and retriever events."""

        def __init__(self) -> None:
            self._llm_starts: List[Dict[str, Any]] = []
            self._llm_ends: List[Dict[str, Any]] = []
            self._retriever_results: List[Dict[str, Any]] = []
            self._components: List[ContextComponent] = []

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            **kwargs: Any,
        ) -> None:
            self._llm_starts.append(
                {
                    "serialized": serialized,
                    "prompts": prompts,
                    "kwargs": kwargs,
                }
            )

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            self._llm_ends.append(
                {
                    "response": response,
                    "kwargs": kwargs,
                }
            )

        def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            **kwargs: Any,
        ) -> None:
            pass

        def on_retriever_end(
            self,
            documents: Sequence[Any],
            **kwargs: Any,
        ) -> None:
            for doc in documents:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                self._retriever_results.append(
                    {
                        "content": content,
                        "metadata": metadata,
                    }
                )

        def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
        ) -> None:
            pass

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
            pass

        def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
            pass

        def on_tool_end(self, output: str, **kwargs: Any) -> None:
            pass

        def build_trace(self, context_limit: int = 128_000) -> ContextTrace:
            """Build a ContextTrace from collected callback data."""
            components: List[ContextComponent] = []

            # Add retriever results as RAG components
            for i, doc in enumerate(self._retriever_results):
                content = doc["content"]
                components.append(
                    ContextComponent(
                        id=f"rag_{i}",
                        type=ComponentType.RAG_DOCUMENT,
                        content=content,
                        token_count=_count_tokens(content),
                        metadata=doc.get("metadata", {}),
                    )
                )

            # Add LLM prompts as components
            for start in self._llm_starts:
                for j, prompt in enumerate(start["prompts"]):
                    components.append(
                        ContextComponent(
                            id=f"prompt_{j}",
                            type=ComponentType.USER_MESSAGE,
                            content=prompt,
                            token_count=_count_tokens(prompt),
                        )
                    )

            # Build LLM trace from last call if available
            llm_trace = None
            if self._llm_starts and self._llm_ends:
                last_start = self._llm_starts[-1]
                last_end = self._llm_ends[-1]
                response_text = ""
                try:
                    response_text = last_end["response"].generations[0][0].text
                except (AttributeError, IndexError):
                    pass

                model = last_start["serialized"].get("name", "unknown")
                llm_trace = LLMTrace(
                    provider="langchain",
                    model=model,
                    messages=[{"role": "user", "content": p} for p in last_start["prompts"]],
                    response=response_text,
                )

            total_tokens = sum(c.token_count for c in components)

            return ContextTrace(
                context_limit=context_limit,
                components=components,
                total_tokens=total_tokens,
                llm_trace=llm_trace,
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=str(uuid.uuid4())[:8],
            )

    return _TracerCallbackHandler


class LangChainTracer(BaseTracer):
    """Context manager providing a LangChain callback handler.

    Usage
    -----
    ::

        with trace_langchain() as tracer:
            chain.invoke({"query": "..."}, config={"callbacks": [tracer.handler]})
        trace = tracer.result
        ctx = ContextWindow(trace=trace)

    Parameters
    ----------
    context_limit : int, optional
        Context window size for the target LLM.
    """

    def __init__(self, context_limit: Optional[int] = None, **kwargs: object) -> None:
        super().__init__(context_limit=context_limit, **kwargs)
        self._handler: Any = None

    @property
    def handler(self) -> Any:
        """The LangChain callback handler. Pass to chain invocations."""
        if self._handler is None:
            handler_cls = _make_handler_class()
            self._handler = handler_cls()
        return self._handler

    def __enter__(self) -> "LangChainTracer":
        # Force handler creation (validates import)
        _ = self.handler
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._handler is not None:
            limit = self._context_limit or 128_000
            self._trace = self._handler.build_trace(context_limit=limit)
