"""OpenAI tracer — captures chat completion calls via monkey-patching."""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from context_engineering_dashboard.core.trace import (
    ComponentType,
    ContextComponent,
    ContextTrace,
    ToolCall,
    Trace,
)
from context_engineering_dashboard.tracers.base_tracer import BaseTracer

# Known context window limits per model
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-16k": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3-mini": 200_000,
}


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken with fallback."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def _role_to_component_type(role: str) -> ComponentType:
    """Map OpenAI message role to ComponentType."""
    mapping = {
        "system": ComponentType.SYSTEM_PROMPT,
        "user": ComponentType.USER_MESSAGE,
        "assistant": ComponentType.CHAT_HISTORY,
        "tool": ComponentType.TOOL,
    }
    return mapping.get(role, ComponentType.USER_MESSAGE)


class OpenAITracer(BaseTracer):
    """Context manager that monkey-patches openai.chat.completions.create.

    Usage
    -----
    ::

        with trace_openai() as tracer:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[...],
            )
        trace = tracer.result
        ctx = ContextWindow(trace=trace)

    Parameters
    ----------
    context_limit : int, optional
        Override the auto-detected context limit.
    """

    def __init__(self, context_limit: Optional[int] = None, **kwargs: object) -> None:
        super().__init__(context_limit=context_limit, **kwargs)
        self._original_create: Any = None
        self._original_acreate: Any = None
        self._captures: List[Dict[str, Any]] = []

    def __enter__(self) -> "OpenAITracer":
        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            raise ImportError(
                "openai is required for OpenAITracer. "
                "Install it with: pip install 'context-engineering-dashboard[openai]'"
            )

        self._original_create = Completions.create
        tracer = self

        def patched_create(self_completions: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.time()
            response = tracer._original_create(self_completions, *args, **kwargs)
            elapsed = (time.time() - start) * 1000
            tracer._capture(kwargs, response, elapsed)
            return response

        Completions.create = patched_create  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        try:
            from openai.resources.chat.completions import Completions

            if self._original_create is not None:
                Completions.create = self._original_create  # type: ignore[assignment]
                self._original_create = None
        except ImportError:
            pass

        if self._captures:
            self._trace = self._build_trace(self._captures[-1])

    def _capture(self, kwargs: Dict[str, Any], response: Any, elapsed_ms: float) -> None:
        """Capture a single API call."""
        self._captures.append(
            {
                "kwargs": kwargs,
                "response": response,
                "elapsed_ms": elapsed_ms,
            }
        )

    def _build_trace(self, capture: Dict[str, Any]) -> ContextTrace:
        """Build a ContextTrace from a captured API call."""
        kwargs = capture["kwargs"]
        response = capture["response"]
        elapsed_ms = capture["elapsed_ms"]

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        # Auto-detect context limit
        context_limit = self._context_limit
        if context_limit is None:
            for prefix, limit in MODEL_CONTEXT_LIMITS.items():
                if model.startswith(prefix):
                    context_limit = limit
                    break
            if context_limit is None:
                context_limit = 128_000

        # Build components from messages
        components = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content is None:
                content = ""
            ct = _role_to_component_type(role)
            comp_id = f"{role}_{i}"
            token_count = _count_tokens(content)
            components.append(
                ContextComponent(
                    id=comp_id,
                    type=ct,
                    content=content,
                    token_count=token_count,
                )
            )

        total_tokens = sum(c.token_count for c in components)

        # Extract response text
        response_text = ""
        tool_calls_list: List[ToolCall] = []
        usage: Dict[str, int] = {}

        try:
            choice = response.choices[0]
            response_text = choice.message.content or ""

            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls_list.append(
                        ToolCall(
                            name=tc.function.name,
                            arguments=(
                                {"raw": tc.function.arguments}
                                if isinstance(tc.function.arguments, str)
                                else tc.function.arguments
                            ),
                        )
                    )

            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                # Use actual prompt tokens if available
                total_tokens = response.usage.prompt_tokens
        except (AttributeError, IndexError):
            pass

        trace = Trace(
            provider="openai",
            model=model,
            messages=[
                {"role": m.get("role", ""), "content": m.get("content", "") or ""} for m in messages
            ],
            response=response_text,
            tool_calls=tool_calls_list,
            usage=usage,
            latency_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=str(uuid.uuid4())[:8],
        )

        return ContextTrace(
            context_limit=context_limit,
            components=components,
            total_tokens=total_tokens,
            trace=trace,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=str(uuid.uuid4())[:8],
        )
