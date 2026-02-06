"""LiteLLM tracer — captures litellm.completion calls via monkey-patching.

LiteLLM provides a unified interface for 100+ LLM providers using the
OpenAI-compatible request/response format. This tracer monkey-patches
litellm.completion() to capture call details, then restores the original
on exit.
"""

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

DEFAULT_CONTEXT_LIMIT = 128_000


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken with fallback."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def _role_to_component_type(role: str) -> ComponentType:
    """Map message role to ComponentType."""
    mapping = {
        "system": ComponentType.SYSTEM_PROMPT,
        "user": ComponentType.USER_MESSAGE,
        "assistant": ComponentType.CHAT_HISTORY,
        "tool": ComponentType.TOOL,
    }
    return mapping.get(role, ComponentType.USER_MESSAGE)


def _extract_provider(model: str) -> str:
    """Extract provider name from LiteLLM model string.

    LiteLLM uses ``provider/model-name`` format, e.g.
    ``anthropic/claude-3-opus``, ``bedrock/anthropic.claude-v2``,
    ``azure/gpt-4``.  Bare model names (no slash) default to ``openai``.
    """
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def _get_context_limit(model: str) -> int:
    """Look up context limit using ``litellm.get_model_info()``.

    Falls back to :data:`DEFAULT_CONTEXT_LIMIT` (128 000) when the lookup
    fails for any reason (missing model, import error, etc.).
    """
    try:
        import litellm

        info = litellm.get_model_info(model)
        return int(info.get("max_input_tokens", DEFAULT_CONTEXT_LIMIT))
    except Exception:
        return DEFAULT_CONTEXT_LIMIT


class LiteLLMTracer(BaseTracer):
    """Context manager that monkey-patches ``litellm.completion``.

    Usage
    -----
    ::

        import litellm
        from context_engineering_dashboard import trace_litellm, ContextWindow

        with trace_litellm() as tracer:
            response = litellm.completion(
                model="anthropic/claude-3-opus",
                messages=[{"role": "user", "content": "Hello!"}],
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
        self._original_completion: Any = None
        self._captures: List[Dict[str, Any]] = []

    def __enter__(self) -> "LiteLLMTracer":
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for LiteLLMTracer. "
                "Install it with: pip install 'context-engineering-dashboard[litellm]'"
            )

        self._original_completion = litellm.completion
        tracer = self

        def patched_completion(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            response = tracer._original_completion(*args, **kwargs)
            elapsed = (time.time() - start) * 1000
            # Merge positional args into kwargs for uniform access
            call_kwargs: Dict[str, Any] = dict(kwargs)
            if args:
                if len(args) >= 1 and "model" not in call_kwargs:
                    call_kwargs["model"] = args[0]
                if len(args) >= 2 and "messages" not in call_kwargs:
                    call_kwargs["messages"] = args[1]
            tracer._capture(call_kwargs, response, elapsed)
            return response

        litellm.completion = patched_completion  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        try:
            import litellm

            if self._original_completion is not None:
                litellm.completion = self._original_completion  # type: ignore[assignment]
                self._original_completion = None
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
            context_limit = _get_context_limit(model)

        provider = _extract_provider(model)

        # Build components from messages
        components: List[ContextComponent] = []
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

        # Extract response (LiteLLM returns OpenAI-compatible ModelResponse)
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
                total_tokens = response.usage.prompt_tokens
        except (AttributeError, IndexError):
            pass

        trace = Trace(
            provider=provider,
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
