"""Tests for OpenAI tracer (all mocked, no real API calls)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from context_engineering_dashboard.core.trace import ComponentType
from context_engineering_dashboard.tracers.openai_tracer import (
    MODEL_CONTEXT_LIMITS,
    OpenAITracer,
    _count_tokens,
    _role_to_component_type,
)


def _mock_response(content="Hello!", model="gpt-4o", prompt_tokens=100, completion_tokens=20):
    """Create a mock OpenAI response object."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(
        id="chatcmpl-test",
        model=model,
        choices=[choice],
        usage=usage,
    )


def _mock_response_with_tool_calls():
    """Create a mock response with tool calls."""
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="search", arguments='{"query": "test"}'),
        type="function",
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60)
    return SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-4o",
        choices=[choice],
        usage=usage,
    )


class TestOpenAITracer:
    def test_captures_messages(self):
        """Tracer captures messages from the API call."""
        tracer = OpenAITracer()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there!"},
        ]
        response = _mock_response()

        # Simulate a capture
        tracer._capture(
            {"model": "gpt-4o", "messages": messages},
            response,
            100.0,
        )
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert len(trace.components) == 2
        assert trace.components[0].type == ComponentType.SYSTEM_PROMPT
        assert trace.components[1].type == ComponentType.USER_MESSAGE

    def test_captures_response(self):
        """Tracer captures response text."""
        tracer = OpenAITracer()
        messages = [{"role": "user", "content": "Hello"}]
        response = _mock_response(content="Hi there!")

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert trace.llm_trace is not None
        assert trace.llm_trace.response == "Hi there!"

    def test_captures_usage(self):
        """Tracer captures token usage statistics."""
        tracer = OpenAITracer()
        messages = [{"role": "user", "content": "Hello"}]
        response = _mock_response(prompt_tokens=150, completion_tokens=30)

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert trace.llm_trace is not None
        assert trace.llm_trace.usage["prompt_tokens"] == 150
        assert trace.llm_trace.usage["completion_tokens"] == 30
        assert trace.llm_trace.usage["total_tokens"] == 180

    def test_captures_tool_calls(self):
        """Tracer captures tool calls from the response."""
        tracer = OpenAITracer()
        messages = [{"role": "user", "content": "Search for test"}]
        response = _mock_response_with_tool_calls()

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert trace.llm_trace is not None
        assert len(trace.llm_trace.tool_calls) == 1
        assert trace.llm_trace.tool_calls[0].name == "search"

    def test_auto_detects_context_limit(self):
        """Tracer auto-detects context limit from model name."""
        tracer = OpenAITracer()
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response(model="gpt-4o")

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.context_limit == 128_000

    def test_custom_limit_override(self):
        """Custom context limit overrides auto-detection."""
        tracer = OpenAITracer(context_limit=50_000)
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response(model="gpt-4o")

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.context_limit == 50_000

    def test_restores_original_method(self):
        """Tracer restores original create method on exit."""
        # Create mock openai module structure
        mock_completions_cls = MagicMock()
        original_create = MagicMock()
        mock_completions_cls.create = original_create

        with patch.dict("sys.modules", {"openai": MagicMock()}):
            with patch(
                "context_engineering_dashboard.tracers.openai_tracer.Completions",
                mock_completions_cls,
                create=True,
            ):
                tracer = OpenAITracer()
                tracer._original_create = original_create

                # Simulate exit
                tracer.__exit__(None, None, None)

    def test_import_error_helpful(self):
        """ImportError should have a helpful message."""
        with patch.dict("sys.modules", {"openai": None}):
            tracer = OpenAITracer()
            with pytest.raises(ImportError, match="openai is required"):
                tracer.__enter__()

    def test_latency_captured(self):
        """Latency should be captured in the LLM trace."""
        tracer = OpenAITracer()
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response()

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 1234.5)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.llm_trace is not None
        assert tracer.result.llm_trace.latency_ms == 1234.5

    def test_model_context_limits_known(self):
        """Known models should have context limits defined."""
        assert "gpt-4o" in MODEL_CONTEXT_LIMITS
        assert "gpt-4" in MODEL_CONTEXT_LIMITS
        assert "gpt-3.5-turbo" in MODEL_CONTEXT_LIMITS

    def test_model_context_limits_o1(self):
        """o1 model should have 200K context limit."""
        assert MODEL_CONTEXT_LIMITS["o1"] == 200_000


def test_role_to_component_type():
    assert _role_to_component_type("system") == ComponentType.SYSTEM_PROMPT
    assert _role_to_component_type("user") == ComponentType.USER_MESSAGE
    assert _role_to_component_type("assistant") == ComponentType.CHAT_HISTORY
    assert _role_to_component_type("tool") == ComponentType.TOOL
    assert _role_to_component_type("unknown") == ComponentType.USER_MESSAGE


def test_count_tokens():
    """Token counting should work."""
    tokens = _count_tokens("Hello, world!")
    assert tokens > 0
    assert isinstance(tokens, int)
