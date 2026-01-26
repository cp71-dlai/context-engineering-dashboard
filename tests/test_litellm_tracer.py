"""Tests for LiteLLM tracer (all mocked, no real API calls)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from context_engineering_dashboard.core.trace import ComponentType
from context_engineering_dashboard.tracers.litellm_tracer import (
    DEFAULT_CONTEXT_LIMIT,
    LiteLLMTracer,
    _count_tokens,
    _extract_provider,
    _get_context_limit,
    _role_to_component_type,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_response(
    content="Hello!",
    model="gpt-4o",
    prompt_tokens=100,
    completion_tokens=20,
):
    """Create a mock LiteLLM ModelResponse (OpenAI-compatible)."""
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
        model="anthropic/claude-3-opus",
        choices=[choice],
        usage=usage,
    )


# ---------------------------------------------------------------------------
# _extract_provider
# ---------------------------------------------------------------------------


class TestExtractProvider:
    def test_anthropic_prefix(self):
        assert _extract_provider("anthropic/claude-3-opus") == "anthropic"

    def test_bedrock_prefix(self):
        assert _extract_provider("bedrock/anthropic.claude-v2") == "bedrock"

    def test_azure_prefix(self):
        assert _extract_provider("azure/gpt-4") == "azure"

    def test_vertex_prefix(self):
        assert _extract_provider("vertex_ai/gemini-pro") == "vertex_ai"

    def test_cohere_prefix(self):
        assert _extract_provider("cohere/command-r-plus") == "cohere"

    def test_bare_model_defaults_openai(self):
        assert _extract_provider("gpt-4o") == "openai"

    def test_empty_string(self):
        assert _extract_provider("") == "openai"


# ---------------------------------------------------------------------------
# _get_context_limit
# ---------------------------------------------------------------------------


class TestGetContextLimit:
    def test_uses_litellm_model_info(self):
        mock_litellm = MagicMock()
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 200_000}
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = _get_context_limit("anthropic/claude-3-opus")
        mock_litellm.get_model_info.assert_called_once_with("anthropic/claude-3-opus")
        assert result == 200_000

    def test_fallback_when_key_missing(self):
        mock_litellm = MagicMock()
        mock_litellm.get_model_info.return_value = {}
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = _get_context_limit("some-model")
        assert result == DEFAULT_CONTEXT_LIMIT

    def test_fallback_on_exception(self):
        mock_litellm = MagicMock()
        mock_litellm.get_model_info.side_effect = Exception("Unknown model")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = _get_context_limit("unknown/model")
        assert result == DEFAULT_CONTEXT_LIMIT

    def test_default_context_limit_value(self):
        assert DEFAULT_CONTEXT_LIMIT == 128_000


# ---------------------------------------------------------------------------
# _role_to_component_type
# ---------------------------------------------------------------------------


class TestRoleToComponentType:
    def test_system(self):
        assert _role_to_component_type("system") == ComponentType.SYSTEM_PROMPT

    def test_user(self):
        assert _role_to_component_type("user") == ComponentType.USER_MESSAGE

    def test_assistant(self):
        assert _role_to_component_type("assistant") == ComponentType.CHAT_HISTORY

    def test_tool(self):
        assert _role_to_component_type("tool") == ComponentType.TOOL

    def test_unknown_defaults_user(self):
        assert _role_to_component_type("unknown") == ComponentType.USER_MESSAGE


# ---------------------------------------------------------------------------
# _count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_returns_positive_int(self):
        tokens = _count_tokens("Hello, world!")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_empty_string(self):
        assert _count_tokens("") == 0


# ---------------------------------------------------------------------------
# LiteLLMTracer
# ---------------------------------------------------------------------------


class TestLiteLLMTracer:
    def test_captures_messages(self):
        tracer = LiteLLMTracer()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there!"},
        ]
        response = _mock_response()

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 100.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert len(trace.components) == 2
        assert trace.components[0].type == ComponentType.SYSTEM_PROMPT
        assert trace.components[1].type == ComponentType.USER_MESSAGE

    def test_captures_response(self):
        tracer = LiteLLMTracer()
        messages = [{"role": "user", "content": "Hello"}]
        response = _mock_response(content="Hi there!")

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert trace.llm_trace is not None
        assert trace.llm_trace.response == "Hi there!"

    def test_captures_usage(self):
        tracer = LiteLLMTracer()
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
        tracer = LiteLLMTracer()
        messages = [{"role": "user", "content": "Search for test"}]
        response = _mock_response_with_tool_calls()

        tracer._capture({"model": "anthropic/claude-3-opus", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert trace.llm_trace is not None
        assert len(trace.llm_trace.tool_calls) == 1
        assert trace.llm_trace.tool_calls[0].name == "search"

    def test_provider_extracted_from_model(self):
        tracer = LiteLLMTracer()
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response()

        tracer._capture({"model": "anthropic/claude-3-opus", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.llm_trace is not None
        assert tracer.result.llm_trace.provider == "anthropic"

    def test_provider_defaults_openai_for_bare_model(self):
        tracer = LiteLLMTracer()
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response()

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.llm_trace is not None
        assert tracer.result.llm_trace.provider == "openai"

    @patch("context_engineering_dashboard.tracers.litellm_tracer._get_context_limit")
    def test_auto_detects_context_limit(self, mock_get_limit):
        mock_get_limit.return_value = 200_000
        tracer = LiteLLMTracer()
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response()

        tracer._capture({"model": "anthropic/claude-3-opus", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.context_limit == 200_000
        mock_get_limit.assert_called_once_with("anthropic/claude-3-opus")

    def test_custom_limit_override(self):
        tracer = LiteLLMTracer(context_limit=50_000)
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response()

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.context_limit == 50_000

    def test_restores_original_function(self):
        mock_litellm = MagicMock()
        original_completion = MagicMock()
        mock_litellm.completion = original_completion

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            tracer = LiteLLMTracer()
            tracer._original_completion = original_completion
            tracer.__exit__(None, None, None)
            assert mock_litellm.completion == original_completion

    def test_import_error_helpful(self):
        with patch.dict("sys.modules", {"litellm": None}):
            tracer = LiteLLMTracer()
            with pytest.raises(ImportError, match="litellm is required"):
                tracer.__enter__()

    def test_latency_captured(self):
        tracer = LiteLLMTracer()
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response()

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 1234.5)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.llm_trace is not None
        assert tracer.result.llm_trace.latency_ms == 1234.5

    def test_result_none_before_exit(self):
        tracer = LiteLLMTracer()
        assert tracer.result is None

    def test_timestamp_and_session_id(self):
        tracer = LiteLLMTracer()
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response()

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert trace.timestamp != ""
        assert trace.session_id != ""

    def test_model_stored_in_trace(self):
        tracer = LiteLLMTracer()
        messages = [{"role": "user", "content": "Hi"}]
        response = _mock_response()

        tracer._capture(
            {"model": "bedrock/anthropic.claude-v2", "messages": messages}, response, 50.0
        )
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        assert tracer.result is not None
        assert tracer.result.llm_trace is not None
        assert tracer.result.llm_trace.model == "bedrock/anthropic.claude-v2"
        assert tracer.result.llm_trace.provider == "bedrock"

    def test_context_manager_patches_and_restores(self):
        mock_litellm = MagicMock()
        original_fn = MagicMock(return_value=_mock_response())
        mock_litellm.completion = original_fn

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            tracer = LiteLLMTracer()
            tracer.__enter__()

            # After enter, completion should be patched (different function)
            assert mock_litellm.completion != original_fn

            # Call the patched function
            mock_litellm.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
            )

            assert len(tracer._captures) == 1

            tracer.__exit__(None, None, None)

            # After exit, should be restored
            assert mock_litellm.completion == original_fn

    def test_handles_none_content(self):
        tracer = LiteLLMTracer()
        messages = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Hi"},
        ]
        response = _mock_response()

        tracer._capture({"model": "gpt-4o", "messages": messages}, response, 50.0)
        tracer._trace = tracer._build_trace(tracer._captures[-1])

        trace = tracer.result
        assert trace is not None
        assert trace.components[0].content == ""


# ---------------------------------------------------------------------------
# trace_litellm convenience function
# ---------------------------------------------------------------------------


class TestTraceLiteLLMConvenience:
    def test_trace_litellm_returns_tracer(self):
        from context_engineering_dashboard import trace_litellm

        tracer = trace_litellm()
        assert isinstance(tracer, LiteLLMTracer)

    def test_trace_litellm_passes_context_limit(self):
        from context_engineering_dashboard import trace_litellm

        tracer = trace_litellm(context_limit=32_000)
        assert tracer._context_limit == 32_000
