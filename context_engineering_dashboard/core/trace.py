"""
Trace data structures for context engineering visualization.

These dataclasses define the schema for capturing and storing
traces from LLM calls and context assembly.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ComponentType(Enum):
    """Type of context component."""

    SYSTEM_PROMPT = "system_prompt"
    USER_MESSAGE = "user_message"
    CHAT_HISTORY = "chat_history"
    RAG = "rag"
    TOOL = "tool"
    EXAMPLE = "example"
    SCRATCHPAD = "scratchpad"


@dataclass
class ContextComponent:
    """A single component in the context window."""

    id: str
    type: ComponentType
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextComponent":
        return cls(
            id=data["id"],
            type=ComponentType(data["type"]),
            content=data["content"],
            token_count=data["token_count"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ToolCall:
    """A tool invocation by the LLM."""

    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=data["name"],
            arguments=data["arguments"],
            result=data.get("result"),
        )


@dataclass
class Trace:
    """Trace of a language model call with explicit schema reference.

    This is the standardized format for capturing LLM interactions
    across all providers (OpenAI, Anthropic, Google, etc.).

    Attributes
    ----------
    schema_ref : str
        URL to the JSON Schema for validation.
    schema_version : str
        Version of the schema.
    provider : str
        LLM provider name (openai, anthropic, google, etc.).
    model : str
        Model identifier (e.g., "gpt-4o", "claude-3-sonnet").
    messages : List[Dict[str, str]]
        Full message history sent to the model.
    response : str
        Model's text response.
    tool_calls : List[ToolCall]
        Any tool invocations made by the model.
    usage : Dict[str, int]
        Token usage: prompt_tokens, completion_tokens, total_tokens.
    latency_ms : float
        Request latency in milliseconds.
    timestamp : str
        ISO 8601 timestamp of the call.
    session_id : str
        Session identifier for grouping traces.
    """

    # Schema reference
    schema_ref: str = field(
        default="https://github.com/cp71-dlai/context-engineering-dashboard/schemas/trace.json"
    )
    schema_version: str = "1.0.0"

    # Core fields
    provider: str = ""
    model: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    response: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

    # Metadata
    timestamp: str = ""
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "$schema": self.schema_ref,
            "schema_version": self.schema_version,
            "provider": self.provider,
            "model": self.model,
            "messages": self.messages,
            "response": self.response,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trace":
        return cls(
            schema_ref=data.get("$schema", cls.schema_ref),
            schema_version=data.get("schema_version", "1.0.0"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            messages=data.get("messages", []),
            response=data.get("response", ""),
            tool_calls=[ToolCall.from_dict(t) for t in data.get("tool_calls", [])],
            usage=data.get("usage", {}),
            latency_ms=data.get("latency_ms", 0.0),
            timestamp=data.get("timestamp", ""),
            session_id=data.get("session_id", ""),
        )

    def validate(self, strict: bool = False) -> bool:
        """Validate trace against JSON schema.

        Parameters
        ----------
        strict : bool
            If True, raise ValidationError on failure.
            If False, return bool.

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        try:
            import jsonschema

            # Load bundled schema
            schema_path = Path(__file__).parent.parent.parent / "schemas" / "trace-schema.json"
            if schema_path.exists():
                with open(schema_path) as f:
                    schema = json.load(f)
                jsonschema.validate(self.to_dict(), schema)
            return True
        except ImportError:
            if strict:
                raise ImportError("jsonschema required for validation")
            return True  # Skip validation if jsonschema not installed
        except Exception:
            if strict:
                raise
            return False

    def to_json(self, path: Union[str, Path]) -> None:
        """Save trace to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Trace":
        """Load trace from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class EmbeddingTrace:
    """Trace of an embedding model call."""

    provider: str
    model: str
    input_text: str
    embedding: List[float]
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_text": self.input_text,
            "embedding": self.embedding,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingTrace":
        return cls(
            provider=data["provider"],
            model=data["model"],
            input_text=data["input_text"],
            embedding=data["embedding"],
            latency_ms=data.get("latency_ms", 0.0),
        )


@dataclass
class ContextTrace:
    """Complete trace of a context engineering operation.

    Represents a snapshot of the context window including all components
    and optional LLM call trace.

    Attributes
    ----------
    context_limit : int
        Maximum context window size in tokens.
    components : List[ContextComponent]
        Components in the context window (in order).
    total_tokens : int
        Total tokens used.
    trace : Trace, optional
        The LLM call trace.
    embedding_traces : List[EmbeddingTrace]
        Embedding model call traces.
    timestamp : str
        ISO 8601 timestamp.
    session_id : str
        Session identifier for grouping.
    tags : List[str]
        User-defined filter tags.
    """

    context_limit: int
    components: List[ContextComponent]
    total_tokens: int

    trace: Optional[Trace] = None
    embedding_traces: List[EmbeddingTrace] = field(default_factory=list)

    timestamp: str = ""
    session_id: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_limit": self.context_limit,
            "total_tokens": self.total_tokens,
            "components": [c.to_dict() for c in self.components],
            "trace": self.trace.to_dict() if self.trace else None,
            "embedding_traces": [e.to_dict() for e in self.embedding_traces],
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextTrace":
        return cls(
            context_limit=data["context_limit"],
            total_tokens=data["total_tokens"],
            components=[ContextComponent.from_dict(c) for c in data["components"]],
            trace=Trace.from_dict(data["trace"]) if data.get("trace") else None,
            embedding_traces=[
                EmbeddingTrace.from_dict(e) for e in data.get("embedding_traces", [])
            ],
            timestamp=data.get("timestamp", ""),
            session_id=data.get("session_id", ""),
            tags=data.get("tags", []),
        )

    def to_json(self, path: Union[str, Path]) -> None:
        """Save trace to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ContextTrace":
        """Load trace from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @property
    def unused_tokens(self) -> int:
        """Calculate unused context window space."""
        return self.context_limit - self.total_tokens

    @property
    def utilization(self) -> float:
        """Context window utilization as a percentage (0-100)."""
        if self.context_limit == 0:
            return 0.0
        return (self.total_tokens / self.context_limit) * 100

    def get_components_by_type(self, component_type: ComponentType) -> List[ContextComponent]:
        """Filter components by type."""
        return [c for c in self.components if c.type == component_type]
