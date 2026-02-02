"""
Trace data structures for context engineering visualization.

These dataclasses define the schema for capturing and storing
traces from LLM calls, Chroma queries, and context assembly.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


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
class ChromaRetrievalResult:
    """A document returned from a Chroma query."""

    id: str
    content: str
    token_count: int
    score: float
    selected: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection: str = ""
    embedding_model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "token_count": self.token_count,
            "score": self.score,
            "selected": self.selected,
            "metadata": self.metadata,
            "collection": self.collection,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChromaRetrievalResult":
        return cls(
            id=data["id"],
            content=data["content"],
            token_count=data["token_count"],
            score=data["score"],
            selected=data["selected"],
            metadata=data.get("metadata", {}),
            collection=data.get("collection", ""),
            embedding_model=data.get("embedding_model", ""),
        )


@dataclass
class ChromaQuery:
    """Captures a Chroma retrieval operation."""

    collection: str
    query_text: str
    query_embedding: Optional[List[float]] = None
    n_results: int = 10
    where_filter: Optional[Dict] = None
    results: List[ChromaRetrievalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection": self.collection,
            "query_text": self.query_text,
            "query_embedding": self.query_embedding,
            "n_results": self.n_results,
            "where_filter": self.where_filter,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChromaQuery":
        return cls(
            collection=data["collection"],
            query_text=data["query_text"],
            query_embedding=data.get("query_embedding"),
            n_results=data.get("n_results", 10),
            where_filter=data.get("where_filter"),
            results=[ChromaRetrievalResult.from_dict(r) for r in data.get("results", [])],
        )

    @property
    def selected_results(self) -> List[ChromaRetrievalResult]:
        """Return only the results that were selected for the context window."""
        return [r for r in self.results if r.selected]

    @property
    def unselected_results(self) -> List[ChromaRetrievalResult]:
        """Return results that were not selected."""
        return [r for r in self.results if not r.selected]


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
class LLMTrace:
    """Trace of a language model call."""

    provider: str
    model: str
    messages: List[Dict[str, str]]
    response: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "messages": self.messages,
            "response": self.response,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMTrace":
        return cls(
            provider=data["provider"],
            model=data["model"],
            messages=data["messages"],
            response=data["response"],
            tool_calls=[ToolCall.from_dict(t) for t in data.get("tool_calls", [])],
            usage=data.get("usage", {}),
            latency_ms=data.get("latency_ms", 0.0),
        )


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
    """Complete trace of a context engineering operation."""

    context_limit: int
    components: List[ContextComponent]
    total_tokens: int

    chroma_queries: List[ChromaQuery] = field(default_factory=list)
    llm_trace: Optional[LLMTrace] = None
    embedding_traces: List[EmbeddingTrace] = field(default_factory=list)

    timestamp: str = ""
    session_id: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_limit": self.context_limit,
            "total_tokens": self.total_tokens,
            "components": [c.to_dict() for c in self.components],
            "chroma_queries": [q.to_dict() for q in self.chroma_queries],
            "llm_trace": self.llm_trace.to_dict() if self.llm_trace else None,
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
            chroma_queries=[ChromaQuery.from_dict(q) for q in data.get("chroma_queries", [])],
            llm_trace=LLMTrace.from_dict(data["llm_trace"]) if data.get("llm_trace") else None,
            embedding_traces=[
                EmbeddingTrace.from_dict(e) for e in data.get("embedding_traces", [])
            ],
            timestamp=data.get("timestamp", ""),
            session_id=data.get("session_id", ""),
            tags=data.get("tags", []),
        )

    def to_json(self, path: str | Path) -> None:
        """Save trace to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "ContextTrace":
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
        return (self.total_tokens / self.context_limit) * 100

    def get_components_by_type(self, component_type: ComponentType) -> List[ContextComponent]:
        """Filter components by type."""
        return [c for c in self.components if c.type == component_type]
