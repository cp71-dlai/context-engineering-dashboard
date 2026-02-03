"""
Context resources for managing available pools of content.

ContextResource provides a unified interface for managing pools of content
that can be included in the context window. It supports multiple resource
types (RAG, Examples, Chat History, etc.) and integrates with Chroma.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from context_engineering_dashboard.core.trace import ComponentType, ContextComponent


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken if available."""
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4


class ResourceType(Enum):
    """Type of context resource pool.

    Each type maps to a ComponentType for visualization in the context window.
    """

    RAG = "rag"  # Vector DB documents
    EXAMPLE = "example"  # Single-turn Q&A pairs (few-shot)
    CHAT_HISTORY = "chat_history"  # Multi-turn conversations
    SCRATCHPAD = "scratchpad"  # Flat files, offloaded context
    TOOL = "tool"  # Tool signature registry

    def to_component_type(self) -> ComponentType:
        """Map ResourceType to ComponentType for visualization."""
        mapping = {
            ResourceType.RAG: ComponentType.RAG,
            ResourceType.EXAMPLE: ComponentType.EXAMPLE,
            ResourceType.CHAT_HISTORY: ComponentType.CHAT_HISTORY,
            ResourceType.SCRATCHPAD: ComponentType.SCRATCHPAD,
            ResourceType.TOOL: ComponentType.TOOL,
        }
        return mapping[self]


@dataclass
class ResourceItem:
    """A single item in a resource pool.

    Attributes
    ----------
    id : str
        Unique identifier for this item.
    content : str
        The text content.
    token_count : int
        Number of tokens in the content.
    score : float, optional
        Relevance score (0-1) for ranked resources like RAG.
    metadata : dict
        Additional metadata (source, timestamps, etc.).
    """

    id: str
    content: str
    token_count: int
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "token_count": self.token_count,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            token_count=data["token_count"],
            score=data.get("score"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ContextResource:
    """A pool of content items available for inclusion in the context window.

    ContextResource manages a collection of items that can be selected for
    inclusion in the LLM context. It supports querying (for Chroma-backed
    resources) and manual item management.

    Parameters
    ----------
    name : str
        Human-readable name for this resource.
    resource_type : ResourceType
        The type of content (RAG, EXAMPLE, etc.).
    items : List[ResourceItem]
        Available items in this pool.
    selected_ids : Set[str]
        IDs of items currently selected for the context window.
    source : Any, optional
        Reference to underlying data source (Chroma collection, etc.).

    Examples
    --------
    >>> # RAG resource from Chroma
    >>> rag = ContextResource.from_chroma(collection, ResourceType.RAG, "Docs")
    >>> rag.query(query_texts=["How do I..."], n_results=10)
    >>> rag.select(["doc_1", "doc_2"])
    >>>
    >>> # Static examples resource
    >>> examples = ContextResource.from_items(
    ...     [{"id": "ex1", "content": "Q: What is X?\\nA: X is..."}],
    ...     ResourceType.EXAMPLE,
    ...     "Few-Shot"
    ... )
    >>> examples.select_all()
    """

    name: str
    resource_type: ResourceType
    items: List[ResourceItem] = field(default_factory=list)
    selected_ids: Set[str] = field(default_factory=set)
    source: Optional[Any] = None

    # Query metadata
    last_query: Optional[str] = None
    query_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def selected_items(self) -> List[ResourceItem]:
        """Return items that are selected for the context window."""
        return [item for item in self.items if item.id in self.selected_ids]

    @property
    def unselected_items(self) -> List[ResourceItem]:
        """Return items not selected for the context window."""
        return [item for item in self.items if item.id not in self.selected_ids]

    @property
    def total_selected_tokens(self) -> int:
        """Total tokens in selected items."""
        return sum(item.token_count for item in self.selected_items)

    @property
    def total_tokens(self) -> int:
        """Total tokens in all items."""
        return sum(item.token_count for item in self.items)

    def select(self, item_ids: List[str]) -> None:
        """Mark items as selected for the context window."""
        self.selected_ids.update(item_ids)

    def deselect(self, item_ids: List[str]) -> None:
        """Remove items from the context window selection."""
        self.selected_ids.difference_update(item_ids)

    def select_all(self) -> None:
        """Select all items."""
        self.selected_ids = {item.id for item in self.items}

    def clear_selection(self) -> None:
        """Deselect all items."""
        self.selected_ids.clear()

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Query the underlying data source and populate items.

        Only works for Chroma-backed resources. For static resources,
        use from_items() instead.

        Parameters
        ----------
        query_texts : List[str], optional
            Text queries to search for.
        query_embeddings : List[List[float]], optional
            Embedding vectors to search for.
        n_results : int
            Maximum number of results to return.
        where : dict, optional
            Chroma where filter.
        **kwargs
            Additional arguments passed to the collection.

        Returns
        -------
        dict
            Raw Chroma query results.

        Raises
        ------
        ValueError
            If the resource has no Chroma collection source.
        """
        if self.source is None:
            raise ValueError("query() requires a Chroma collection source. Use from_chroma().")

        # Perform query
        include = kwargs.pop("include", ["documents", "distances", "metadatas"])
        results = self.source.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include,
            **kwargs,
        )

        # Store query metadata
        self.last_query = query_texts[0] if query_texts else None
        self.query_params = {"n_results": n_results, "where": where}

        # Convert to ResourceItems
        self.items = []
        if results and results.get("ids"):
            ids = results["ids"][0] if results["ids"] else []
            documents = results.get("documents", [[]])[0] if results.get("documents") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

            for i, doc_id in enumerate(ids):
                content = documents[i] if i < len(documents) else ""
                distance = distances[i] if i < len(distances) else 0.0
                metadata = metadatas[i] if i < len(metadatas) else {}

                # Convert L2 distance to similarity score (0-1)
                score = 1.0 / (1.0 + distance) if distance >= 0 else 1.0

                self.items.append(
                    ResourceItem(
                        id=str(doc_id),
                        content=content,
                        token_count=count_tokens(content),
                        score=round(score, 4),
                        metadata=metadata or {},
                    )
                )

        return results

    def to_components(self) -> List[ContextComponent]:
        """Convert selected items to ContextComponents for the trace.

        Returns
        -------
        List[ContextComponent]
            Components ready for inclusion in a ContextTrace.
        """
        comp_type = self.resource_type.to_component_type()
        components = []

        for item in self.selected_items:
            metadata = {**item.metadata, "resource": self.name}
            if item.score is not None:
                metadata["score"] = item.score

            components.append(
                ContextComponent(
                    id=item.id,
                    type=comp_type,
                    content=item.content,
                    token_count=item.token_count,
                    metadata=metadata,
                )
            )

        return components

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "resource_type": self.resource_type.value,
            "items": [item.to_dict() for item in self.items],
            "selected_ids": list(self.selected_ids),
            "last_query": self.last_query,
            "query_params": self.query_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextResource":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            resource_type=ResourceType(data["resource_type"]),
            items=[ResourceItem.from_dict(i) for i in data.get("items", [])],
            selected_ids=set(data.get("selected_ids", [])),
            last_query=data.get("last_query"),
            query_params=data.get("query_params", {}),
        )

    # Factory methods

    @classmethod
    def from_chroma(
        cls,
        collection: Any,
        resource_type: ResourceType = ResourceType.RAG,
        name: Optional[str] = None,
    ) -> "ContextResource":
        """Create a ContextResource backed by a Chroma collection.

        Parameters
        ----------
        collection : chromadb.Collection
            The Chroma collection to wrap.
        resource_type : ResourceType
            How to treat retrieved documents.
        name : str, optional
            Name for this resource (defaults to collection name).

        Returns
        -------
        ContextResource
            A resource ready for querying.
        """
        resource_name = name
        if resource_name is None:
            # Try to get collection name, fall back to generic
            try:
                resource_name = str(collection.name)
            except Exception:
                resource_name = "Chroma"

        return cls(
            name=resource_name,
            resource_type=resource_type,
            source=collection,
        )

    @classmethod
    def from_items(
        cls,
        items: List[Dict[str, Any]],
        resource_type: ResourceType,
        name: str,
    ) -> "ContextResource":
        """Create a ContextResource from a list of item dictionaries.

        Parameters
        ----------
        items : List[Dict]
            List of dicts with 'id', 'content', and optionally
            'token_count', 'score', 'metadata'.
        resource_type : ResourceType
            The type of resource.
        name : str
            Name for this resource.

        Returns
        -------
        ContextResource
            A static resource with the provided items.
        """
        resource_items = []
        for item in items:
            content = item.get("content", "")
            resource_items.append(
                ResourceItem(
                    id=item["id"],
                    content=content,
                    token_count=item.get("token_count", count_tokens(content)),
                    score=item.get("score"),
                    metadata=item.get("metadata", {}),
                )
            )

        return cls(
            name=name,
            resource_type=resource_type,
            items=resource_items,
        )
