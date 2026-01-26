"""
Chroma tracer for capturing retrieval operations.

Wraps Chroma collections to capture query details, scores, and selection decisions.
Provides special visualization support for RAG workflows.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import uuid

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from context_engineering_dashboard.core.trace import (
    ContextTrace,
    ContextComponent,
    ComponentType,
    ChromaQuery,
    ChromaRetrievalResult,
)


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens in text using tiktoken."""
    if not HAS_TIKTOKEN:
        return len(text) // 4
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


class TracedCollection:
    """
    Wrapper around a Chroma collection that traces query operations.
    
    Captures:
    - Query text and embedding
    - All returned documents with scores
    - Selection decisions (which docs made it to context)
    - Document metadata
    
    Usage:
        traced = trace_chroma(collection)
        results = traced.query(query_texts=["How do I..."], n_results=10)
        
        # Mark which results were selected for context
        traced.mark_selected(["doc_1", "doc_2"])
        
        # Get the trace
        trace = traced.get_trace()
        ctx = ContextWindow(trace=trace, show_available_pool=True)
    
    Parameters
    ----------
    collection : chromadb.Collection
        The Chroma collection to wrap.
    embedding_model : str
        Name of the embedding model (for metadata).
    context_limit : int
        Context window limit for the target LLM.
    """
    
    def __init__(
        self,
        collection,
        embedding_model: str = "text-embedding-3-small",
        context_limit: int = 128_000,
    ):
        self._collection = collection
        self._embedding_model = embedding_model
        self._context_limit = context_limit
        self._queries: List[ChromaQuery] = []
        self._selected_ids: Set[str] = set()
        self._components: List[ContextComponent] = []
    
    @property
    def name(self) -> str:
        """Get collection name."""
        return self._collection.name
    
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Query the collection and trace the operation.
        
        Wraps the underlying collection.query() method and captures
        all returned results for visualization.
        """
        # Default include to get documents and distances
        if include is None:
            include = ["documents", "distances", "metadatas"]
        
        # Perform the actual query
        results = self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
            **kwargs,
        )
        
        # Build ChromaQuery trace
        query_text = query_texts[0] if query_texts else ""
        query_embedding = query_embeddings[0] if query_embeddings else None
        
        retrieval_results = []
        
        # Process results (handle batched structure)
        if results and results.get("ids"):
            ids = results["ids"][0] if results["ids"] else []
            documents = results.get("documents", [[]])[0] if results.get("documents") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            
            for i, doc_id in enumerate(ids):
                content = documents[i] if i < len(documents) else ""
                distance = distances[i] if i < len(distances) else 0.0
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                # Convert distance to similarity score (assuming cosine)
                # Chroma returns L2 distance by default, smaller = more similar
                # For cosine, distance is 1 - similarity, so similarity = 1 - distance
                # For L2, we approximate with 1 / (1 + distance)
                score = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
                
                retrieval_results.append(ChromaRetrievalResult(
                    id=str(doc_id),
                    content=content,
                    token_count=count_tokens(content),
                    score=round(score, 4),
                    selected=str(doc_id) in self._selected_ids,
                    metadata=metadata or {},
                    collection=self.name,
                    embedding_model=self._embedding_model,
                ))
        
        chroma_query = ChromaQuery(
            collection=self.name,
            query_text=query_text,
            query_embedding=query_embedding,
            n_results=n_results,
            where_filter=where,
            results=retrieval_results,
        )
        
        self._queries.append(chroma_query)
        
        return results
    
    def mark_selected(self, doc_ids: List[str]) -> None:
        """
        Mark documents as selected for the context window.
        
        Call this after deciding which retrieved docs to include
        in the final prompt.
        
        Parameters
        ----------
        doc_ids : List[str]
            IDs of documents to mark as selected.
        """
        self._selected_ids.update(doc_ids)
        
        # Update all query results
        for query in self._queries:
            for result in query.results:
                result.selected = result.id in self._selected_ids
    
    def add_component(self, component: ContextComponent) -> None:
        """
        Add a non-RAG component to the trace.
        
        Use this to add system prompts, user messages, etc.
        that aren't from Chroma retrieval.
        """
        self._components.append(component)
    
    def add_system_prompt(self, content: str, **metadata) -> None:
        """Convenience method to add a system prompt."""
        self._components.append(ContextComponent(
            id=f"sys_{len(self._components)}",
            type=ComponentType.SYSTEM_PROMPT,
            content=content,
            token_count=count_tokens(content),
            metadata=metadata,
        ))
    
    def add_user_message(self, content: str, **metadata) -> None:
        """Convenience method to add a user message."""
        self._components.append(ContextComponent(
            id=f"user_{len(self._components)}",
            type=ComponentType.USER_MESSAGE,
            content=content,
            token_count=count_tokens(content),
            metadata=metadata,
        ))
    
    def get_trace(self, context_limit: Optional[int] = None) -> ContextTrace:
        """
        Build and return the complete trace.
        
        Parameters
        ----------
        context_limit : int, optional
            Override the context limit. Defaults to value from constructor.
        
        Returns
        -------
        ContextTrace
            The complete trace including Chroma queries and all components.
        """
        limit = context_limit or self._context_limit
        
        # Build RAG components from selected documents
        rag_components = []
        for query in self._queries:
            for result in query.results:
                if result.selected:
                    rag_components.append(ContextComponent(
                        id=result.id,
                        type=ComponentType.RAG_DOCUMENT,
                        content=result.content,
                        token_count=result.token_count,
                        metadata={
                            **result.metadata,
                            "chroma_score": result.score,
                            "collection": result.collection,
                            "embedding_model": result.embedding_model,
                        },
                    ))
        
        # Combine with other components
        all_components = self._components + rag_components
        total_tokens = sum(c.token_count for c in all_components)
        
        return ContextTrace(
            context_limit=limit,
            components=all_components,
            total_tokens=total_tokens,
            chroma_queries=self._queries,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=str(uuid.uuid4())[:8],
        )
    
    def reset(self) -> None:
        """Clear all traced data."""
        self._queries = []
        self._selected_ids = set()
        self._components = []
    
    # Delegate other collection methods
    def __getattr__(self, name: str):
        """Delegate unknown attributes to underlying collection."""
        return getattr(self._collection, name)
