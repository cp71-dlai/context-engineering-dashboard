# Context Engineering Dashboard — Specification Document

**Version**: 1.0.0  
**Status**: Draft  
**Target Platform**: Python 3.10+, Jupyter Notebook  
**Design Style**: Neo-Brutalist  

---

## 1. Overview

`context-engineering-dashboard` is a Python library for visualizing and interacting with LLM context windows in Jupyter notebooks. It provides educators and practitioners a tool to understand, debug, and optimize context engineering decisions in RAG and agentic workflows.

### 1.1 Core Value Proposition

- **See** what fills your context window and how much space each component uses
- **Understand** Chroma retrieval decisions: what was fetched vs. what was selected
- **Compare** context states before/after compaction or filtering
- **Edit** context components and re-run prompts interactively

### 1.2 Design Philosophy

Modeled after `aisuite`: lightweight, modular provider architecture, minimal dependencies, OpenAI-style API familiarity.

---

## 2. Architecture

```
context_engineering_dashboard/
├── __init__.py
├── client.py                    # Main entry point
├── core/
│   ├── __init__.py
│   ├── context_window.py        # ContextWindow class
│   ├── context_diff.py          # ContextDiff + Sankey
│   ├── trace.py                 # Trace dataclasses
│   └── renderer.py              # HTML/JS generation
├── tracers/
│   ├── __init__.py
│   ├── base_tracer.py           # Abstract base
│   ├── openai_tracer.py         # OpenAI tracing
│   ├── langchain_tracer.py      # LangChain tracing
│   └── chroma_tracer.py         # Chroma-specific tracing
├── providers/
│   ├── __init__.py
│   ├── openai_provider.py       # Re-run support for OpenAI
│   └── langchain_provider.py    # Re-run support for LangChain
├── styles/
│   ├── __init__.py
│   ├── colors.py                # Color palette definitions
│   ├── patterns.py              # SVG pattern definitions
│   └── neo_brutalist.css        # Base stylesheet
├── layouts/
│   ├── __init__.py
│   ├── horizontal.py            # Left-to-right packing
│   └── treemap.py               # Treemap layout algorithm
└── utils/
    ├── __init__.py
    ├── tokenizer.py             # Token counting utilities
    └── serialization.py         # JSON export/import
```

---

## 3. Data Model

### 3.1 Trace Schema

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class ComponentType(Enum):
    SYSTEM_PROMPT = "system_prompt"
    USER_MESSAGE = "user_message"
    CHAT_HISTORY = "chat_history"
    RAG_DOCUMENT = "rag_document"
    TOOL = "tool"
    FEW_SHOT = "few_shot"
    MEMORY = "memory"

@dataclass
class ContextComponent:
    """A single component in the context window."""
    id: str
    type: ComponentType
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChromaRetrievalResult:
    """A document returned from Chroma query."""
    id: str
    content: str
    token_count: int
    score: float                          # Similarity score
    selected: bool                        # Made it into context window?
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection: str = ""
    embedding_model: str = ""

@dataclass
class ChromaQuery:
    """Captures a Chroma retrieval operation."""
    collection: str
    query_text: str
    query_embedding: Optional[List[float]] = None
    n_results: int = 10
    where_filter: Optional[Dict] = None
    results: List[ChromaRetrievalResult] = field(default_factory=list)

@dataclass
class ToolCall:
    """A tool invocation."""
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None

@dataclass
class LLMTrace:
    """Trace of a language model call."""
    provider: str                         # "openai", "anthropic", etc.
    model: str
    messages: List[Dict[str, str]]        # Full message list sent
    response: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens
    latency_ms: float = 0.0

@dataclass
class EmbeddingTrace:
    """Trace of an embedding model call."""
    provider: str
    model: str
    input_text: str
    embedding: List[float]
    latency_ms: float = 0.0

@dataclass
class ContextTrace:
    """Complete trace of a context engineering operation."""
    
    # Context window contents
    context_limit: int                    # Max tokens (e.g., 128000)
    components: List[ContextComponent]    # What's in the window
    total_tokens: int                     # Sum of component tokens
    
    # Chroma-specific (optional)
    chroma_queries: List[ChromaQuery] = field(default_factory=list)
    
    # Model traces
    llm_trace: Optional[LLMTrace] = None
    embedding_traces: List[EmbeddingTrace] = field(default_factory=list)
    
    # Metadata
    timestamp: str = ""
    session_id: str = ""
    tags: List[str] = field(default_factory=list)
```

### 3.2 JSON Schema (for serialization)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ContextTrace",
  "type": "object",
  "required": ["context_limit", "components", "total_tokens"],
  "properties": {
    "context_limit": { "type": "integer" },
    "total_tokens": { "type": "integer" },
    "components": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "type", "content", "token_count"],
        "properties": {
          "id": { "type": "string" },
          "type": { 
            "type": "string",
            "enum": ["system_prompt", "user_message", "chat_history", 
                     "rag_document", "tool", "few_shot", "memory"]
          },
          "content": { "type": "string" },
          "token_count": { "type": "integer" },
          "metadata": { "type": "object" }
        }
      }
    },
    "chroma_queries": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "collection": { "type": "string" },
          "query_text": { "type": "string" },
          "n_results": { "type": "integer" },
          "results": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["id", "content", "token_count", "score", "selected"],
              "properties": {
                "id": { "type": "string" },
                "content": { "type": "string" },
                "token_count": { "type": "integer" },
                "score": { "type": "number" },
                "selected": { "type": "boolean" },
                "metadata": { "type": "object" }
              }
            }
          }
        }
      }
    },
    "llm_trace": {
      "type": "object",
      "properties": {
        "provider": { "type": "string" },
        "model": { "type": "string" },
        "messages": { "type": "array" },
        "response": { "type": "string" },
        "usage": { "type": "object" }
      }
    }
  }
}
```

---

## 4. Visual Design

### 4.1 Neo-Brutalist Style Guide

| Property | Value |
|----------|-------|
| Borders | 3px solid black |
| Border radius | 0 (sharp corners) |
| Shadows | None |
| Gradients | None |
| Typography | `JetBrains Mono`, `IBM Plex Mono`, monospace fallback |
| Labels | ALL CAPS, 12px, bold |
| Background | White (`#FFFFFF`) |
| Grid lines | 1px dashed `#CCCCCC` |

### 4.2 Color Palette (WCAG AA Compliant)

| Component | Hex | RGB | Text Color |
|-----------|-----|-----|------------|
| System Prompt | `#FF6B00` | 255, 107, 0 | Black |
| User Message | `#0066FF` | 0, 102, 255 | White |
| Chat History | `#00BFFF` | 0, 191, 255 | Black |
| RAG Document | `#00AA55` | 0, 170, 85 | White |
| Tool | `#FFCC00` | 255, 204, 0 | Black |
| Few-Shot | `#AA44FF` | 170, 68, 255 | White |
| Memory | `#00CCAA` | 0, 204, 170 | Black |
| Unused | `#E0E0E0` | 224, 224, 224 | Black (dashed border) |
| Unselected (Chroma) | `#F5F5F5` | 245, 245, 245 | Grey text |

### 4.3 Accessibility Patterns

Optional pattern fills for colorblind users:

| Component | Pattern |
|-----------|---------|
| System Prompt | Solid fill |
| User Message | Horizontal stripes |
| Chat History | Diagonal stripes (/) |
| RAG Document | Dots |
| Tool | Crosshatch |
| Few-Shot | Vertical stripes |
| Memory | Diagonal stripes (\\) |

### 4.4 Icons

Simple geometric shapes, rendered as inline SVG:

| Component | Icon |
|-----------|------|
| System Prompt | Gear (⚙) |
| User Message | Speech bubble |
| Chat History | Clock |
| RAG Document | Document with folded corner |
| Tool | Wrench |
| Few-Shot | List with checkmark |
| Memory | Brain outline |

---

## 5. Layout Algorithms

### 5.1 Horizontal Layout (Default)

Components packed left-to-right. Height is fixed. Width proportional to token count.

```
┌──────────────────────────────────────────────────────────────┐
│ SYS │    RAG DOC 1    │ RAG 2 │ USER │░░░░░░ UNUSED ░░░░░░░│
│ 2K  │       8K        │  3K   │  1K  │░░░░░░  114K  ░░░░░░░│
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Treemap Layout

Squarified treemap algorithm. Better visibility for small components.

```
┌─────────────────────┬────────────────────────────────────────┐
│                     │  RAG DOC 1 (8K)                        │
│  UNUSED (114K)      ├───────────────┬────────────────────────┤
│                     │ RAG 2 (3K)    │  SYS (2K)  │ USER (1K) │
└─────────────────────┴───────────────┴────────────┴───────────┘
```

### 5.3 Layout Toggle

Setting in UI allows switching between layouts without re-rendering data.

---

## 6. Interaction Modes

### 6.1 View Mode

- **Hover**: Tooltip shows component name + token count
- **Chroma docs**: Score badge visible on component (`0.92`)
- **Click**: No action
- **Double-click**: No action

### 6.2 Explore Mode

- **Hover**: Tooltip shows component name + token count
- **Chroma docs**: Score badge visible
- **Click**: Highlights component, shows summary in sidebar
- **Double-click**: Opens modal with:
  - Full verbatim text (scrollable, max-height 400px)
  - Token count
  - For Chroma docs: score, metadata table, collection name

### 6.3 Edit Mode

- **Hover**: Tooltip + edit cursor
- **Click**: Selects component for editing
- **Double-click**: Opens inline editor (textarea)
  - Save button commits change
  - Cancel reverts
  - Re-run button triggers callback with modified trace
- **Drag**: Reorder components (future enhancement)

---

## 7. Sankey Diff View

### 7.1 Purpose

Visualize context transformations: compaction, filtering, summarization.

### 7.2 Layout

```
   BEFORE (40K total)              AFTER (25K total)
┌──────────────────────┐        ┌────────────────┐
│ System (4K)          │────────│ System (4K)    │
├──────────────────────┤   ╲    ├────────────────┤
│ History (20K)        │────────│ History (8K)   │  ← 60% reduction
├──────────────────────┤    ╲   ├────────────────┤
│ RAG (15K)            │─────╲──│ RAG (10K)      │  ← 33% reduction
├──────────────────────┤      ╲ ├────────────────┤
│ Tools (1K)           │───────X│ [REMOVED]      │  ← cut entirely
└──────────────────────┘        ├────────────────┤
                                │ UNUSED (103K)  │
                                └────────────────┘
```

### 7.3 Visual Rules

- Flow width = token count
- Removed components flow to a "waste" node (grey)
- Compression shown by narrowing flow
- Labels show before/after token counts
- Percentage change badges on flows

---

## 8. Chroma Integration

### 8.1 Available Pool View

Left panel shows all documents returned by Chroma query. Right panel shows curated context window. Scissors icon between them represents curation.

```
┌─ AVAILABLE (Chroma) ─────────┐  ✂️  ┌─ CONTEXT WINDOW ──────────┐
│ ┌─────────────────────────┐  │      │ ┌─────────────────────────┐│
│ │ doc_14  0.92  340 tok ✓ │──┼──────┼─│ doc_14  340 tok         ││
│ ├─────────────────────────┤  │      │ ├─────────────────────────┤│
│ │ doc_7   0.87  520 tok ✓ │──┼──────┼─│ doc_7   520 tok         ││
│ ├─────────────────────────┤  │      │ ├─────────────────────────┤│
│ │ doc_22  0.85  290 tok ✓ │──┼──────┼─│ doc_22  290 tok         ││
│ ├─────────────────────────┤  │      │ └─────────────────────────┘│
│ │ doc_3   0.81 1200 tok ✗ │  │      │                            │
│ ├─────────────────────────┤  │      │ ┌─────────────────────────┐│
│ │ doc_9   0.79  890 tok ✗ │  │      │ │ System Prompt  2K tok   ││
│ └─────────────────────────┘  │      │ └─────────────────────────┘│
└──────────────────────────────┘      └────────────────────────────┘
```

### 8.2 Chroma-Specific Display

**VIEW Mode:**
- Score badge on each retrieved doc
- Selected/unselected visual state

**EXPLORE Mode (double-click):**
- Full document text
- Similarity score
- Metadata table (key-value pairs)
- Collection name
- Embedding model used

---

## 9. API Reference

### 9.1 Tracing

```python
from context_engineering_dashboard import trace_openai, trace_langchain, trace_chroma

# OpenAI tracing
with trace_openai() as tracer:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[...]
    )
trace = tracer.result

# LangChain tracing
with trace_langchain() as tracer:
    result = chain.invoke({"query": "..."})
trace = tracer.result

# Chroma tracing (wraps collection)
traced_collection = trace_chroma(collection)
results = traced_collection.query(query_texts=["..."], n_results=10)
chroma_trace = traced_collection.get_trace()
```

### 9.2 Visualization

```python
from context_engineering_dashboard import ContextWindow, ContextDiff

# Basic view
ctx = ContextWindow(trace=trace, context_limit=128_000)
ctx.display()

# With options
ctx = ContextWindow(
    trace=trace,
    context_limit=128_000,
    mode="explore",           # "view" | "explore" | "edit"
    layout="horizontal",      # "horizontal" | "treemap"
    show_patterns=False,      # Accessibility patterns
    show_available_pool=True, # Show Chroma retrieval pool
    theme="neo_brutalist"     # Only option for now
)
ctx.display()

# Edit mode with callback
def on_rerun(modified_trace):
    # Re-execute with modified context
    pass

ctx = ContextWindow(trace=trace, mode="edit")
ctx.on_rerun(callback=on_rerun)
ctx.display()

# Diff view
diff = ContextDiff(before=trace1, after=trace2)
diff.sankey()

# With labels
diff = ContextDiff(
    before=trace1,
    after=trace2,
    before_label="Original",
    after_label="After /compact"
)
diff.sankey()
```

### 9.3 Serialization

```python
from context_engineering_dashboard import ContextTrace

# Export
trace.to_json("trace.json")
trace.to_dict()

# Import
trace = ContextTrace.from_json("trace.json")
trace = ContextTrace.from_dict(data)
```

---

## 10. Settings Panel

Accessible via gear icon in visualization header.

| Setting | Options | Default |
|---------|---------|---------|
| Layout | Horizontal, Treemap | Horizontal |
| Mode | View, Explore, Edit | View |
| Show Patterns | On, Off | Off |
| Show Available Pool | On, Off | On |
| Token Display | Absolute, Percentage | Absolute |
| Theme | Neo-Brutalist | Neo-Brutalist |

---

## 11. Dependencies

**Required:**
- `ipython` >= 8.0
- `tiktoken` >= 0.5 (token counting)

**Optional:**
- `openai` (for OpenAI tracing)
- `langchain` (for LangChain tracing)
- `chromadb` (for Chroma tracing)

Installation:

```bash
pip install context-engineering-dashboard

# With providers
pip install 'context-engineering-dashboard[openai]'
pip install 'context-engineering-dashboard[langchain]'
pip install 'context-engineering-dashboard[chroma]'
pip install 'context-engineering-dashboard[all]'
```

---

## 12. Implementation Phases

### Phase 1: Core (MVP)
- [ ] Trace dataclasses
- [ ] ContextWindow with VIEW mode
- [ ] Horizontal layout
- [ ] Neo-brutalist CSS
- [ ] OpenAI tracer
- [ ] `_repr_html_` rendering

### Phase 2: Interactivity
- [ ] EXPLORE mode (double-click modal)
- [ ] Treemap layout
- [ ] Settings panel
- [ ] Chroma tracer with scores/metadata

### Phase 3: Advanced
- [ ] EDIT mode
- [ ] Re-run callback
- [ ] ContextDiff Sankey
- [ ] LangChain tracer
- [ ] Accessibility patterns

### Phase 4: Polish
- [ ] JSON export/import
- [ ] Available pool view
- [ ] Animation for Sankey flows
- [ ] Documentation site

---

## 13. Testing Strategy

- **Unit tests**: Trace parsing, token counting, layout algorithms
- **Visual regression**: Snapshot testing of rendered HTML
- **Integration tests**: Tracer accuracy with real API calls (mocked)
- **Accessibility audit**: WCAG AA compliance check

---

## 14. Open Questions

1. Should we support streaming responses in the tracer?
2. Do we need a CLI for generating traces outside notebooks?
3. Should the Sankey view support >2 states (e.g., 3-way diff)?

---

## Appendix A: Sample Trace JSON

```json
{
  "context_limit": 128000,
  "total_tokens": 14350,
  "components": [
    {
      "id": "sys_001",
      "type": "system_prompt",
      "content": "You are a helpful assistant...",
      "token_count": 2000,
      "metadata": {}
    },
    {
      "id": "rag_001",
      "type": "rag_document",
      "content": "ChromaDB documentation: Getting started...",
      "token_count": 8000,
      "metadata": {
        "source": "docs/getting-started.md",
        "chroma_score": 0.92
      }
    },
    {
      "id": "user_001",
      "type": "user_message",
      "content": "How do I create a collection in Chroma?",
      "token_count": 15,
      "metadata": {}
    }
  ],
  "chroma_queries": [
    {
      "collection": "documentation",
      "query_text": "How do I create a collection in Chroma?",
      "n_results": 10,
      "results": [
        {
          "id": "doc_14",
          "content": "ChromaDB documentation: Getting started...",
          "token_count": 8000,
          "score": 0.92,
          "selected": true,
          "metadata": {"source": "docs/getting-started.md"}
        },
        {
          "id": "doc_7",
          "content": "Advanced collection configuration...",
          "token_count": 5200,
          "score": 0.87,
          "selected": false,
          "metadata": {"source": "docs/advanced.md"}
        }
      ]
    }
  ],
  "llm_trace": {
    "provider": "openai",
    "model": "gpt-4o",
    "messages": [...],
    "response": "To create a collection in Chroma...",
    "usage": {
      "prompt_tokens": 14350,
      "completion_tokens": 256
    }
  },
  "timestamp": "2025-01-26T10:30:00Z",
  "session_id": "sess_abc123"
}
```
