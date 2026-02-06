# Claude Code Build Instructions

## Project: context-engineering-dashboard

A Python library for visualizing LLM context windows in Jupyter notebooks. Built in partnership with Chroma for the DeepLearning.AI Context Engineering course.

---

## Build Order

Execute phases sequentially. Each phase depends on the previous.

### Phase 1: Core Data Structures ✅

Files to verify/complete:
- `context_engineering_dashboard/core/trace.py` — Dataclasses for traces
- `schemas/trace-schema.json` — JSON Schema validation

**Tests to write:**
```bash
tests/
├── test_trace.py           # Serialize/deserialize traces
├── test_trace_schema.py    # Validate against JSON schema
```

**Acceptance criteria:**
- [ ] `ContextTrace.to_json()` produces valid JSON
- [ ] `ContextTrace.from_json()` round-trips correctly
- [ ] All component types serialize with correct enum values

---

### Phase 2: Basic Visualization

Files to complete:
- `context_engineering_dashboard/core/context_window.py`
- `context_engineering_dashboard/styles/colors.py`

**Key requirements:**
1. `ContextWindow._repr_html_()` returns valid HTML for Jupyter
2. Horizontal layout: components as flex items, width ∝ token_count
3. Neo-brutalist styling:
   - 3px solid black borders
   - No border-radius, shadows, or gradients
   - Monospace font (JetBrains Mono fallback chain)
   - ALL CAPS labels
4. Unused space shown with dashed grey border

**Tests:**
```python
def test_context_window_renders_html():
    trace = ContextTrace(...)
    ctx = ContextWindow(trace=trace, context_limit=128000)
    html = ctx.to_html()
    assert '<div class="ced-container"' in html
    assert 'ced-component' in html
```

---

### Phase 3: Chroma Integration (Priority)

Files:
- `context_engineering_dashboard/tracers/chroma_tracer.py`

**Chroma-specific features:**

1. **VIEW mode**: Score badge visible on RAG docs
   ```html
   <span class="ced-score-badge">0.92</span>
   ```

2. **EXPLORE mode** (double-click modal):
   - Full document text
   - Similarity score
   - Metadata table (all key-value pairs)
   - Collection name
   - Embedding model name

3. **Available Pool**: Show all retrieved docs, highlight selected
   ```
   ┌─ AVAILABLE ─────────┐  ✂️  ┌─ CONTEXT ──────────┐
   │ doc_1  0.92  ✓      │──────│ doc_1              │
   │ doc_2  0.87  ✓      │──────│ doc_2              │
   │ doc_3  0.81  ✗      │      │                    │
   └─────────────────────┘      └────────────────────┘
   ```

**API:**
```python
from context_engineering_dashboard import trace_chroma

traced = trace_chroma(collection)
results = traced.query(query_texts=["How do I..."], n_results=10)

# Mark which docs made the cut
traced.mark_selected(["doc_1", "doc_2"])

# Add other context
traced.add_system_prompt("You are a helpful assistant...")
traced.add_user_message("How do I create a collection?")

# Get trace
trace = traced.get_trace()
```

**Tests:**
```python
def test_chroma_tracer_captures_scores():
    # Mock Chroma collection
    trace = traced.get_trace()
    rag_docs = trace.get_components_by_type(ComponentType.RAG_DOCUMENT)
    assert all("chroma_score" in doc.metadata for doc in rag_docs)

def test_chroma_tracer_tracks_selection():
    traced.mark_selected(["doc_1"])
    trace = traced.get_trace()
    assert trace.chroma_queries[0].results[0].selected == True
    assert trace.chroma_queries[0].results[1].selected == False
```

---

### Phase 4: Interaction Modes

**VIEW mode** (default):
- Hover: tooltip with component name + token count
- Click: no action
- Double-click: no action

**EXPLORE mode**:
- Hover: tooltip
- Click: highlight component
- Double-click: open modal with full content + metadata

**EDIT mode** (future):
- Double-click: inline textarea editor
- Save/Cancel buttons
- Re-run callback: `ctx.on_rerun(callback)`

**Implementation:**
```javascript
// Mode switching
const mode = '{{mode}}';  // Injected from Python

comp.addEventListener('dblclick', () => {
    if (mode === 'view') return;
    if (mode === 'explore') openModal(comp);
    if (mode === 'edit') openEditor(comp);
});
```

---

### Phase 5: Sankey Diff View

File: `context_engineering_dashboard/core/context_diff.py`

**Requirements:**
1. SVG-based Sankey diagram
2. Before column (left) → After column (right)
3. Flow width = token count
4. Compression shown by narrowing paths
5. Removed components flow to "waste" node
6. Percentage badges on flows showing change

**API:**
```python
from context_engineering_dashboard import ContextDiff

diff = ContextDiff(before=trace1, after=trace2)
diff.sankey()  # Render in notebook
diff.summary() # Print text summary
```

---

### Phase 6: Layout Toggle

Add treemap layout option using squarify algorithm.

```python
ctx = ContextWindow(trace=trace, layout="treemap")
```

**Toggle in settings panel:**
```html
<button class="ced-btn" data-layout="toggle">Treemap</button>
```

---

### Phase 7: OpenAI Tracer

File: `context_engineering_dashboard/tracers/openai_tracer.py`

**Requirements:**
1. Monkey-patch `openai.chat.completions.create`
2. Capture messages, response, usage
3. Auto-detect model context limit
4. Calculate token counts with tiktoken

**API:**
```python
from context_engineering_dashboard import trace_openai

with trace_openai() as tracer:
    response = client.chat.completions.create(...)

trace = tracer.result
```

---

### Phase 8: LangChain Tracer

File: `context_engineering_dashboard/tracers/langchain_tracer.py`

**Requirements:**
1. Implement `BaseCallbackHandler`
2. Capture `on_llm_start`, `on_llm_end`
3. Capture `on_retriever_start`, `on_retriever_end`
4. Handle both `invoke()` and `stream()` patterns

---

## File Structure

```
context-engineering-dashboard/
├── pyproject.toml
├── README.md
├── SPECIFICATION.md
├── CLAUDE_CODE_INSTRUCTIONS.md
├── schemas/
│   └── trace-schema.json
├── mockups/
│   └── neo-brutalist-mockup.html
├── context_engineering_dashboard/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── trace.py
│   │   ├── context_window.py
│   │   └── context_diff.py
│   ├── tracers/
│   │   ├── __init__.py
│   │   ├── base_tracer.py
│   │   ├── openai_tracer.py
│   │   ├── langchain_tracer.py
│   │   └── chroma_tracer.py
│   ├── styles/
│   │   ├── __init__.py
│   │   └── colors.py
│   └── layouts/
│       ├── __init__.py
│       ├── horizontal.py
│       └── treemap.py
├── tests/
│   ├── __init__.py
│   ├── test_trace.py
│   ├── test_context_window.py
│   ├── test_context_diff.py
│   ├── test_openai_tracer.py
│   ├── test_langchain_tracer.py
│   └── test_chroma_tracer.py
└── examples/
    ├── basic_usage.ipynb
    ├── chroma_rag.ipynb
    └── diff_view.ipynb
```

---

## Style Constants

### Colors (WCAG AA Compliant)

```python
COMPONENT_COLORS = {
    "system_prompt": "#FF6B00",   # Orange
    "user_message": "#0066FF",    # Blue
    "chat_history": "#00BFFF",    # Light blue
    "rag_document": "#00AA55",    # Green
    "tool": "#FFCC00",            # Yellow
    "few_shot": "#AA44FF",        # Purple
    "memory": "#00CCAA",          # Teal
    "unused": "#E0E0E0",          # Grey
}
```

### CSS Variables

```css
:root {
    --border-width: 3px;
    --color-black: #000000;
    --color-white: #FFFFFF;
    --font-mono: 'JetBrains Mono', 'IBM Plex Mono', 'Consolas', monospace;
}
```

---

## Testing Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=context_engineering_dashboard --cov-report=html

# Type checking
mypy context_engineering_dashboard/

# Linting
ruff check context_engineering_dashboard/

# Format
black context_engineering_dashboard/
```

---

## Dependencies

**Core (required):**
- ipython >= 8.0
- tiktoken >= 0.5.0

**Optional:**
- openai >= 1.0.0 (for OpenAI tracer)
- chromadb >= 0.4.0 (for Chroma tracer)
- langchain-core >= 0.1.0 (for LangChain tracer)

**Dev:**
- pytest >= 7.0
- pytest-cov >= 4.0
- black >= 23.0
- ruff >= 0.1.0
- mypy >= 1.0

---

## Key Design Decisions

1. **HTML rendering**: Use `_repr_html_()` for automatic Jupyter display
2. **No external JS libs**: Pure vanilla JS for interactivity
3. **CSS-in-HTML**: Styles embedded in rendered output (no external CSS files)
4. **Provider pattern**: Follow `aisuite` model — `<provider>_tracer.py` naming
5. **Dataclass serialization**: Custom `to_dict()`/`from_dict()` methods
6. **Token counting**: Use tiktoken with graceful fallback (chars/4)

---

## Common Pitfalls

1. **Don't use `localStorage`** — Not available in Jupyter iframe
2. **Escape HTML content** — User content may contain `<script>` tags
3. **Handle missing dependencies** — Tracers should fail gracefully if SDK not installed
4. **Test in JupyterLab and classic Notebook** — CSS may render differently

---

## Definition of Done

A feature is complete when:
- [ ] Code passes all tests
- [ ] Code passes type checking (mypy)
- [ ] Code passes linting (ruff)
- [ ] Example notebook demonstrates usage
- [ ] Renders correctly in Jupyter Lab
- [ ] Renders correctly in classic Notebook
- [ ] No console errors in browser
