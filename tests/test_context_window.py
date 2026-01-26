"""Tests for ContextWindow HTML rendering."""

from context_engineering_dashboard.core.context_window import ContextWindow
from context_engineering_dashboard.core.trace import (
    ComponentType,
    ContextComponent,
    ContextTrace,
)


def _make_trace():
    components = [
        ContextComponent(
            "sys_1",
            ComponentType.SYSTEM_PROMPT,
            "You are a helpful assistant.",
            2000,
        ),
        ContextComponent(
            "rag_1",
            ComponentType.RAG_DOCUMENT,
            "Document content here.",
            8000,
            metadata={"chroma_score": 0.92, "source": "test.md"},
        ),
        ContextComponent(
            "rag_2",
            ComponentType.RAG_DOCUMENT,
            "Another doc.",
            3000,
            metadata={"chroma_score": 0.87},
        ),
        ContextComponent(
            "user_1",
            ComponentType.USER_MESSAGE,
            "How do I use Chroma?",
            350,
        ),
    ]
    return ContextTrace(
        context_limit=128000,
        components=components,
        total_tokens=13350,
    )


def test_html_contains_container():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-container" in h


def test_html_contains_all_component_ids():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert 'data-comp-id="sys_1"' in h
    assert 'data-comp-id="rag_1"' in h
    assert 'data-comp-id="rag_2"' in h
    assert 'data-comp-id="user_1"' in h


def test_html_token_counter_format():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "13,350 / 128,000 TOKENS (10%)" in h


def test_html_unused_div():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-comp-unused" in h
    assert 'data-comp-id="_unused"' in h


def test_html_score_badges():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "0.92" in h
    assert "0.87" in h
    assert "ced-score-badge" in h


def test_html_neo_brutalist_css():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "3px solid black" in h
    assert "monospace" in h
    assert "uppercase" in h


def test_html_xss_escaping():
    components = [
        ContextComponent(
            "xss_1",
            ComponentType.SYSTEM_PROMPT,
            '<script>alert("xss")</script>',
            10,
        ),
    ]
    trace = ContextTrace(context_limit=1000, components=components, total_tokens=10)
    ctx = ContextWindow(trace=trace)
    h = ctx.to_html()
    # The content is in the JSON data, so it should be JSON-escaped
    assert '<script>alert("xss")</script>' not in h
    # But should be present in escaped form
    assert "xss" in h


def test_unique_instance_ids():
    trace = _make_trace()
    c1 = ContextWindow(trace=trace)
    c2 = ContextWindow(trace=trace)
    h1 = c1.to_html()
    h2 = c2.to_html()
    assert c1._uid != c2._uid
    assert c1._uid in h1
    assert c2._uid in h2


def test_mode_buttons():
    ctx = ContextWindow(trace=_make_trace(), mode="explore")
    h = ctx.to_html()
    assert 'data-mode="view"' in h
    assert 'data-mode="explore"' in h
    assert 'data-mode="edit"' in h


def test_component_data_json():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert f"cedData_{ctx._uid}" in h
    assert '"system_prompt"' in h
    assert '"rag_document"' in h


def test_legend_present():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-legend" in h
    assert "#FF6B00" in h  # System prompt color
    assert "#0066FF" in h  # User message color
    assert "#00AA55" in h  # RAG document color


def test_tooltip_div():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-tooltip" in h


def test_modal_overlay():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-modal-overlay" in h
    assert "ced-modal-header" in h


def test_horizontal_layout_flex():
    ctx = ContextWindow(trace=_make_trace(), layout="horizontal")
    h = ctx.to_html()
    assert "ced-horizontal" in h
    # System prompt: 2000 tokens -> flex: 2.0
    assert "flex:2.0" in h or "flex:2" in h


def test_context_limit_override():
    trace = _make_trace()
    ctx = ContextWindow(trace=trace, context_limit=200000)
    h = ctx.to_html()
    assert "200,000" in h


def test_repr_html_matches_to_html():
    ctx = ContextWindow(trace=_make_trace())
    assert ctx._repr_html_() == ctx.to_html()
