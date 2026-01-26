"""Tests for interaction mode JS and HTML in ContextWindow."""

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
            metadata={"source": "config"},
        ),
        ContextComponent(
            "rag_1",
            ComponentType.RAG_DOCUMENT,
            "Document content here.",
            8000,
            metadata={"chroma_score": 0.92, "collection": "docs"},
        ),
    ]
    return ContextTrace(context_limit=128000, components=components, total_tokens=10000)


def test_modal_overlay_present():
    ctx = ContextWindow(trace=_make_trace(), mode="explore")
    h = ctx.to_html()
    assert "ced-modal-overlay" in h
    assert "ced-modal" in h
    assert "ced-modal-header" in h
    assert "ced-modal-body" in h
    assert "ced-modal-close" in h


def test_tooltip_js_present():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "mouseenter" in h
    assert "mouseleave" in h
    assert "ced-tooltip" in h


def test_dblclick_handlers():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "dblclick" in h


def test_mode_buttons_present():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert 'data-mode="view"' in h
    assert 'data-mode="explore"' in h
    assert 'data-mode="edit"' in h
    assert "cedSetMode_" in h


def test_component_data_js_object():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    uid = ctx._uid
    assert f"cedData_{uid}" in h
    assert '"sys_1"' in h
    assert '"rag_1"' in h
    assert '"system_prompt"' in h
    assert '"rag_document"' in h


def test_json_escaping_in_component_data():
    """Ensure special chars in content are properly JSON-escaped."""
    components = [
        ContextComponent(
            "test_1",
            ComponentType.SYSTEM_PROMPT,
            'Content with "quotes" and\nnewlines and <tags>',
            10,
        ),
    ]
    trace = ContextTrace(context_limit=1000, components=components, total_tokens=10)
    ctx = ContextWindow(trace=trace)
    h = ctx.to_html()
    # JSON should escape quotes and newlines
    assert '\\"quotes\\"' in h
    assert "\\n" in h


def test_explore_mode_active_button():
    ctx = ContextWindow(trace=_make_trace(), mode="explore")
    h = ctx.to_html()
    # The explore button should be active
    assert 'data-mode="explore"' in h
    # Check that explore button has ced-active class near it
    idx = h.find('data-mode="explore"')
    assert idx > 0
    preceding = h[max(0, idx - 80) : idx]
    assert "ced-active" in preceding


def test_edit_mode_textarea_in_js():
    ctx = ContextWindow(trace=_make_trace(), mode="edit")
    h = ctx.to_html()
    assert "textarea" in h
    assert "Save" in h
    assert "Cancel" in h


def test_highlight_class_in_js():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-highlighted" in h


def test_modal_close_function():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    uid = ctx._uid
    assert f"cedCloseModal_{uid}" in h


def test_metadata_table_in_js():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-metadata-table" in h


def test_color_map_in_js():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "colorMap" in h
    assert "#FF6B00" in h
