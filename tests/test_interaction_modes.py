"""Tests for gesture-based interactions in ContextWindow.

Interaction model:
- Hover → Tooltip (component type + token count)
- Click → Modal with full content and metadata
- Click on text in modal → Switch to edit mode with Save button
"""

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
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-modal-overlay" in h
    assert "ced-modal" in h
    assert "ced-modal-header" in h
    assert "ced-modal-body" in h
    assert "ced-modal-close" in h


def test_tooltip_on_hover():
    """Hover shows tooltip with component type and tokens."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "mouseenter" in h
    assert "mouseleave" in h
    assert "ced-tooltip" in h
    assert "TOKENS" in h


def test_click_opens_modal():
    """Single click opens modal with content."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "addEventListener('click'" in h
    assert "showModal(info)" in h


def test_click_to_edit_in_modal():
    """Clicking on text in modal switches to edit mode."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "Click text to edit" in h
    assert "switchToEditMode" in h
    assert "ced-content-text" in h


def test_save_button_in_header():
    """Save button is in header, hidden by default."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-modal-save" in h
    assert "ced-modal-actions" in h
    assert 'style="display:none;">Save</button>' in h


def test_editable_modal_has_textarea():
    """Edit mode shows textarea for content editing."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-modal-textarea" in h
    assert "ced-edit-textarea" in h


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
    assert '\\"quotes\\"' in h
    assert "\\n" in h


def test_modal_close_function():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    uid = ctx._uid
    assert f"cedCloseModal_{uid}" in h


def test_metadata_table_in_modal():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-metadata-table" in h


def test_color_map_in_js():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "colorMap" in h
    assert "#FF6B00" in h


def test_no_mode_buttons():
    """Mode buttons should not exist - simplified UX."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert 'data-mode="view"' not in h
    assert 'data-mode="explore"' not in h
    assert 'data-mode="edit"' not in h
    assert "cedSetMode_" not in h


def test_no_double_click():
    """Double-click handler removed - click text to edit instead."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "dblclick" not in h


def test_header_has_settings_button():
    """Header should have settings gear button."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-header" in h
    assert "\u2699" in h


def test_text_scrollbar_css():
    """Text content should have scrollbar for long content."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "ced-modal-text" in h
    assert "overflow-y: auto" in h
    assert "max-height:" in h
