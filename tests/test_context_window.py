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
            ComponentType.RAG,
            "Document content here.",
            8000,
            metadata={"chroma_score": 0.92, "source": "test.md"},
        ),
        ContextComponent(
            "rag_2",
            ComponentType.RAG,
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


def test_component_data_json():
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert f"cedData_{ctx._uid}" in h
    assert '"system_prompt"' in h
    assert '"rag"' in h


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


def test_vertical_layout_height():
    ctx = ContextWindow(trace=_make_trace(), layout="vertical")
    h = ctx.to_html()
    assert "ced-vertical" in h
    # Components should have height styling
    assert "height:" in h


def test_context_limit_override():
    trace = _make_trace()
    ctx = ContextWindow(trace=trace, context_limit=200000)
    h = ctx.to_html()
    assert "200,000" in h


def test_repr_html_matches_to_html():
    ctx = ContextWindow(trace=_make_trace())
    assert ctx._repr_html_() == ctx.to_html()


# ============================================================================
# ContextBuilder stateful editing tests
# ============================================================================


def test_backward_compat_context_window_alias():
    """ContextWindow should be an alias for ContextBuilder."""
    from context_engineering_dashboard import ContextBuilder, ContextWindow

    assert ContextWindow is ContextBuilder


def test_get_trace_returns_copy():
    """get_trace() should return a deep copy of the working trace."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    builder = ContextBuilder(trace=trace)
    returned_trace = builder.get_trace()

    # Should be equal in content
    assert returned_trace.total_tokens == trace.total_tokens
    assert len(returned_trace.components) == len(trace.components)

    # But not the same object
    assert returned_trace is not builder._working_trace
    assert returned_trace.components is not builder._working_trace.components


def test_apply_edit_updates_content():
    """apply_edit() should update the component content."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    builder = ContextBuilder(trace=trace)

    new_content = "This is new system prompt content."
    builder.apply_edit("sys_1", new_content)

    # Check the working trace was updated
    comp = next(c for c in builder._working_trace.components if c.id == "sys_1")
    assert comp.content == new_content


def test_apply_edit_recounts_tokens():
    """apply_edit() should recount tokens for the edited content."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    original_total = trace.total_tokens
    builder = ContextBuilder(trace=trace)

    # Short content should have fewer tokens
    new_content = "Short."
    builder.apply_edit("sys_1", new_content)

    comp = next(c for c in builder._working_trace.components if c.id == "sys_1")

    # Token count should be updated (not the original 2000)
    assert comp.token_count != 2000
    assert comp.token_count < 100  # Short content

    # Total tokens should be adjusted
    assert builder._working_trace.total_tokens != original_total


def test_apply_edit_raises_on_missing_id():
    """apply_edit() should raise KeyError for unknown component ID."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    builder = ContextBuilder(trace=trace)

    try:
        builder.apply_edit("nonexistent_id", "content")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "nonexistent_id" in str(e)


def test_apply_reorder_changes_order():
    """apply_reorder() should reorder components."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    builder = ContextBuilder(trace=trace)

    # Original order: sys_1, rag_1, rag_2, user_1
    original_ids = [c.id for c in builder._working_trace.components]
    assert original_ids == ["sys_1", "rag_1", "rag_2", "user_1"]

    # Reorder: user_1 first
    new_order = ["user_1", "sys_1", "rag_1", "rag_2"]
    builder.apply_reorder(new_order)

    reordered_ids = [c.id for c in builder._working_trace.components]
    assert reordered_ids == new_order


def test_has_changes_after_edit():
    """has_changes() should return True after editing."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    builder = ContextBuilder(trace=trace)

    assert not builder.has_changes()

    builder.apply_edit("sys_1", "New content")

    assert builder.has_changes()


def test_has_changes_after_reorder():
    """has_changes() should return True after reordering."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    builder = ContextBuilder(trace=trace)

    assert not builder.has_changes()

    builder.apply_reorder(["user_1", "sys_1", "rag_1", "rag_2"])

    assert builder.has_changes()


def test_reset_clears_edits():
    """reset() should restore the original trace."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    original_content = trace.components[0].content
    builder = ContextBuilder(trace=trace)

    # Make some edits
    builder.apply_edit("sys_1", "Modified content")
    builder.apply_reorder(["user_1", "sys_1", "rag_1", "rag_2"])

    assert builder.has_changes()

    # Reset
    builder.reset()

    assert not builder.has_changes()

    # Content should be restored
    comp = next(c for c in builder._working_trace.components if c.id == "sys_1")
    assert comp.content == original_content

    # Order should be restored
    ids = [c.id for c in builder._working_trace.components]
    assert ids == ["sys_1", "rag_1", "rag_2", "user_1"]


def test_trace_property_returns_working_trace():
    """trace property should return the working trace (backward compat)."""
    from context_engineering_dashboard import ContextBuilder

    trace = _make_trace()
    builder = ContextBuilder(trace=trace)

    assert builder.trace is builder._working_trace


def test_html_contains_context_builder_label():
    """HTML should contain 'Context Builder' label."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "Context Builder" in h


def test_state_retrieval_function_present():
    """JavaScript should include cedGetState function."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    uid = ctx._uid
    assert f"cedGetState_{uid}" in h


def test_save_stores_edit_in_data_attribute():
    """Save handler should store edits in data-edits attribute."""
    ctx = ContextWindow(trace=_make_trace())
    h = ctx.to_html()
    assert "data-edits" in h
    assert "data-has-changes" in h
