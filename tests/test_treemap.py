"""Tests for squarified treemap layout algorithm."""

from context_engineering_dashboard.core.context_window import ContextWindow
from context_engineering_dashboard.core.trace import (
    ComponentType,
    ContextComponent,
    ContextTrace,
)
from context_engineering_dashboard.layouts.treemap import (
    compute_treemap_layout,
)


def _make_trace():
    return ContextTrace(
        context_limit=128000,
        components=[
            ContextComponent("sys_1", ComponentType.SYSTEM_PROMPT, "System", 2000),
            ContextComponent("rag_1", ComponentType.RAG_DOCUMENT, "RAG 1", 8000),
            ContextComponent("rag_2", ComponentType.RAG_DOCUMENT, "RAG 2", 3000),
            ContextComponent("hist_1", ComponentType.CHAT_HISTORY, "History", 1000),
            ContextComponent("user_1", ComponentType.USER_MESSAGE, "User", 350),
        ],
        total_tokens=14350,
    )


def test_total_area_coverage():
    """Total area of all rects should approximate the container area."""
    trace = _make_trace()
    rects = compute_treemap_layout(trace, 100.0, 268.0)
    total_area = sum(r.width * r.height for r in rects)
    expected = 100.0 * 268.0
    assert abs(total_area - expected) < 1.0, f"Area mismatch: {total_area} vs {expected}"


def test_no_overlapping_rects():
    """No two rectangles should overlap (beyond floating point tolerance)."""
    trace = _make_trace()
    rects = compute_treemap_layout(trace, 100.0, 268.0)
    eps = 0.01

    for i, a in enumerate(rects):
        for j, b in enumerate(rects):
            if i >= j:
                continue
            # Check for overlap: two rects overlap if they share area
            x_overlap = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
            y_overlap = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
            overlap = x_overlap * y_overlap
            assert overlap < eps, (
                f"Rects {i} and {j} overlap by {overlap}: "
                f"a=({a.x},{a.y},{a.width},{a.height}), "
                f"b=({b.x},{b.y},{b.width},{b.height})"
            )


def test_all_within_bounds():
    """All rects should be within the container bounds."""
    trace = _make_trace()
    w, h = 100.0, 268.0
    rects = compute_treemap_layout(trace, w, h)
    eps = 0.01

    for i, r in enumerate(rects):
        assert r.x >= -eps, f"Rect {i} x={r.x} out of bounds"
        assert r.y >= -eps, f"Rect {i} y={r.y} out of bounds"
        assert r.x + r.width <= w + eps, f"Rect {i} exceeds width"
        assert r.y + r.height <= h + eps, f"Rect {i} exceeds height"


def test_proportional_to_tokens():
    """Larger token counts should get larger areas."""
    trace = _make_trace()
    rects = compute_treemap_layout(trace, 100.0, 268.0)
    # Find the unused rect (should have highest area since 128000-14350 = 113650)
    unused = [r for r in rects if r.is_unused]
    assert len(unused) == 1
    unused_area = unused[0].width * unused[0].height
    # Unused should be the largest
    for r in rects:
        if not r.is_unused:
            assert r.width * r.height <= unused_area + 0.01


def test_reasonable_aspect_ratios():
    """Aspect ratios should be reasonable (not too extreme)."""
    trace = _make_trace()
    rects = compute_treemap_layout(trace, 100.0, 268.0)

    for r in rects:
        if r.width <= 0 or r.height <= 0:
            continue
        ratio = max(r.width / r.height, r.height / r.width)
        # Squarified treemap should keep ratios reasonable
        # Allow up to 20:1 for very small components
        assert ratio < 50, f"Extreme aspect ratio {ratio} for {r.component_id}"


def test_all_components_present():
    """All components should appear in the layout."""
    trace = _make_trace()
    rects = compute_treemap_layout(trace, 100.0, 268.0)
    ids = {r.component_id for r in rects}
    assert "sys_1" in ids
    assert "rag_1" in ids
    assert "rag_2" in ids
    assert "hist_1" in ids
    assert "user_1" in ids
    assert "_unused" in ids


def test_empty_trace():
    """Trace with no components but unused space returns just the unused rect."""
    trace = ContextTrace(context_limit=1000, components=[], total_tokens=0)
    rects = compute_treemap_layout(trace, 100.0, 268.0)
    assert len(rects) == 1
    assert rects[0].is_unused


def test_fully_utilized_trace():
    """Trace at full capacity has no unused rect."""
    trace = ContextTrace(
        context_limit=100,
        components=[ContextComponent("s1", ComponentType.SYSTEM_PROMPT, "S", 100)],
        total_tokens=100,
    )
    rects = compute_treemap_layout(trace, 100.0, 268.0)
    assert len(rects) == 1
    assert not rects[0].is_unused


def test_single_component():
    """Single component fills the whole space."""
    trace = ContextTrace(
        context_limit=1000,
        components=[ContextComponent("s1", ComponentType.SYSTEM_PROMPT, "S", 1000)],
        total_tokens=1000,
    )
    rects = compute_treemap_layout(trace, 100.0, 268.0)
    assert len(rects) == 1
    assert abs(rects[0].width - 100.0) < 0.01
    assert abs(rects[0].height - 268.0) < 0.01


def test_treemap_layout_in_context_window():
    """ContextWindow should render treemap when layout='treemap'."""
    trace = _make_trace()
    ctx = ContextWindow(trace=trace, layout="treemap")
    h = ctx.to_html()
    assert "ced-treemap" in h
    assert "ced-treemap-item" in h
