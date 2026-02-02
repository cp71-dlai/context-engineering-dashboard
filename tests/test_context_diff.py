"""Tests for ContextDiff Sankey diff view."""

import io
import sys

from context_engineering_dashboard import ContextDiff as ContextDiffTop
from context_engineering_dashboard.core.context_diff import ContextDiff
from context_engineering_dashboard.core.trace import (
    ComponentType,
    ContextComponent,
    ContextTrace,
)


def _make_before():
    return ContextTrace(
        context_limit=128000,
        components=[
            ContextComponent("sys_1", ComponentType.SYSTEM_PROMPT, "System", 4000),
            ContextComponent("hist_1", ComponentType.CHAT_HISTORY, "History", 20000),
            ContextComponent("rag_1", ComponentType.RAG, "RAG docs", 15000),
            ContextComponent("tool_1", ComponentType.TOOL, "Tool output", 1000),
        ],
        total_tokens=40000,
    )


def _make_after():
    return ContextTrace(
        context_limit=128000,
        components=[
            ContextComponent("sys_1", ComponentType.SYSTEM_PROMPT, "System", 4000),
            ContextComponent("hist_1", ComponentType.CHAT_HISTORY, "Condensed", 8000),
            ContextComponent("rag_1", ComponentType.RAG, "Trimmed RAG", 10000),
            ContextComponent("tool_1", ComponentType.TOOL, "Tool output", 1000),
        ],
        total_tokens=23000,
    )


def test_export():
    """ContextDiff is exported from the top-level package."""
    assert ContextDiffTop is ContextDiff


def test_sankey_html_contains_svg():
    diff = ContextDiff(before=_make_before(), after=_make_after())
    h = diff._repr_html_()
    assert "<svg" in h
    assert "</svg>" in h


def test_sankey_html_contains_rects():
    diff = ContextDiff(before=_make_before(), after=_make_after())
    h = diff._repr_html_()
    assert "<rect" in h


def test_sankey_html_contains_paths():
    diff = ContextDiff(before=_make_before(), after=_make_after())
    h = diff._repr_html_()
    assert "<path" in h


def test_sankey_percentage_badges():
    diff = ContextDiff(before=_make_before(), after=_make_after())
    h = diff._repr_html_()
    # History went from 20K -> 8K, that's -60%
    assert "-60%" in h
    # RAG went from 15K -> 10K, that's -33%
    assert "-33%" in h


def test_sankey_waste_node():
    diff = ContextDiff(before=_make_before(), after=_make_after())
    h = diff._repr_html_()
    assert "UNUSED" in h
    assert "freed" in h


def test_sankey_labels():
    diff = ContextDiff(
        before=_make_before(),
        after=_make_after(),
        before_label="Original",
        after_label="Compacted",
    )
    h = diff._repr_html_()
    assert "Original" in h
    assert "Compacted" in h


def test_sankey_token_counts():
    diff = ContextDiff(before=_make_before(), after=_make_after())
    h = diff._repr_html_()
    assert "40,000" in h
    assert "23,000" in h


def test_summary_output():
    diff = ContextDiff(before=_make_before(), after=_make_after())
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        diff.summary()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert "System" in output
    assert "History" in output
    assert "RAG" in output or "rag" in output.lower()
    assert "Tool" in output
    assert "Total" in output
    assert "Saved" in output or "saved" in output.lower()


def test_removed_component():
    """A component in before but not after should still render."""
    before = ContextTrace(
        context_limit=10000,
        components=[
            ContextComponent("mem_1", ComponentType.SCRATCHPAD, "Memory", 3000),
            ContextComponent("sys_1", ComponentType.SYSTEM_PROMPT, "System", 2000),
        ],
        total_tokens=5000,
    )
    after = ContextTrace(
        context_limit=10000,
        components=[
            ContextComponent("sys_1", ComponentType.SYSTEM_PROMPT, "System", 2000),
        ],
        total_tokens=2000,
    )
    diff = ContextDiff(before=before, after=after)
    h = diff._repr_html_()
    assert "<svg" in h
    assert "freed" in h


def test_new_component():
    """A component in after but not before should still render."""
    before = ContextTrace(
        context_limit=10000,
        components=[
            ContextComponent("sys_1", ComponentType.SYSTEM_PROMPT, "System", 2000),
        ],
        total_tokens=2000,
    )
    after = ContextTrace(
        context_limit=10000,
        components=[
            ContextComponent("sys_1", ComponentType.SYSTEM_PROMPT, "System", 2000),
            ContextComponent("mem_1", ComponentType.SCRATCHPAD, "Memory", 3000),
        ],
        total_tokens=5000,
    )
    diff = ContextDiff(before=before, after=after)
    h = diff._repr_html_()
    assert "<svg" in h
