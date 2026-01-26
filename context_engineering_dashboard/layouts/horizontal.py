"""Horizontal layout algorithm for context window visualization."""

from typing import Any, Dict, List

from context_engineering_dashboard.core.trace import ContextTrace


def compute_horizontal_layout(trace: ContextTrace) -> List[Dict[str, Any]]:
    """Compute horizontal layout items from a context trace.

    Returns a list of dicts with keys: id, type, token_count, flex, is_unused.
    Components are ordered as they appear in the trace, with unused space appended.
    """
    items: List[Dict[str, Any]] = []

    for comp in trace.components:
        flex = max(comp.token_count / 1000, 0.3)
        items.append(
            {
                "id": comp.id,
                "type": comp.type,
                "token_count": comp.token_count,
                "flex": round(flex, 2),
                "is_unused": False,
            }
        )

    unused = trace.unused_tokens
    if unused > 0:
        items.append(
            {
                "id": "_unused",
                "type": None,
                "token_count": unused,
                "flex": round(max(unused / 1000, 0.3), 2),
                "is_unused": True,
            }
        )

    return items
