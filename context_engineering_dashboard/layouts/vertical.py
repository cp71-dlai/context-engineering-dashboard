"""Vertical layout algorithm for context window visualization."""

from typing import Any, Dict, List

from context_engineering_dashboard.core.trace import ContextTrace


def compute_vertical_layout(trace: ContextTrace) -> List[Dict[str, Any]]:
    """Compute vertical layout items from a context trace.

    Returns a list of dicts with keys: id, type, token_count, height, is_unused.
    Components are ordered as they appear in the trace, with unused space appended.

    Heights are calculated proportionally to token share of the context limit,
    with a minimum height of 40px and maximum of 200px for any single component.
    """
    items: List[Dict[str, Any]] = []
    total = trace.context_limit if trace.context_limit > 0 else 1
    min_height = 40  # px minimum for visibility
    max_height = 200  # px maximum cap
    total_height = 400  # px total available height for components

    for comp in trace.components:
        ratio = comp.token_count / total
        height = max(min_height, min(max_height, ratio * total_height))
        items.append(
            {
                "id": comp.id,
                "type": comp.type,
                "token_count": comp.token_count,
                "height": round(height, 1),
                "is_unused": False,
            }
        )

    unused = trace.unused_tokens
    if unused > 0:
        ratio = unused / total
        height = max(min_height, min(max_height, ratio * total_height))
        items.append(
            {
                "id": "_unused",
                "type": None,
                "token_count": unused,
                "height": round(height, 1),
                "is_unused": True,
            }
        )

    return items
