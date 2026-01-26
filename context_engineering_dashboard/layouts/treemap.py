"""Squarified treemap layout algorithm (Bruls-Huizing-van Wijk)."""

from dataclasses import dataclass
from typing import List, Optional

from context_engineering_dashboard.core.trace import ComponentType, ContextTrace


@dataclass
class TreemapRect:
    """A rectangle in the treemap layout."""

    x: float
    y: float
    width: float
    height: float
    component_id: Optional[str]
    component_type: Optional[ComponentType]
    token_count: int
    is_unused: bool = False


def compute_treemap_layout(
    trace: ContextTrace,
    container_width: float = 100.0,
    container_height: float = 268.0,
) -> List[TreemapRect]:
    """Compute squarified treemap layout from a context trace.

    Parameters
    ----------
    trace : ContextTrace
        The trace to lay out.
    container_width : float
        Width of the container (in percentage units for CSS).
    container_height : float
        Height of the container in pixels.

    Returns
    -------
    List[TreemapRect]
        Rectangles positioned within the container bounds.
    """
    # Build items: list of (id, type, token_count, is_unused)
    items: List[tuple] = []
    for comp in trace.components:
        items.append((comp.id, comp.type, comp.token_count, False))

    unused = trace.unused_tokens
    if unused > 0:
        items.append(("_unused", None, unused, True))

    # Sort descending by token count
    items.sort(key=lambda x: x[2], reverse=True)

    # Normalize to area fractions
    total = sum(item[2] for item in items)
    if total == 0:
        return []

    areas = [item[2] / total * container_width * container_height for item in items]

    # Run squarified treemap algorithm
    raw_rects = _squarify(areas, 0, 0, container_width, container_height)

    result = []
    for i, (x, y, w, h) in enumerate(raw_rects):
        item = items[i]
        result.append(
            TreemapRect(
                x=x,
                y=y,
                width=w,
                height=h,
                component_id=item[0],
                component_type=item[1],
                token_count=item[2],
                is_unused=item[3],
            )
        )

    return result


def _squarify(
    areas: List[float],
    x: float,
    y: float,
    w: float,
    h: float,
) -> List[tuple]:
    """Squarified treemap algorithm.

    Returns list of (x, y, width, height) tuples.
    """
    if not areas:
        return []

    if len(areas) == 1:
        return [(x, y, w, h)]

    result: List[tuple] = [None] * len(areas)  # type: ignore[list-item]
    _squarify_recursive(areas, list(range(len(areas))), x, y, w, h, result)
    return result


def _squarify_recursive(
    areas: List[float],
    indices: List[int],
    x: float,
    y: float,
    w: float,
    h: float,
    result: List[tuple],
) -> None:
    """Recursive squarified treemap layout."""
    if not indices:
        return

    if len(indices) == 1:
        result[indices[0]] = (x, y, w, h)
        return

    total_area = sum(areas[i] for i in indices)
    if total_area <= 0:
        for i in indices:
            result[i] = (x, y, 0, 0)
        return

    # Lay along shorter side
    if w >= h:
        _layout_row_horizontal(areas, indices, x, y, w, h, total_area, result)
    else:
        _layout_row_vertical(areas, indices, x, y, w, h, total_area, result)


def _worst_ratio(row_areas: List[float], side: float) -> float:
    """Compute worst aspect ratio for a row laid along a given side."""
    s = sum(row_areas)
    if s <= 0 or side <= 0:
        return float("inf")
    ratios = []
    for a in row_areas:
        if a <= 0:
            continue
        row_w = s / side
        item_h = a / row_w if row_w > 0 else 0
        ratio = max(row_w / item_h, item_h / row_w) if item_h > 0 and row_w > 0 else float("inf")
        ratios.append(ratio)
    return max(ratios) if ratios else float("inf")


def _layout_row_horizontal(
    areas: List[float],
    indices: List[int],
    x: float,
    y: float,
    w: float,
    h: float,
    total_area: float,
    result: List[tuple],
) -> None:
    """Lay items along horizontal (width) axis."""
    row_indices = [indices[0]]
    rest_indices = list(indices[1:])
    row_areas = [areas[indices[0]]]

    while rest_indices:
        next_idx = rest_indices[0]
        candidate = row_areas + [areas[next_idx]]
        if _worst_ratio(candidate, h) <= _worst_ratio(row_areas, h):
            row_indices.append(next_idx)
            row_areas.append(areas[next_idx])
            rest_indices.pop(0)
        else:
            break

    # Layout this row as a vertical strip on the left
    row_total = sum(row_areas)
    row_width = row_total / h if h > 0 else 0

    cy = y
    for i, idx in enumerate(row_indices):
        item_h = row_areas[i] / row_width if row_width > 0 else 0
        result[idx] = (x, cy, row_width, item_h)
        cy += item_h

    # Recurse on remaining space
    _squarify_recursive(areas, rest_indices, x + row_width, y, w - row_width, h, result)


def _layout_row_vertical(
    areas: List[float],
    indices: List[int],
    x: float,
    y: float,
    w: float,
    h: float,
    total_area: float,
    result: List[tuple],
) -> None:
    """Lay items along vertical (height) axis."""
    row_indices = [indices[0]]
    rest_indices = list(indices[1:])
    row_areas = [areas[indices[0]]]

    while rest_indices:
        next_idx = rest_indices[0]
        candidate = row_areas + [areas[next_idx]]
        if _worst_ratio(candidate, w) <= _worst_ratio(row_areas, w):
            row_indices.append(next_idx)
            row_areas.append(areas[next_idx])
            rest_indices.pop(0)
        else:
            break

    # Layout this row as a horizontal strip on the top
    row_total = sum(row_areas)
    row_height = row_total / w if w > 0 else 0

    cx = x
    for i, idx in enumerate(row_indices):
        item_w = row_areas[i] / row_height if row_height > 0 else 0
        result[idx] = (cx, y, item_w, row_height)
        cx += item_w

    # Recurse on remaining space
    _squarify_recursive(areas, rest_indices, x, y + row_height, w, h - row_height, result)
