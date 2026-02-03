"""ContextDiff — Sankey diff view comparing before/after context traces."""

import html
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from context_engineering_dashboard.core.trace import ComponentType, ContextTrace
from context_engineering_dashboard.styles.colors import (
    COMPONENT_COLORS,
    COMPONENT_LABELS,
    UNUSED_COLOR,
)


@dataclass
class ContextDiff:
    """Compares two context traces and renders a Sankey diff diagram.

    Parameters
    ----------
    before : ContextTrace
        The original trace (before compaction/changes).
    after : ContextTrace
        The modified trace (after compaction/changes).
    before_label : str
        Label for the before column.
    after_label : str
        Label for the after column.
    """

    before: ContextTrace
    after: ContextTrace
    before_label: str = "Before"
    after_label: str = "After"

    def _repr_html_(self) -> str:
        """Jupyter auto-display."""
        return self._render_html()

    def sankey(self) -> None:
        """Render the Sankey diff in Jupyter."""
        from IPython.display import HTML
        from IPython.display import display as ipy_display

        ipy_display(HTML(self._render_html()))

    def summary(self) -> None:
        """Print a text summary of changes."""
        before_groups = self._group_by_type(self.before)
        after_groups = self._group_by_type(self.after)
        all_types = sorted(
            set(before_groups.keys()) | set(after_groups.keys()),
            key=lambda t: t.value,
        )

        lines = []
        lines.append(f"{'Component':<20} {'Before':>10} {'After':>10} {'Change':>10}")
        lines.append("-" * 52)

        for ct in all_types:
            b = before_groups.get(ct, 0)
            a = after_groups.get(ct, 0)
            diff = a - b
            pct = f"{diff / b * 100:+.0f}%" if b > 0 else ("new" if a > 0 else "—")
            label = COMPONENT_LABELS.get(ct, ct.value)
            lines.append(f"{label:<20} {b:>10,} {a:>10,} {pct:>10}")

        total_b = self.before.total_tokens
        total_a = self.after.total_tokens
        saved = total_b - total_a
        lines.append("-" * 52)
        lines.append(f"{'Total':<20} {total_b:>10,} {total_a:>10,} " f"{-saved:>+10,}")
        if saved > 0:
            lines.append(f"\nSaved {saved:,} tokens " f"({saved / total_b * 100:.1f}% reduction)")

        print("\n".join(lines))

    def _group_by_type(self, trace: ContextTrace) -> Dict[ComponentType, int]:
        groups: Dict[ComponentType, int] = defaultdict(int)
        for comp in trace.components:
            groups[comp.type] += comp.token_count
        return dict(groups)

    def _render_html(self) -> str:
        before_groups = self._group_by_type(self.before)
        after_groups = self._group_by_type(self.after)
        all_types = sorted(
            set(before_groups.keys()) | set(after_groups.keys()),
            key=lambda t: t.value,
        )

        total_b = self.before.total_tokens
        total_a = self.after.total_tokens
        saved = total_b - total_a
        pct_saved = f"{saved / total_b * 100:.1f}" if total_b > 0 else "0"

        # Layout constants
        svg_w = 800
        col_w = 150
        left_x = 20
        right_x = svg_w - col_w - 20
        padding_top = 20
        padding_bottom = 20
        gap = 5

        # Calculate number of visible components to determine usable height
        num_before = sum(1 for ct in all_types if before_groups.get(ct, 0) > 0)
        num_after = sum(1 for ct in all_types if after_groups.get(ct, 0) > 0)
        has_waste = saved > 0 and total_b > 0
        num_after_with_waste = num_after + (1 if has_waste else 0)
        max_components = max(num_before, num_after_with_waste, 1)

        # Set minimum usable height and calculate total SVG height
        min_usable_h = 260
        # Account for gaps: n components need (n-1) gaps
        total_gaps = gap * (max_components - 1) if max_components > 1 else 0
        usable_h = min_usable_h + total_gaps
        svg_h = usable_h + padding_top + padding_bottom

        # Build before rects
        before_rects = self._build_rects(
            before_groups, all_types, total_b, left_x, col_w, usable_h, gap
        )
        # Build after rects (include unused/freed space)
        after_types = list(all_types)
        after_rects = self._build_rects(
            after_groups, after_types, total_b, right_x, col_w, usable_h, gap
        )

        # Add waste/freed node if saved > 0
        if has_waste:
            # Calculate waste height proportionally
            waste_h = max(saved / total_b * (usable_h - total_gaps), 15)
            # Position after the last after rect, or calculate from bottom
            if after_rects:
                last_rect = after_rects[-1]
                waste_y = last_rect["y"] + last_rect["h"] + gap
            else:
                waste_y = padding_top
            after_rects.append(
                {
                    "type": None,
                    "x": right_x,
                    "y": waste_y,
                    "w": col_w,
                    "h": waste_h,
                    "tokens": saved,
                    "label": "UNUSED",
                    "color": UNUSED_COLOR,
                    "is_waste": True,
                }
            )

        # Calculate actual required height based on rendered content
        max_y = padding_top
        for r in before_rects + after_rects:
            rect_bottom = r["y"] + r["h"]
            if rect_bottom > max_y:
                max_y = rect_bottom
        svg_h = max(svg_h, max_y + padding_bottom)

        # SVG elements
        svg_parts = [f'<svg width="100%" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">']

        # Before rects
        for r in before_rects:
            svg_parts.append(self._rect_svg(r))

        # After rects
        for r in after_rects:
            svg_parts.append(self._rect_svg(r, show_change=True, before_groups=before_groups))

        # Flow paths
        svg_parts.extend(self._flow_paths(before_rects, after_rects, all_types))

        svg_parts.append("</svg>")

        # Wrap in container
        header = (
            f'<div style="display:flex;justify-content:space-between;margin-bottom:16px;'
            f"font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;"
            f'text-transform:uppercase;">'
            f"<span>{html.escape(self.before_label)}: {total_b:,} tokens</span>"
            f"<span>{html.escape(self.after_label)}: {total_a:,} tokens "
            f"({pct_saved}% saved)</span>"
            f"</div>"
        )

        return (
            f'<div class="ced-sankey-container" style="border:3px solid black;padding:16px;'
            f"font-family:'JetBrains Mono','IBM Plex Mono','Consolas',monospace;\">"
            f"{header}"
            f'{"".join(svg_parts)}'
            f"</div>"
        )

    def _build_rects(
        self,
        groups: Dict[ComponentType, int],
        types: List[ComponentType],
        total: int,
        x: float,
        w: float,
        usable_h: float,
        gap: float = 5,
    ) -> list:
        rects = []
        y = 20.0
        # Count visible components to calculate available height for bars (excluding gaps)
        visible_count = sum(1 for ct in types if groups.get(ct, 0) > 0)
        total_gaps = gap * (visible_count - 1) if visible_count > 1 else 0
        bars_h = usable_h - total_gaps  # Height available for actual bars

        for ct in types:
            tokens = groups.get(ct, 0)
            if tokens <= 0:
                continue
            h = max(tokens / total * bars_h, 15) if total > 0 else 15
            color = COMPONENT_COLORS.get(ct, "#999999")
            label = COMPONENT_LABELS.get(ct, ct.value).upper()
            rects.append(
                {
                    "type": ct,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "tokens": tokens,
                    "label": label,
                    "color": color,
                    "is_waste": False,
                }
            )
            y += h + gap
        return rects

    def _rect_svg(
        self,
        r: dict,
        show_change: bool = False,
        before_groups: Dict[ComponentType, int] | None = None,
    ) -> str:
        stroke_dash = ' stroke-dasharray="8,4"' if r.get("is_waste") else ""
        text_color = "#888" if r.get("is_waste") else "black"

        # Token label
        tok_label = r["label"]
        if r["tokens"] >= 1000:
            tok_label += f" {r['tokens'] // 1000}K"
        else:
            tok_label += f" {r['tokens']}"

        cx = r["x"] + r["w"] / 2
        cy = r["y"] + r["h"] / 2

        parts = [
            f'<rect x="{r["x"]}" y="{r["y"]}" width="{r["w"]}" height="{r["h"]}" '
            f'fill="{r["color"]}" stroke="black" stroke-width="3"{stroke_dash}/>',
            f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="monospace" font-size="11" font-weight="bold" '
            f'fill="{text_color}">{html.escape(tok_label)}</text>',
        ]

        # Change percentage badge
        if show_change and before_groups and r["type"] is not None and not r.get("is_waste"):
            before_val = before_groups.get(r["type"], 0)
            if before_val > 0:
                change = (r["tokens"] - before_val) / before_val * 100
                if abs(change) > 0.5:
                    change_str = f"{change:+.0f}%"
                    change_color = "#00AA55" if change < 0 else "#CC0000"
                    parts.append(
                        f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" '
                        f'font-family="monospace" font-size="9" '
                        f'fill="{change_color}">{change_str}</text>'
                    )

        # Freed tokens badge on waste node
        if r.get("is_waste") and r["tokens"] > 0:
            freed_str = (
                f"+{r['tokens'] // 1000}K freed" if r["tokens"] >= 1000 else f"+{r['tokens']} freed"
            )
            parts.append(
                f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" '
                f'font-family="monospace" font-size="10" fill="#888">'
                f"{html.escape(freed_str)}</text>"
            )

        return "\n".join(parts)

    def _flow_paths(
        self,
        before_rects: list,
        after_rects: list,
        all_types: List[ComponentType],
    ) -> List[str]:
        paths = []
        before_map = {r["type"]: r for r in before_rects if r["type"] is not None}
        after_map = {r["type"]: r for r in after_rects if r["type"] is not None}

        for ct in all_types:
            if ct not in before_map or ct not in after_map:
                continue
            br = before_map[ct]
            ar = after_map[ct]
            x1 = br["x"] + br["w"]
            y1 = br["y"] + br["h"] / 2
            x2 = ar["x"]
            y2 = ar["y"] + ar["h"] / 2
            mx = (x1 + x2) / 2
            stroke_w = max(min(br["h"], ar["h"]) * 0.8, 5)
            color = COMPONENT_COLORS.get(ct, "#999")
            paths.append(
                f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
                f'fill="none" stroke="{color}" stroke-width="{stroke_w:.0f}" '
                f'opacity="0.6"/>'
            )
        return paths
