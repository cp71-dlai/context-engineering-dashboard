"""ContextBuilder — main Jupyter widget for building and visualizing LLM context windows."""

import copy
import html
import json
import uuid
from typing import TYPE_CHECKING, List, Optional

from context_engineering_dashboard.core.trace import ComponentType, ContextComponent, ContextTrace
from context_engineering_dashboard.layouts.horizontal import compute_horizontal_layout
from context_engineering_dashboard.styles.colors import (
    COMPONENT_COLORS,
    COMPONENT_ICONS,
    COMPONENT_LABELS,
    CSS_CLASSES,
    TEXT_COLORS,
    UNUSED_COLOR,
    UNUSED_TEXT_COLOR,
)

if TYPE_CHECKING:
    from context_engineering_dashboard.core.resource import ContextResource


class ContextBuilder:
    """Stateful editor for building and visualizing LLM context windows.

    ContextBuilder provides:
    - Neo-brutalist visualization in Jupyter notebooks
    - Two-panel view with Available resources and Context Builder
    - Drag-and-drop between panels to select/deselect items
    - In-browser editing with Save button persistence
    - Export of modified traces

    Parameters
    ----------
    trace : ContextTrace, optional
        The trace data to visualize. Can be None if only using resources.
    context_limit : int
        Maximum context window size in tokens.
    layout : str
        Layout algorithm: "horizontal" or "treemap".
    resources : List[ContextResource], optional
        Resource pools to show in Available panel.

    Interactions
    ------------
    - Hover: Tooltip with component type and token count
    - Click: Modal with full content and metadata
    - Click on text in modal: Switch to edit mode with Save button
    - Drag items: Move between Available and Context panels

    Examples
    --------
    >>> rag = ContextResource.from_chroma(collection, ResourceType.RAG)
    >>> rag.query(query_texts=["How do I..."], n_results=10)
    >>> rag.select(["doc_1", "doc_2"])
    >>>
    >>> builder = ContextBuilder(resources=[rag])
    >>> builder.display()
    >>> # After dragging items and clicking Save...
    >>> builder.apply_selections()
    >>> new_trace = builder.get_trace()
    """

    def __init__(
        self,
        trace: Optional[ContextTrace] = None,
        context_limit: int = 128_000,
        layout: str = "horizontal",
        resources: Optional[List["ContextResource"]] = None,
    ) -> None:
        # Initialize trace (create empty if not provided)
        if trace is None:
            trace = ContextTrace(
                context_limit=context_limit,
                components=[],
                total_tokens=0,
            )
        self._original_trace = trace  # Immutable reference
        self._working_trace = copy.deepcopy(trace)  # Mutable copy for edits
        self._edits: dict = {}  # Track edit history
        self._reorder: Optional[List[str]] = None  # Track reorder history
        self.context_limit = context_limit or trace.context_limit
        self.layout = layout.lower()
        self._resources = resources or []
        self._uid = uuid.uuid4().hex[:12]
        self._pending_selections: dict = {}  # Track pending selection changes

    @property
    def trace(self) -> ContextTrace:
        """Return the working trace (for backward compatibility)."""
        return self._working_trace

    def get_trace(self) -> ContextTrace:
        """Return a deep copy of the current working trace.

        Returns
        -------
        ContextTrace
            A new ContextTrace object with all current edits applied.
        """
        return copy.deepcopy(self._working_trace)

    def apply_edit(self, component_id: str, new_content: str) -> None:
        """Edit a component's content and recount tokens.

        Parameters
        ----------
        component_id : str
            The ID of the component to edit.
        new_content : str
            The new content for the component.

        Raises
        ------
        KeyError
            If the component_id is not found.
        """
        for comp in self._working_trace.components:
            if comp.id == component_id:
                old_content = comp.content
                old_tokens = comp.token_count
                new_tokens = self._count_tokens(new_content)

                # Update component
                # Create new component with updated values (dataclass is immutable)
                idx = self._working_trace.components.index(comp)
                self._working_trace.components[idx] = ContextComponent(
                    id=comp.id,
                    type=comp.type,
                    content=new_content,
                    token_count=new_tokens,
                    metadata=comp.metadata,
                )

                # Update total tokens
                self._working_trace.total_tokens += new_tokens - old_tokens

                # Track edit
                self._edits[component_id] = {
                    "original": old_content,
                    "edited": new_content,
                    "original_tokens": old_tokens,
                    "new_tokens": new_tokens,
                }
                return

        raise KeyError(f"Component '{component_id}' not found")

    def apply_reorder(self, new_order: List[str]) -> None:
        """Reorder components according to the given order.

        Parameters
        ----------
        new_order : List[str]
            List of component IDs in the desired order.
        """
        # Build a map of id -> component
        comp_map = {c.id: c for c in self._working_trace.components}

        # Reorder components
        reordered = []
        for comp_id in new_order:
            if comp_id in comp_map:
                reordered.append(comp_map[comp_id])

        # Add any components not in new_order (shouldn't happen, but be safe)
        for comp in self._working_trace.components:
            if comp.id not in new_order:
                reordered.append(comp)

        self._working_trace.components = reordered
        self._reorder = new_order

    def reset(self) -> None:
        """Reset to the original trace, discarding all edits."""
        self._working_trace = copy.deepcopy(self._original_trace)
        self._edits = {}
        self._reorder = None

    def has_changes(self) -> bool:
        """Check if any edits have been made.

        Returns
        -------
        bool
            True if the trace has been modified.
        """
        return bool(self._edits) or self._reorder is not None

    def to_json(self, path: str) -> None:
        """Save the working trace to a JSON file.

        Parameters
        ----------
        path : str
            File path to save to.
        """
        self._working_trace.to_json(path)

    def _count_tokens(self, content: str) -> int:
        """Count tokens in content using tiktoken."""
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(content))
        except Exception:
            # Fallback: rough estimate
            return len(content) // 4

    @property
    def resources(self) -> List["ContextResource"]:
        """Return the list of resources."""
        return self._resources

    def apply_selections(self, selections: Optional[dict] = None) -> None:
        """Apply selection changes from browser to resources.

        This method updates each resource's selected_ids based on
        the selections made via drag-and-drop in the browser.

        Parameters
        ----------
        selections : dict, optional
            Dict mapping resource_name -> list of selected item IDs.
            If not provided, uses pending selections from _pending_selections.
        """
        if selections is None:
            selections = self._pending_selections

        for resource in self._resources:
            if resource.name in selections:
                resource.selected_ids = set(selections[resource.name])

        # Rebuild components from selected resource items
        self._rebuild_components_from_resources()
        self._pending_selections = {}

    def _rebuild_components_from_resources(self) -> None:
        """Rebuild trace components from currently selected resource items."""
        # Keep non-resource components (system prompt, user message, etc.)
        resource_types = {r.resource_type.to_component_type() for r in self._resources}
        non_resource_components = [
            c for c in self._working_trace.components if c.type not in resource_types
        ]

        # Add selected items from resources as components
        resource_components = []
        for resource in self._resources:
            resource_components.extend(resource.to_components())

        # Combine and update trace
        self._working_trace.components = non_resource_components + resource_components
        self._working_trace.total_tokens = sum(
            c.token_count for c in self._working_trace.components
        )

    def _repr_html_(self) -> str:
        """Jupyter auto-display."""
        return self.to_html()

    def display(self) -> None:
        """Display in IPython/Jupyter."""
        from IPython.display import HTML
        from IPython.display import display as ipy_display

        ipy_display(HTML(self.to_html()))

    def to_html(self) -> str:
        """Generate the full HTML string."""
        uid = self._uid
        parts = [
            f'<div id="ced-{uid}" class="ced-container">',
            f"<style>{self._css(uid)}</style>",
            self._header_html(uid),
        ]

        if self._resources:
            parts.append(self._resources_panel_html(uid))
        else:
            parts.append(self._context_window_html(uid))

        parts.append(self._legend_html(uid))

        # Tooltip div
        parts.append(
            f'<div id="ced-tooltip-{uid}" class="ced-tooltip" style="display:none;"></div>'
        )

        # Modal overlay (hidden by default)
        parts.append(self._modal_html(uid))

        # Component data + JS
        parts.append(self._component_data_script(uid))
        parts.append(f"<script>{self._js(uid)}</script>")
        parts.append("</div>")
        return "\n".join(parts)

    # ------------------------------------------------------------------ CSS
    def _css(self, uid: str) -> str:
        s = f"#ced-{uid}"
        # Build component color rules dynamically
        comp_rules = []
        for ct, css_cls in CSS_CLASSES.items():
            bg = COMPONENT_COLORS[ct]
            fg = TEXT_COLORS[ct]
            comp_rules.append(f"{s} .{css_cls} {{" f" background: {bg}; color: {fg}; }}")
        comp_css = "\n".join(comp_rules)
        return f"""
{s} * {{ box-sizing: border-box; margin: 0; padding: 0; }}
{s} {{
  font-family: 'JetBrains Mono', 'IBM Plex Mono',
    'Consolas', monospace;
  line-height: 1.4;
  position: relative;
}}
{s} .ced-header {{
  display: flex; justify-content: space-between;
  align-items: center;
  border: 3px solid black; padding: 16px;
  margin-bottom: 16px; background: #F5F5F5;
}}
{s} .ced-title {{
  font-size: 14px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1px;
}}
{s} .ced-controls {{ display: flex; gap: 8px; }}
{s} .ced-btn {{
  border: 2px solid black; background: white;
  padding: 4px 8px; font-family: inherit;
  font-size: 11px; font-weight: 700;
  text-transform: uppercase; cursor: pointer;
  transition: background 0.1s;
}}
{s} .ced-btn:hover {{ background: black; color: white; }}
{s} .ced-btn.ced-active {{ background: black; color: white; }}
{s} .ced-context-window {{
  border: 3px solid black; background: white;
  position: relative; margin-bottom: 24px;
}}
{s} .ced-window-label {{
  position: absolute; top: -12px; left: 16px;
  background: white; padding: 0 4px;
  font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1px;
}}
{s} .ced-token-counter {{
  position: absolute; top: -12px; right: 16px;
  background: white; padding: 0 4px;
  font-size: 10px; font-weight: 700;
}}
{s} .ced-horizontal {{
  display: flex; height: 120px; padding: 16px; gap: 2px;
}}
{s} .ced-component {{
  display: flex; flex-direction: column;
  justify-content: center; align-items: center;
  border: 3px solid black; padding: 8px;
  cursor: pointer; position: relative;
  transition: transform 0.1s;
  overflow: hidden; min-width: 30px;
}}
{s} .ced-component:hover {{
  transform: translateY(-2px); z-index: 10;
}}
{s} .ced-component .ced-icon {{
  font-size: 20px; margin-bottom: 4px;
}}
{s} .ced-component .ced-label {{
  font-size: 9px; font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px; text-align: center;
  white-space: nowrap;
}}
{s} .ced-component .ced-tokens {{
  font-size: 10px; font-weight: 400; margin-top: 2px;
}}
{comp_css}
{s} .ced-comp-unused {{
  background: {UNUSED_COLOR}; border-style: dashed;
  color: {UNUSED_TEXT_COLOR};
}}
{s} .ced-comp-unused.ced-collapsed {{
  flex: 0.15 !important; min-width: 24px; max-width: 40px;
}}
{s} .ced-comp-unused.ced-collapsed .ced-label,
{s} .ced-comp-unused.ced-collapsed .ced-tokens {{
  display: none;
}}
{s} .ced-comp-unused .ced-lacuna {{
  display: none; font-size: 12px; font-weight: 700;
  writing-mode: vertical-rl; text-orientation: mixed;
}}
{s} .ced-comp-unused.ced-collapsed .ced-lacuna {{
  display: block;
}}
{s} .ced-treemap .ced-comp-unused.ced-collapsed {{
  width: 24px !important; left: auto !important; right: 16px !important;
}}
{s} .ced-treemap .ced-comp-unused.ced-collapsed .ced-label,
{s} .ced-treemap .ced-comp-unused.ced-collapsed .ced-tokens {{
  display: none;
}}
{s} .ced-score-badge {{
  position: absolute; top: 4px; right: 4px;
  background: black; color: white; font-size: 9px;
  font-weight: 700; padding: 2px 4px;
}}
{s} .ced-legend {{
  display: flex; flex-wrap: wrap; gap: 16px; padding: 16px;
  border: 3px solid black; margin-top: 24px;
}}
{s} .ced-legend-item {{ display: flex; align-items: center; gap: 4px; }}
{s} .ced-legend-swatch {{
  width: 16px; height: 16px; border: 2px solid black;
}}
{s} .ced-legend-swatch.ced-dashed {{ border-style: dashed; }}
{s} .ced-legend-label {{
  font-size: 10px; font-weight: 700; text-transform: uppercase;
}}
{s} .ced-tooltip {{
  position: absolute; background: black; color: white;
  padding: 4px 8px; font-size: 10px; font-weight: 700;
  text-transform: uppercase; pointer-events: none;
  white-space: nowrap; z-index: 100;
  font-family: 'JetBrains Mono', 'IBM Plex Mono', 'Consolas', monospace;
}}
{s} .ced-modal-overlay {{
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  min-height: 500px;
  background: rgba(0,0,0,0.5); display: flex;
  align-items: center; justify-content: center; z-index: 1000;
}}
{s} .ced-modal {{
  background: white; border: 3px solid black; width: 600px;
  max-width: 90%; max-height: 450px; display: flex; flex-direction: column;
  font-family: 'JetBrains Mono', 'IBM Plex Mono', 'Consolas', monospace;
}}
{s} .ced-modal-header {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 16px; border-bottom: 3px solid black;
}}
{s} .ced-modal-title {{
  font-size: 12px; font-weight: 700; text-transform: uppercase;
}}
{s} .ced-modal-actions {{
  display: flex; align-items: center; gap: 8px;
}}
{s} .ced-modal-save {{
  padding: 4px 12px; font-size: 11px;
}}
{s} .ced-modal-close {{
  background: white; border: 2px solid black; width: 28px; height: 28px;
  font-size: 16px; font-weight: 700; cursor: pointer;
  font-family: inherit;
}}
{s} .ced-modal-close:hover {{ background: black; color: white; }}
{s} .ced-modal-body {{
  padding: 16px; overflow-y: auto; max-height: 350px;
}}
{s} .ced-modal-section {{ margin-bottom: 16px; }}
{s} .ced-modal-section-title {{
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 4px; color: #666;
}}
{s} .ced-modal-text {{
  font-size: 12px; line-height: 1.6; background: #F5F5F5;
  padding: 16px; border: 2px solid black;
  white-space: pre-wrap; word-break: break-word;
  max-height: 200px; overflow-y: auto;
  cursor: pointer; transition: background 0.1s;
}}
{s} .ced-modal-text:hover {{ background: #E8E8E8; }}
{s} .ced-modal-text-hint {{
  font-size: 9px; color: #888; margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.5px;
}}
{s} .ced-metadata-table {{
  width: 100%; border-collapse: collapse; font-size: 11px;
}}
{s} .ced-metadata-table th, {s} .ced-metadata-table td {{
  border: 2px solid black; padding: 8px; text-align: left;
}}
{s} .ced-metadata-table th {{
  background: #F5F5F5; font-weight: 700; text-transform: uppercase; font-size: 10px;
}}
{s} .ced-modal-textarea {{
  width: 100%; min-height: 150px; max-height: 200px; font-family: inherit;
  font-size: 12px; line-height: 1.6; padding: 16px;
  border: 2px solid black; resize: vertical; overflow-y: auto;
}}
{s} .ced-two-panel {{
  display: grid; grid-template-columns: 1fr auto 1fr;
  gap: 16px; align-items: start; margin-bottom: 24px;
}}
{s} .ced-panel {{ border: 3px solid black; }}
{s} .ced-panel-header {{
  background: #F5F5F5; border-bottom: 3px solid black;
  padding: 8px 16px; font-size: 11px; font-weight: 700;
  text-transform: uppercase;
}}
{s} .ced-panel-content {{ padding: 16px; }}
{s} .ced-doc-item {{
  display: flex; align-items: center; gap: 8px; padding: 8px;
  border: 2px solid black; margin-bottom: 4px; font-size: 11px;
}}
{s} .ced-doc-item.ced-selected {{
  background: {COMPONENT_COLORS[ComponentType.RAG]};
  color: white;
}}
{s} .ced-doc-item.ced-unselected {{
  background: #F5F5F5; color: #888; border-style: dashed;
}}
{s} .ced-doc-score {{
  background: black; color: white; padding: 2px 6px;
  font-weight: 700; font-size: 10px;
}}
{s} .ced-doc-tokens {{ margin-left: auto; font-size: 10px; }}
{s} .ced-doc-check {{ font-weight: 700; }}
{s} .ced-scissors {{
  display: flex; align-items: center; justify-content: center;
  font-size: 32px; padding: 16px;
}}
{s} .ced-panel-divider {{
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 8px; gap: 8px;
}}
{s} .ced-save-selections {{
  background: #00AA55; color: white; border: 2px solid black;
  padding: 8px 16px; font-weight: 700;
}}
{s} .ced-save-selections:hover {{
  background: #008844;
}}
{s} .ced-resource-header {{
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  padding: 8px; background: #F5F5F5; margin-bottom: 4px;
  border: 2px solid black;
}}
{s} .ced-token-badge {{
  font-size: 9px; font-weight: 400; margin-left: 8px;
}}
{s} .ced-doc-item[draggable="true"] {{
  cursor: grab;
}}
{s} .ced-doc-item[draggable="true"]:active {{
  cursor: grabbing;
}}
{s} .ced-doc-item.ced-dragging {{
  opacity: 0.5; box-shadow: 0 2px 0 black;
}}
{s} .ced-panel.ced-drop-target {{
  background: #E8F4E8; border-color: #00AA55;
}}
{s} .ced-panel.ced-drop-target .ced-panel-header {{
  background: #00AA55; color: white;
}}
{s} .ced-available-panel .ced-panel-content {{
  max-height: 400px; overflow-y: auto;
}}
{s} .ced-context-panel .ced-panel-content {{
  max-height: 400px; overflow-y: auto;
}}
{s} .ced-treemap {{
  position: relative; height: 300px; padding: 16px;
}}
{s} .ced-treemap-item {{
  position: absolute; display: flex; flex-direction: column;
  justify-content: center; align-items: center;
  border: 3px solid black; cursor: pointer; overflow: hidden;
  transition: transform 0.1s;
}}
{s} .ced-treemap-item:hover {{ z-index: 10; }}
{s} .ced-treemap-item .ced-icon {{ font-size: 16px; margin-bottom: 2px; }}
{s} .ced-treemap-item .ced-label {{
  font-size: 9px; font-weight: 700; text-transform: uppercase;
  text-align: center; white-space: nowrap;
}}
{s} .ced-treemap-item .ced-tokens {{ font-size: 9px; margin-top: 1px; }}
{s} .ced-component.ced-dragging {{
  opacity: 0.5; transform: scale(1.02); z-index: 1000;
  cursor: grabbing !important; box-shadow: 0 4px 0 black;
}}
{s} .ced-component.ced-drop-left::before {{
  content: ''; position: absolute; left: -4px; top: 0; bottom: 0;
  width: 4px; background: #0066FF;
}}
{s} .ced-component.ced-drop-right::after {{
  content: ''; position: absolute; right: -4px; top: 0; bottom: 0;
  width: 4px; background: #0066FF;
}}
{s} .ced-horizontal.ced-dragging-active {{
  cursor: grabbing;
}}
{s} .ced-horizontal.ced-dragging-active .ced-comp-unused {{
  opacity: 0.3; pointer-events: none;
}}
"""

    # -------------------------------------------------------------- Header
    def _header_html(self, uid: str) -> str:
        return (
            f'<div class="ced-header">'
            f'<span class="ced-title">Context Builder</span>'
            f'<div class="ced-controls">'
            f'<button class="ced-btn">\u2699</button>'
            f"</div>"
            f"</div>"
        )

    # ------------------------------------------------------ Context window
    def _context_window_html(self, uid: str) -> str:
        total = self.trace.total_tokens
        limit = self.context_limit
        pct = round(total / limit * 100) if limit else 0
        token_str = f"{total:,} / {limit:,} TOKENS ({pct}%)"

        parts = [
            '<div class="ced-context-window">',
            '<span class="ced-window-label">Context Builder</span>',
            f'<span class="ced-token-counter">{token_str}</span>',
        ]

        # Horizontal layout
        h_style = "" if self.layout == "horizontal" else "display:none;"
        parts.append(f'<div class="ced-horizontal" id="ced-hlayout-{uid}" style="{h_style}">')
        items = compute_horizontal_layout(self.trace)
        for item in items:
            parts.append(self._component_div(item, uid))
        parts.append("</div>")

        # Treemap layout placeholder (filled in Phase 6)
        t_style = "" if self.layout == "treemap" else "display:none;"
        parts.append(f'<div class="ced-treemap" id="ced-tlayout-{uid}" style="{t_style}">')
        if self.layout == "treemap":
            parts.append(self._treemap_items_html(uid))
        parts.append("</div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _treemap_items_html(self, uid: str) -> str:
        """Render treemap items using absolute positioning."""
        try:
            from context_engineering_dashboard.layouts.treemap import compute_treemap_layout
        except ImportError:
            return ""

        container_w = 100.0  # percentage
        container_h = 268.0  # pixels (300 - 2*16 padding)
        rects = compute_treemap_layout(self.trace, container_w, container_h)
        parts = []
        for r in rects:
            if r.is_unused or r.component_type is None:
                css_cls = "ced-comp-unused"
            else:
                css_cls = CSS_CLASSES.get(r.component_type, "")

            icon = ""
            label = "Unused"
            lacuna = ""
            if r.is_unused:
                lacuna = '<span class="ced-lacuna">\\...\\</span>'
            if not r.is_unused and r.component_type is not None:
                icon = COMPONENT_ICONS.get(r.component_type, "")
                label = COMPONENT_LABELS.get(r.component_type, "")
                if r.component_id and r.component_id.startswith("rag_"):
                    label = f"RAG {r.component_id}"

            score_badge = ""
            if not r.is_unused:
                for comp in self.trace.components:
                    if comp.id == r.component_id:
                        sc = comp.metadata.get("chroma_score")
                        if sc is not None:
                            score_badge = f'<span class="ced-score-badge">{sc}</span>'
                        break

            parts.append(
                f'<div class="ced-treemap-item {css_cls}" '
                f'data-comp-id="{html.escape(r.component_id or "_unused")}" '
                f'data-orig-x="{r.x}" data-orig-y="{r.y}" '
                f'data-orig-w="{r.width}" data-orig-h="{r.height}" '
                f'data-tokens="{r.token_count}" '
                f'style="left:{r.x}%;top:{r.y}px;width:{r.width}%;height:{r.height}px;">'
                f"{score_badge}"
                f'<span class="ced-icon">{icon}</span>'
                f'<span class="ced-label">{html.escape(label)}</span>'
                f'<span class="ced-tokens">{r.token_count:,}</span>'
                f"{lacuna}"
                f"</div>"
            )
        return "\n".join(parts)

    def _component_div(self, item: dict, uid: str) -> str:
        """Render a single component div for horizontal layout."""
        if item["is_unused"]:
            return (
                f'<div class="ced-component ced-comp-unused" '
                f'style="flex:{item["flex"]};" data-comp-id="_unused">'
                f'<span class="ced-label">Unused</span>'
                f'<span class="ced-tokens">{item["token_count"]:,}</span>'
                f'<span class="ced-lacuna">\\...\\</span>'
                f"</div>"
            )

        comp_type = item["type"]
        css_cls = CSS_CLASSES.get(comp_type, "")
        icon = COMPONENT_ICONS.get(comp_type, "")
        label = COMPONENT_LABELS.get(comp_type, "")
        comp_id = html.escape(item["id"])

        # Score badge for RAG docs
        score_badge = ""
        for comp in self.trace.components:
            if comp.id == item["id"]:
                sc = comp.metadata.get("chroma_score")
                if sc is not None:
                    score_badge = f'<span class="ced-score-badge">{sc}</span>'
                break

        return (
            f'<div class="ced-component {css_cls}" '
            f'style="flex:{item["flex"]};" data-comp-id="{comp_id}">'
            f"{score_badge}"
            f'<span class="ced-icon">{icon}</span>'
            f'<span class="ced-label">{html.escape(label)}</span>'
            f'<span class="ced-tokens">{item["token_count"]:,}</span>'
            f"</div>"
        )

    # -------------------------------------------------------- Resources panel
    def _resources_panel_html(self, uid: str) -> str:
        """Render two-panel view with Available resources and Context Builder."""
        # Calculate total selected tokens
        total_selected = sum(r.total_selected_tokens for r in self._resources)
        total_selected += sum(
            c.token_count
            for c in self._working_trace.components
            if c.type not in {r.resource_type.to_component_type() for r in self._resources}
        )

        # Left panel: Available items from all resources
        left_items = []
        for resource in self._resources:
            # Resource header
            left_items.append(
                f'<div class="ced-resource-header" data-resource="{html.escape(resource.name)}">'
                f"{html.escape(resource.name)} ({len(resource.items)} items)"
                f"</div>"
            )
            # Sort by score if available
            sorted_items = sorted(
                resource.items, key=lambda x: x.score if x.score is not None else 0, reverse=True
            )
            for item in sorted_items:
                is_selected = item.id in resource.selected_ids
                sel_cls = "ced-selected" if is_selected else "ced-unselected"
                score_badge = ""
                if item.score is not None:
                    score_badge = f'<span class="ced-doc-score">{item.score:.2f}</span>'
                check = "\u2713" if is_selected else ""
                res_type_color = COMPONENT_COLORS.get(resource.resource_type.to_component_type(), "#999")

                left_items.append(
                    f'<div class="ced-doc-item {sel_cls}" '
                    f'data-item-id="{html.escape(item.id)}" '
                    f'data-resource="{html.escape(resource.name)}" '
                    f'draggable="true" '
                    f'style="border-left: 4px solid {res_type_color};">'
                    f"{score_badge}"
                    f"<span>{html.escape(item.id)}</span>"
                    f'<span class="ced-doc-tokens">{item.token_count:,} tok</span>'
                    f'<span class="ced-doc-check">{check}</span>'
                    f"</div>"
                )

        # Right panel: Selected items + trace components
        right_items = []

        # Add selected resource items
        for resource in self._resources:
            for item in resource.selected_items:
                score_badge = ""
                if item.score is not None:
                    score_badge = f'<span class="ced-doc-score">{item.score:.2f}</span>'
                res_type_color = COMPONENT_COLORS.get(resource.resource_type.to_component_type(), "#999")
                res_type_label = COMPONENT_LABELS.get(resource.resource_type.to_component_type(), "")

                right_items.append(
                    f'<div class="ced-doc-item ced-selected" '
                    f'data-item-id="{html.escape(item.id)}" '
                    f'data-resource="{html.escape(resource.name)}" '
                    f'draggable="true" '
                    f'style="background: {res_type_color}; color: white;">'
                    f"{score_badge}"
                    f"<span>{html.escape(item.id)}</span>"
                    f'<span class="ced-doc-tokens">{item.token_count:,} tok</span>'
                    f"</div>"
                )

        # Add non-resource trace components
        resource_types = {r.resource_type.to_component_type() for r in self._resources}
        for comp in self._working_trace.components:
            if comp.type not in resource_types:
                css_cls = CSS_CLASSES.get(comp.type, "")
                icon = COMPONENT_ICONS.get(comp.type, "")
                label = COMPONENT_LABELS.get(comp.type, "")
                right_items.append(
                    f'<div class="ced-doc-item {css_cls}" '
                    f'data-comp-id="{html.escape(comp.id)}" '
                    f'style="border: 2px solid black;">'
                    f"<span>{icon} {html.escape(label)}</span>"
                    f'<span class="ced-doc-tokens">{comp.token_count:,} tok</span>'
                    f"</div>"
                )

        pct = round(total_selected / self.context_limit * 100) if self.context_limit else 0
        token_str = f"{total_selected:,} / {self.context_limit:,} TOKENS ({pct}%)"

        return (
            f'<div class="ced-two-panel" id="ced-panels-{uid}">'
            f'<div class="ced-panel ced-available-panel" id="ced-available-{uid}">'
            f'<div class="ced-panel-header">Available</div>'
            f'<div class="ced-panel-content" id="ced-available-content-{uid}">{"".join(left_items)}</div>'
            f"</div>"
            f'<div class="ced-panel-divider">'
            f'<button class="ced-btn ced-save-selections" id="ced-save-btn-{uid}" '
            f'style="display:none;">Save</button>'
            f"</div>"
            f'<div class="ced-panel ced-context-panel" id="ced-context-{uid}">'
            f'<div class="ced-panel-header">Context Builder <span class="ced-token-badge">{token_str}</span></div>'
            f'<div class="ced-panel-content" id="ced-context-content-{uid}">{"".join(right_items)}</div>'
            f"</div>"
            f"</div>"
        )

    # -------------------------------------------------------------- Legend
    def _legend_html(self, uid: str) -> str:
        items = []
        for ct in ComponentType:
            color = COMPONENT_COLORS[ct]
            label = COMPONENT_LABELS[ct]
            items.append(
                f'<div class="ced-legend-item">'
                f'<div class="ced-legend-swatch" style="background:{color};"></div>'
                f'<span class="ced-legend-label">{html.escape(label)}</span>'
                f"</div>"
            )
        # Unused
        items.append(
            f'<div class="ced-legend-item">'
            f'<div class="ced-legend-swatch ced-dashed" '
            f'style="background:{UNUSED_COLOR};"></div>'
            f'<span class="ced-legend-label">Unused</span>'
            f"</div>"
        )
        return f'<div class="ced-legend">{"".join(items)}</div>'

    # -------------------------------------------------------------- Modal
    def _modal_html(self, uid: str) -> str:
        return (
            f'<div id="ced-modal-{uid}" class="ced-modal-overlay" style="display:none;">'
            f'<div class="ced-modal">'
            f'<div class="ced-modal-header" id="ced-modal-header-{uid}">'
            f'<span class="ced-modal-title" id="ced-modal-title-{uid}"></span>'
            f'<div class="ced-modal-actions">'
            f'<button class="ced-btn ced-modal-save" id="ced-modal-save-{uid}" '
            f'style="display:none;">Save</button>'
            f'<button class="ced-modal-close" '
            f'onclick="cedCloseModal_{uid}()">\u2715</button>'
            f"</div>"
            f"</div>"
            f'<div class="ced-modal-body" id="ced-modal-body-{uid}"></div>'
            f"</div>"
            f"</div>"
        )

    # ------------------------------------------------- Component data JSON
    def _component_data_script(self, uid: str) -> str:
        """Embed component data as a JS object for interaction modes."""
        data = {}
        for comp in self._working_trace.components:
            data[comp.id] = {
                "id": comp.id,
                "type": comp.type.value,
                "content": comp.content,
                "token_count": comp.token_count,
                "metadata": comp.metadata,
            }

        # Also embed resource data
        resources_data = {}
        for resource in self._resources:
            resources_data[resource.name] = {
                "name": resource.name,
                "type": resource.resource_type.value,
                "selected_ids": list(resource.selected_ids),
                "items": [
                    {
                        "id": item.id,
                        "content": item.content,
                        "token_count": item.token_count,
                        "score": item.score,
                    }
                    for item in resource.items
                ],
            }

        return (
            f"<script>var cedData_{uid} = {json.dumps(data, ensure_ascii=False)};\n"
            f"var cedResources_{uid} = {json.dumps(resources_data, ensure_ascii=False)};</script>"
        )

    # ---------------------------------------------------------- JavaScript
    def _js(self, uid: str) -> str:
        return f"""
(function() {{
  var container = document.getElementById('ced-{uid}');
  if (!container) return;
  var tooltip = document.getElementById('ced-tooltip-{uid}');
  var modal = document.getElementById('ced-modal-{uid}');
  var saveBtn = document.getElementById('ced-modal-save-{uid}');
  var data = typeof cedData_{uid} !== 'undefined' ? cedData_{uid} : {{}};
  var currentInfo = null;

  // Color map
  var colorMap = {json.dumps({ct.value: COMPONENT_COLORS[ct] for ct in ComponentType})};
  var textColorMap = {json.dumps({ct.value: TEXT_COLORS[ct] for ct in ComponentType})};
  var labelMap = {json.dumps({ct.value: COMPONENT_LABELS[ct] for ct in ComponentType})};
  var iconMap = {json.dumps({ct.value: COMPONENT_ICONS[ct] for ct in ComponentType})};

  // Get all component elements
  var components = container.querySelectorAll('.ced-component, .ced-treemap-item');

  // Layout toggle
  window.cedToggleLayout_{uid} = function(layout) {{
    var h = document.getElementById('ced-hlayout-{uid}');
    var t = document.getElementById('ced-tlayout-{uid}');
    if (h) h.style.display = layout === 'horizontal' ? '' : 'none';
    if (t) t.style.display = layout === 'treemap' ? '' : 'none';
  }};

  // Close modal
  window.cedCloseModal_{uid} = function() {{
    if (modal) {{
      modal.style.display = 'none';
      saveBtn.style.display = 'none';
      currentInfo = null;
    }}
  }};

  // Show modal in view mode (click text to edit)
  function showModal(info) {{
    currentInfo = info;
    var header = document.getElementById('ced-modal-header-{uid}');
    var title = document.getElementById('ced-modal-title-{uid}');
    var body = document.getElementById('ced-modal-body-{uid}');
    var bg = colorMap[info.type] || '#999';
    var fg = textColorMap[info.type] || '#000';
    header.style.background = bg;
    header.style.color = fg;
    var icon = iconMap[info.type] || '';
    var label = labelMap[info.type] || info.type;
    title.textContent = icon + ' ' + label + ' — ' + info.id;
    saveBtn.style.display = 'none';

    // Clickable read-only content
    var contentHtml = '<div class="ced-modal-section">' +
      '<div class="ced-modal-section-title">Content</div>' +
      '<div class="ced-modal-text" id="ced-content-text-{uid}">' +
      escapeHtml(info.content) + '</div>' +
      '<div class="ced-modal-text-hint">Click text to edit</div></div>';

    // Token count
    contentHtml += '<div class="ced-modal-section">' +
      '<div class="ced-modal-section-title">Tokens</div>' +
      '<div>' + info.token_count.toLocaleString() + '</div></div>';

    // Metadata table
    var meta = info.metadata || {{}};
    var metaKeys = Object.keys(meta);
    if (metaKeys.length > 0) {{
      contentHtml += '<div class="ced-modal-section">' +
        '<div class="ced-modal-section-title">Metadata</div>' +
        '<table class="ced-metadata-table"><tr><th>Key</th><th>Value</th></tr>';
      metaKeys.forEach(function(k) {{
        contentHtml += '<tr><td>' + escapeHtml(k) + '</td><td>' +
          escapeHtml(String(meta[k])) + '</td></tr>';
      }});
      contentHtml += '</table></div>';
    }}

    body.innerHTML = contentHtml;
    modal.style.display = 'flex';

    // Add click-to-edit on content text
    var contentText = document.getElementById('ced-content-text-{uid}');
    if (contentText) {{
      contentText.addEventListener('click', function() {{
        switchToEditMode(info);
      }});
    }}
  }}

  // Switch to edit mode
  function switchToEditMode(info) {{
    var title = document.getElementById('ced-modal-title-{uid}');
    var icon = iconMap[info.type] || '';
    var label = labelMap[info.type] || info.type;
    title.textContent = icon + ' EDIT: ' + label + ' — ' + info.id;

    // Replace text with textarea
    var contentSection = document.querySelector('#ced-{uid} .ced-modal-section');
    if (contentSection) {{
      contentSection.innerHTML =
        '<div class="ced-modal-section-title">Content</div>' +
        '<textarea class="ced-modal-textarea" id="ced-edit-textarea-{uid}">' +
        escapeHtml(info.content) + '</textarea>';
    }}

    // Show save button
    saveBtn.style.display = 'block';

    // Focus textarea
    var textarea = document.getElementById('ced-edit-textarea-{uid}');
    if (textarea) textarea.focus();
  }}

  // Recalculate treemap positions when unused is collapsed
  function recalcTreemap(collapsed) {{
    var treemap = container.querySelector('.ced-treemap');
    if (!treemap) return;

    var items = treemap.querySelectorAll('.ced-treemap-item');
    var unusedEl = treemap.querySelector('[data-comp-id="_unused"]');
    if (!unusedEl) return;

    if (collapsed) {{
      // Calculate total tokens for used components
      var totalUsedTokens = 0;
      items.forEach(function(el) {{
        if (el.getAttribute('data-comp-id') !== '_unused') {{
          totalUsedTokens += parseInt(el.getAttribute('data-tokens') || 0);
        }}
      }});

      // Collapse unused to thin strip on right
      var th = treemap.clientHeight - 32; // padding
      unusedEl.style.height = th + 'px';
      unusedEl.style.top = '16px';

      // Scale other items to fill ~97% of width (leaving 3% for collapsed unused)
      var scaleFactor = 97 / 100;
      items.forEach(function(el) {{
        if (el.getAttribute('data-comp-id') !== '_unused') {{
          var origW = parseFloat(el.getAttribute('data-orig-w'));
          var origX = parseFloat(el.getAttribute('data-orig-x'));
          el.style.width = (origW * scaleFactor) + '%';
          el.style.left = (origX * scaleFactor) + '%';
        }}
      }});
    }} else {{
      // Restore original positions
      items.forEach(function(el) {{
        el.style.width = el.getAttribute('data-orig-w') + '%';
        el.style.left = el.getAttribute('data-orig-x') + '%';
        el.style.height = el.getAttribute('data-orig-h') + 'px';
        el.style.top = el.getAttribute('data-orig-y') + 'px';
        el.style.right = '';
      }});
    }}
  }}

  // Drag-and-drop state
  var dragState = {{
    isDragging: false,
    draggedEl: null,
    draggedId: null,
    startX: 0,
    startY: 0,
    startTime: 0,
    currentDropTarget: null,
    dropPosition: null
  }};
  var DRAG_THRESHOLD_PX = 5;
  var DRAG_THRESHOLD_MS = 150;

  function handleDragStart(el, e) {{
    if (el.getAttribute('data-comp-id') === '_unused') return;
    if (el.classList.contains('ced-treemap-item')) return; // Treemap not draggable

    dragState.startX = e.clientX;
    dragState.startY = e.clientY;
    dragState.startTime = Date.now();
    dragState.draggedEl = el;
    dragState.draggedId = el.getAttribute('data-comp-id');

    document.addEventListener('mousemove', handleDragMove);
    document.addEventListener('mouseup', handleDragEnd);
  }}

  function handleDragMove(e) {{
    if (!dragState.draggedEl) return;

    var dx = e.clientX - dragState.startX;
    var dy = e.clientY - dragState.startY;
    var distance = Math.sqrt(dx * dx + dy * dy);
    var elapsed = Date.now() - dragState.startTime;

    if (!dragState.isDragging && (distance > DRAG_THRESHOLD_PX || elapsed > DRAG_THRESHOLD_MS)) {{
      enterDragMode();
    }}

    if (dragState.isDragging) {{
      updateDropTarget(e);
    }}
  }}

  function enterDragMode() {{
    dragState.isDragging = true;
    dragState.draggedEl.classList.add('ced-dragging');

    var horizontal = container.querySelector('.ced-horizontal');
    if (horizontal) horizontal.classList.add('ced-dragging-active');

    tooltip.style.display = 'none';
  }}

  function updateDropTarget(e) {{
    var horizontal = container.querySelector('.ced-horizontal');
    if (!horizontal) return;

    clearDropIndicators();

    var comps = horizontal.querySelectorAll('.ced-component:not(.ced-comp-unused):not(.ced-dragging)');

    for (var i = 0; i < comps.length; i++) {{
      var comp = comps[i];
      var rect = comp.getBoundingClientRect();

      if (e.clientX >= rect.left && e.clientX <= rect.right &&
          e.clientY >= rect.top && e.clientY <= rect.bottom) {{

        var midX = rect.left + rect.width / 2;
        var position = e.clientX < midX ? 'before' : 'after';

        dragState.currentDropTarget = comp;
        dragState.dropPosition = position;

        comp.classList.add(position === 'before' ? 'ced-drop-left' : 'ced-drop-right');
        break;
      }}
    }}
  }}

  function clearDropIndicators() {{
    var indicators = container.querySelectorAll('.ced-drop-left, .ced-drop-right');
    indicators.forEach(function(el) {{
      el.classList.remove('ced-drop-left', 'ced-drop-right');
    }});
    dragState.currentDropTarget = null;
    dragState.dropPosition = null;
  }}

  function handleDragEnd(e) {{
    document.removeEventListener('mousemove', handleDragMove);
    document.removeEventListener('mouseup', handleDragEnd);

    if (dragState.isDragging) {{
      if (dragState.currentDropTarget && dragState.draggedEl) {{
        performReorder();
      }}
      exitDragMode();
    }} else {{
      dragState.draggedEl = null;
      dragState.draggedId = null;
    }}
  }}

  function performReorder() {{
    var target = dragState.currentDropTarget;
    var dragged = dragState.draggedEl;
    var position = dragState.dropPosition;

    if (!target || !dragged || target === dragged) return;

    var horizontal = container.querySelector('.ced-horizontal');
    if (!horizontal) return;

    if (position === 'before') {{
      horizontal.insertBefore(dragged, target);
    }} else {{
      var next = target.nextElementSibling;
      if (next) {{
        horizontal.insertBefore(dragged, next);
      }} else {{
        horizontal.appendChild(dragged);
      }}
    }}

    updateComponentOrder();
  }}

  function exitDragMode() {{
    if (dragState.draggedEl) {{
      dragState.draggedEl.classList.remove('ced-dragging');
    }}

    var horizontal = container.querySelector('.ced-horizontal');
    if (horizontal) horizontal.classList.remove('ced-dragging-active');

    clearDropIndicators();

    dragState.isDragging = false;
    dragState.draggedEl = null;
    dragState.draggedId = null;
    dragState.startX = 0;
    dragState.startY = 0;
    dragState.startTime = 0;
  }}

  function updateComponentOrder() {{
    var horizontal = container.querySelector('.ced-horizontal');
    if (!horizontal) return;

    var newOrder = [];
    var comps = horizontal.querySelectorAll('.ced-component:not(.ced-comp-unused)');
    comps.forEach(function(el) {{
      var id = el.getAttribute('data-comp-id');
      if (id && id !== '_unused') {{
        newOrder.push(id);
      }}
    }});

    container.setAttribute('data-component-order', JSON.stringify(newOrder));

    var event = new CustomEvent('ced-reorder', {{
      detail: {{ order: newOrder, uid: '{uid}' }}
    }});
    container.dispatchEvent(event);
  }}

  // Save button handler - persist edits to data attributes
  if (saveBtn) {{
    saveBtn.addEventListener('click', function() {{
      var textarea = document.getElementById('ced-edit-textarea-{uid}');
      if (textarea && currentInfo) {{
        var edits = JSON.parse(container.getAttribute('data-edits') || '{{}}');
        edits[currentInfo.id] = {{
          original: currentInfo.content,
          edited: textarea.value,
          timestamp: new Date().toISOString()
        }};
        container.setAttribute('data-edits', JSON.stringify(edits));
        currentInfo.content = textarea.value;
        data[currentInfo.id].content = textarea.value;
        container.setAttribute('data-has-changes', 'true');
      }}
      cedCloseModal_{uid}();
    }});
  }}

  // State retrieval function for Python sync
  window.cedGetState_{uid} = function() {{
    return {{
      edits: JSON.parse(container.getAttribute('data-edits') || '{{}}'),
      componentOrder: JSON.parse(container.getAttribute('data-component-order') || '[]'),
      selections: JSON.parse(container.getAttribute('data-selections') || '{{}}'),
      hasChanges: container.getAttribute('data-has-changes') === 'true'
    }};
  }};

  // Cross-panel drag-and-drop for resources
  var availablePanel = document.getElementById('ced-available-{uid}');
  var contextPanel = document.getElementById('ced-context-{uid}');
  var saveSelectionsBtn = document.getElementById('ced-save-btn-{uid}');
  var pendingSelections = {{}};

  function initCrossPanelDrag() {{
    if (!availablePanel || !contextPanel) return;

    // Get all draggable items
    var draggableItems = container.querySelectorAll('.ced-doc-item[draggable="true"]');

    draggableItems.forEach(function(item) {{
      item.addEventListener('dragstart', function(e) {{
        e.dataTransfer.setData('text/plain', JSON.stringify({{
          itemId: item.getAttribute('data-item-id'),
          resource: item.getAttribute('data-resource'),
          fromContext: item.closest('.ced-context-panel') !== null
        }}));
        e.dataTransfer.effectAllowed = 'move';
        item.classList.add('ced-dragging');
      }});

      item.addEventListener('dragend', function(e) {{
        item.classList.remove('ced-dragging');
        availablePanel.classList.remove('ced-drop-target');
        contextPanel.classList.remove('ced-drop-target');
      }});
    }});

    // Context panel accepts drops from Available
    contextPanel.addEventListener('dragover', function(e) {{
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      contextPanel.classList.add('ced-drop-target');
    }});

    contextPanel.addEventListener('dragleave', function(e) {{
      contextPanel.classList.remove('ced-drop-target');
    }});

    contextPanel.addEventListener('drop', function(e) {{
      e.preventDefault();
      contextPanel.classList.remove('ced-drop-target');
      try {{
        var data = JSON.parse(e.dataTransfer.getData('text/plain'));
        if (!data.fromContext && data.itemId && data.resource) {{
          selectItem(data.resource, data.itemId);
        }}
      }} catch (err) {{}}
    }});

    // Available panel accepts drops from Context
    availablePanel.addEventListener('dragover', function(e) {{
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      availablePanel.classList.add('ced-drop-target');
    }});

    availablePanel.addEventListener('dragleave', function(e) {{
      availablePanel.classList.remove('ced-drop-target');
    }});

    availablePanel.addEventListener('drop', function(e) {{
      e.preventDefault();
      availablePanel.classList.remove('ced-drop-target');
      try {{
        var data = JSON.parse(e.dataTransfer.getData('text/plain'));
        if (data.fromContext && data.itemId && data.resource) {{
          deselectItem(data.resource, data.itemId);
        }}
      }} catch (err) {{}}
    }});
  }}

  function selectItem(resourceName, itemId) {{
    // Initialize resource selections if needed
    if (!pendingSelections[resourceName]) {{
      pendingSelections[resourceName] = [];
      // Get currently selected items
      var selectedInContext = contextPanel.querySelectorAll(
        '.ced-doc-item[data-resource="' + resourceName + '"]'
      );
      selectedInContext.forEach(function(el) {{
        pendingSelections[resourceName].push(el.getAttribute('data-item-id'));
      }});
    }}

    // Add to selections
    if (pendingSelections[resourceName].indexOf(itemId) === -1) {{
      pendingSelections[resourceName].push(itemId);
    }}

    // Update UI
    updateSelectionUI(resourceName, itemId, true);
    showSaveButton();
  }}

  function deselectItem(resourceName, itemId) {{
    if (!pendingSelections[resourceName]) {{
      pendingSelections[resourceName] = [];
      var selectedInContext = contextPanel.querySelectorAll(
        '.ced-doc-item[data-resource="' + resourceName + '"]'
      );
      selectedInContext.forEach(function(el) {{
        pendingSelections[resourceName].push(el.getAttribute('data-item-id'));
      }});
    }}

    // Remove from selections
    var idx = pendingSelections[resourceName].indexOf(itemId);
    if (idx !== -1) {{
      pendingSelections[resourceName].splice(idx, 1);
    }}

    // Update UI
    updateSelectionUI(resourceName, itemId, false);
    showSaveButton();
  }}

  function updateSelectionUI(resourceName, itemId, selected) {{
    // Update item in Available panel
    var availableItem = availablePanel.querySelector(
      '.ced-doc-item[data-item-id="' + itemId + '"][data-resource="' + resourceName + '"]'
    );
    if (availableItem) {{
      if (selected) {{
        availableItem.classList.add('ced-selected');
        availableItem.classList.remove('ced-unselected');
        availableItem.querySelector('.ced-doc-check').textContent = '\\u2713';
      }} else {{
        availableItem.classList.remove('ced-selected');
        availableItem.classList.add('ced-unselected');
        availableItem.querySelector('.ced-doc-check').textContent = '';
      }}
    }}

    // Add/remove from Context panel
    var contextContent = document.getElementById('ced-context-content-{uid}');
    var contextItem = contextPanel.querySelector(
      '.ced-doc-item[data-item-id="' + itemId + '"][data-resource="' + resourceName + '"]'
    );

    if (selected && !contextItem && availableItem) {{
      // Clone and add to context
      var clone = availableItem.cloneNode(true);
      clone.classList.remove('ced-unselected');
      clone.classList.add('ced-selected');
      clone.style.background = '#00AA55';
      clone.style.color = 'white';
      clone.querySelector('.ced-doc-check').textContent = '';
      contextContent.appendChild(clone);
      // Re-init drag on the new element
      initDragOnElement(clone);
    }} else if (!selected && contextItem) {{
      contextItem.remove();
    }}
  }}

  function initDragOnElement(item) {{
    item.addEventListener('dragstart', function(e) {{
      e.dataTransfer.setData('text/plain', JSON.stringify({{
        itemId: item.getAttribute('data-item-id'),
        resource: item.getAttribute('data-resource'),
        fromContext: item.closest('.ced-context-panel') !== null
      }}));
      e.dataTransfer.effectAllowed = 'move';
      item.classList.add('ced-dragging');
    }});

    item.addEventListener('dragend', function(e) {{
      item.classList.remove('ced-dragging');
      availablePanel.classList.remove('ced-drop-target');
      contextPanel.classList.remove('ced-drop-target');
    }});
  }}

  function showSaveButton() {{
    if (saveSelectionsBtn) {{
      saveSelectionsBtn.style.display = 'block';
      container.setAttribute('data-selections', JSON.stringify(pendingSelections));
      container.setAttribute('data-has-changes', 'true');
    }}
  }}

  // Save selections button handler
  if (saveSelectionsBtn) {{
    saveSelectionsBtn.addEventListener('click', function() {{
      container.setAttribute('data-selections', JSON.stringify(pendingSelections));
      saveSelectionsBtn.style.display = 'none';

      // Dispatch event for external listeners
      var event = new CustomEvent('ced-selections-saved', {{
        detail: {{ selections: pendingSelections, uid: '{uid}' }}
      }});
      container.dispatchEvent(event);
    }});
  }}

  // Initialize cross-panel drag
  initCrossPanelDrag();

  // Event handlers for each component
  components.forEach(function(el) {{
    // Hover → Tooltip
    el.addEventListener('mouseenter', function(e) {{
      var compId = el.getAttribute('data-comp-id');
      var info = data[compId];
      var text;
      if (compId === '_unused') {{
        text = el.classList.contains('ced-collapsed')
          ? 'CLICK TO EXPAND'
          : 'UNUSED — CLICK TO COLLAPSE';
      }} else {{
        text = info ? info.type.toUpperCase().replace('_', ' ') + ' — ' +
          info.token_count.toLocaleString() + ' TOKENS' : compId;
      }}
      tooltip.textContent = text;
      tooltip.style.display = 'block';
      var rect = el.getBoundingClientRect();
      var cRect = container.getBoundingClientRect();
      tooltip.style.left = (rect.left - cRect.left + rect.width / 2 -
        tooltip.offsetWidth / 2) + 'px';
      tooltip.style.top = (rect.top - cRect.top - tooltip.offsetHeight - 8) + 'px';
    }});
    el.addEventListener('mouseleave', function() {{
      tooltip.style.display = 'none';
    }});

    // Click → Toggle unused collapse OR open modal (skip if dragging)
    el.addEventListener('click', function() {{
      if (dragState.isDragging) return;

      var compId = el.getAttribute('data-comp-id');
      if (compId === '_unused') {{
        el.classList.toggle('ced-collapsed');
        recalcTreemap(el.classList.contains('ced-collapsed'));
        return;
      }}
      var info = data[compId];
      if (!info) return;
      showModal(info);
    }});

    // Mousedown → Start potential drag (horizontal layout only)
    el.addEventListener('mousedown', function(e) {{
      if (e.button !== 0) return;
      handleDragStart(el, e);
    }});
  }});

  // Close modal on overlay click
  if (modal) {{
    modal.addEventListener('click', function(e) {{
      if (e.target === modal) cedCloseModal_{uid}();
    }});
  }}

  function escapeHtml(str) {{
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }}
}})();
"""


# Backward compatibility alias
ContextWindow = ContextBuilder
