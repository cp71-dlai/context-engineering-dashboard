"""ContextWindow — main Jupyter widget for visualizing LLM context windows."""

import html
import json
import uuid
from typing import Optional

from context_engineering_dashboard.core.trace import ComponentType, ContextTrace
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


class ContextWindow:
    """Renders a neo-brutalist visualization of a context trace in Jupyter.

    Parameters
    ----------
    trace : ContextTrace
        The trace data to visualize.
    context_limit : int, optional
        Override the trace's context_limit for display.
    layout : str
        Layout algorithm: "horizontal" or "treemap".
    show_available_pool : bool
        Show Chroma available pool two-panel view.
    show_patterns : bool
        Show pattern analysis overlay.

    Interactions
    ------------
    - Hover: Tooltip with component type and token count
    - Click: Modal with full content and metadata
    - Click on text in modal: Switch to edit mode with Save button
    """

    def __init__(
        self,
        trace: ContextTrace,
        context_limit: Optional[int] = None,
        layout: str = "horizontal",
        show_available_pool: bool = False,
        show_patterns: bool = False,
    ) -> None:
        self.trace = trace
        self.context_limit = context_limit or trace.context_limit
        self.layout = layout.lower()
        self.show_available_pool = show_available_pool
        self.show_patterns = show_patterns
        self._uid = uuid.uuid4().hex[:12]

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

        if self.show_available_pool and self.trace.chroma_queries:
            parts.append(self._available_pool_html(uid))
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
  background: {COMPONENT_COLORS[ComponentType.RAG_DOCUMENT]};
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
"""

    # -------------------------------------------------------------- Header
    def _header_html(self, uid: str) -> str:
        return (
            f'<div class="ced-header">'
            f'<span class="ced-title">Context Window</span>'
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
            '<span class="ced-window-label">Context Window</span>',
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
                f'style="left:{r.x}%;top:{r.y}px;width:{r.width}%;height:{r.height}px;">'
                f"{score_badge}"
                f'<span class="ced-icon">{icon}</span>'
                f'<span class="ced-label">{html.escape(label)}</span>'
                f'<span class="ced-tokens">{r.token_count:,}</span>'
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

    # -------------------------------------------------------- Available pool
    def _available_pool_html(self, uid: str) -> str:
        """Render two-panel available pool view."""
        if not self.trace.chroma_queries:
            return self._context_window_html(uid)

        query = self.trace.chroma_queries[0]
        n = query.n_results
        sorted_results = sorted(query.results, key=lambda r: r.score, reverse=True)

        # Left panel: all results
        left_items = []
        for r in sorted_results:
            sel_cls = "ced-selected" if r.selected else "ced-unselected"
            check = "\u2713" if r.selected else "\u2717"
            left_items.append(
                f'<div class="ced-doc-item {sel_cls}">'
                f'<span class="ced-doc-score">{r.score:.2f}</span>'
                f"<span>{html.escape(r.id)}</span>"
                f'<span class="ced-doc-tokens">{r.token_count:,} tok</span>'
                f'<span class="ced-doc-check">{check}</span>'
                f"</div>"
            )

        # Right panel: selected docs + other components
        right_items = []
        for r in sorted_results:
            if r.selected:
                right_items.append(
                    f'<div class="ced-doc-item ced-selected">'
                    f'<span class="ced-doc-score">{r.score:.2f}</span>'
                    f"<span>{html.escape(r.id)}</span>"
                    f'<span class="ced-doc-tokens">{r.token_count:,} tok</span>'
                    f"</div>"
                )
        # Other components
        for comp in self.trace.components:
            if comp.type != ComponentType.RAG_DOCUMENT:
                css_cls = CSS_CLASSES.get(comp.type, "")
                icon = COMPONENT_ICONS.get(comp.type, "")
                label = COMPONENT_LABELS.get(comp.type, "")
                right_items.append(
                    f'<div class="ced-doc-item {css_cls}" '
                    f'style="border: 2px solid black;">'
                    f"<span>{icon} {html.escape(label)}</span>"
                    f'<span class="ced-doc-tokens">{comp.token_count:,} tok</span>'
                    f"</div>"
                )

        return (
            f'<div class="ced-two-panel">'
            f'<div class="ced-panel">'
            f'<div class="ced-panel-header">Available (Chroma Query: n={n})</div>'
            f'<div class="ced-panel-content">{"".join(left_items)}</div>'
            f"</div>"
            f'<div class="ced-scissors">\u2702\ufe0f</div>'
            f'<div class="ced-panel">'
            f'<div class="ced-panel-header">Context Window</div>'
            f'<div class="ced-panel-content">{"".join(right_items)}</div>'
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
        for comp in self.trace.components:
            data[comp.id] = {
                "id": comp.id,
                "type": comp.type.value,
                "content": comp.content,
                "token_count": comp.token_count,
                "metadata": comp.metadata,
            }
        return f"<script>var cedData_{uid} = {json.dumps(data, ensure_ascii=False)};</script>"

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

  // Save button handler
  if (saveBtn) {{
    saveBtn.addEventListener('click', function() {{
      var textarea = document.getElementById('ced-edit-textarea-{uid}');
      if (textarea && currentInfo) {{
        currentInfo.content = textarea.value;
      }}
      cedCloseModal_{uid}();
    }});
  }}

  // Event handlers for each component
  components.forEach(function(el) {{
    // Hover → Tooltip
    el.addEventListener('mouseenter', function(e) {{
      var compId = el.getAttribute('data-comp-id');
      var info = data[compId];
      var text = compId === '_unused' ? 'UNUSED' :
        (info ? info.type.toUpperCase().replace('_', ' ') + ' — ' +
        info.token_count.toLocaleString() + ' TOKENS' : compId);
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

    // Click → Modal (view mode, click text to edit)
    el.addEventListener('click', function() {{
      var compId = el.getAttribute('data-comp-id');
      if (compId === '_unused') return;
      var info = data[compId];
      if (!info) return;
      showModal(info);
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
