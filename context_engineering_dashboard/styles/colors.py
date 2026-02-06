"""Neo-brutalist design system colors and constants."""

from context_engineering_dashboard.core.trace import ComponentType

COMPONENT_COLORS = {
    ComponentType.SYSTEM_PROMPT: "#FF6B00",
    ComponentType.USER_MESSAGE: "#0066FF",
    ComponentType.CHAT_HISTORY: "#00BFFF",
    ComponentType.RAG: "#00AA55",
    ComponentType.TOOL: "#FFCC00",
    ComponentType.EXAMPLE: "#AA44FF",
    ComponentType.SCRATCHPAD: "#00CCAA",
}

TEXT_COLORS = {
    ComponentType.SYSTEM_PROMPT: "#000000",
    ComponentType.USER_MESSAGE: "#FFFFFF",
    ComponentType.CHAT_HISTORY: "#000000",
    ComponentType.RAG: "#FFFFFF",
    ComponentType.TOOL: "#000000",
    ComponentType.EXAMPLE: "#FFFFFF",
    ComponentType.SCRATCHPAD: "#000000",
}

CSS_CLASSES = {
    ComponentType.SYSTEM_PROMPT: "ced-comp-system",
    ComponentType.USER_MESSAGE: "ced-comp-user",
    ComponentType.CHAT_HISTORY: "ced-comp-history",
    ComponentType.RAG: "ced-comp-rag",
    ComponentType.TOOL: "ced-comp-tool",
    ComponentType.EXAMPLE: "ced-comp-example",
    ComponentType.SCRATCHPAD: "ced-comp-scratchpad",
}

COMPONENT_ICONS = {
    ComponentType.SYSTEM_PROMPT: "\u2699",  # gear
    ComponentType.USER_MESSAGE: "\U0001f4ac",  # speech bubble
    ComponentType.CHAT_HISTORY: "\U0001f550",  # clock
    ComponentType.RAG: "\U0001f4c4",  # page
    ComponentType.TOOL: "\U0001f527",  # wrench
    ComponentType.EXAMPLE: "\U0001f4cb",  # clipboard
    ComponentType.SCRATCHPAD: "\U0001f9e0",  # brain
}

COMPONENT_LABELS = {
    ComponentType.SYSTEM_PROMPT: "System",
    ComponentType.USER_MESSAGE: "User",
    ComponentType.CHAT_HISTORY: "History",
    ComponentType.RAG: "RAG",
    ComponentType.TOOL: "Tool",
    ComponentType.EXAMPLE: "Example",
    ComponentType.SCRATCHPAD: "Scratchpad",
}

UNUSED_COLOR = "#E0E0E0"
UNUSED_TEXT_COLOR = "#888888"
