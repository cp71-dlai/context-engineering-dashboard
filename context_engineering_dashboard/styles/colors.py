"""Neo-brutalist design system colors and constants."""

from context_engineering_dashboard.core.trace import ComponentType

COMPONENT_COLORS = {
    ComponentType.SYSTEM_PROMPT: "#FF6B00",
    ComponentType.USER_MESSAGE: "#0066FF",
    ComponentType.CHAT_HISTORY: "#00BFFF",
    ComponentType.RAG_DOCUMENT: "#00AA55",
    ComponentType.TOOL: "#FFCC00",
    ComponentType.FEW_SHOT: "#AA44FF",
    ComponentType.MEMORY: "#00CCAA",
}

TEXT_COLORS = {
    ComponentType.SYSTEM_PROMPT: "#000000",
    ComponentType.USER_MESSAGE: "#FFFFFF",
    ComponentType.CHAT_HISTORY: "#000000",
    ComponentType.RAG_DOCUMENT: "#FFFFFF",
    ComponentType.TOOL: "#000000",
    ComponentType.FEW_SHOT: "#FFFFFF",
    ComponentType.MEMORY: "#000000",
}

CSS_CLASSES = {
    ComponentType.SYSTEM_PROMPT: "ced-comp-system",
    ComponentType.USER_MESSAGE: "ced-comp-user",
    ComponentType.CHAT_HISTORY: "ced-comp-history",
    ComponentType.RAG_DOCUMENT: "ced-comp-rag",
    ComponentType.TOOL: "ced-comp-tool",
    ComponentType.FEW_SHOT: "ced-comp-fewshot",
    ComponentType.MEMORY: "ced-comp-memory",
}

COMPONENT_ICONS = {
    ComponentType.SYSTEM_PROMPT: "\u2699",  # gear
    ComponentType.USER_MESSAGE: "\U0001f4ac",  # speech bubble
    ComponentType.CHAT_HISTORY: "\U0001f550",  # clock
    ComponentType.RAG_DOCUMENT: "\U0001f4c4",  # page
    ComponentType.TOOL: "\U0001f527",  # wrench
    ComponentType.FEW_SHOT: "\U0001f4cb",  # clipboard
    ComponentType.MEMORY: "\U0001f9e0",  # brain
}

COMPONENT_LABELS = {
    ComponentType.SYSTEM_PROMPT: "System",
    ComponentType.USER_MESSAGE: "User",
    ComponentType.CHAT_HISTORY: "History",
    ComponentType.RAG_DOCUMENT: "RAG Doc",
    ComponentType.TOOL: "Tool",
    ComponentType.FEW_SHOT: "Few-Shot",
    ComponentType.MEMORY: "Memory",
}

UNUSED_COLOR = "#E0E0E0"
UNUSED_TEXT_COLOR = "#888888"
