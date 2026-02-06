"""Tests for ContextResource and related classes."""

from context_engineering_dashboard.core.resource import (
    ContextResource,
    ResourceItem,
    ResourceType,
)
from context_engineering_dashboard.core.trace import ComponentType


def test_resource_type_values():
    """ResourceType enum has expected values."""
    assert ResourceType.RAG.value == "rag"
    assert ResourceType.EXAMPLE.value == "example"
    assert ResourceType.CHAT_HISTORY.value == "chat_history"
    assert ResourceType.SCRATCHPAD.value == "scratchpad"
    assert ResourceType.TOOL.value == "tool"


def test_resource_type_to_component_type():
    """ResourceType maps to correct ComponentType."""
    assert ResourceType.RAG.to_component_type() == ComponentType.RAG
    assert ResourceType.EXAMPLE.to_component_type() == ComponentType.EXAMPLE
    assert ResourceType.CHAT_HISTORY.to_component_type() == ComponentType.CHAT_HISTORY
    assert ResourceType.SCRATCHPAD.to_component_type() == ComponentType.SCRATCHPAD
    assert ResourceType.TOOL.to_component_type() == ComponentType.TOOL


def test_resource_item_creation():
    """ResourceItem can be created with all fields."""
    item = ResourceItem(
        id="doc_1",
        content="Hello world",
        token_count=2,
        score=0.95,
        metadata={"source": "test.md"},
    )
    assert item.id == "doc_1"
    assert item.content == "Hello world"
    assert item.token_count == 2
    assert item.score == 0.95
    assert item.metadata == {"source": "test.md"}


def test_resource_item_defaults():
    """ResourceItem uses sensible defaults."""
    item = ResourceItem(id="doc_1", content="text", token_count=1)
    assert item.score is None
    assert item.metadata == {}


def test_context_resource_creation():
    """ContextResource can be created with items."""
    items = [
        ResourceItem(id="d1", content="Doc 1", token_count=10),
        ResourceItem(id="d2", content="Doc 2", token_count=20),
    ]
    resource = ContextResource(
        name="Test Docs",
        resource_type=ResourceType.RAG,
        items=items,
    )
    assert resource.name == "Test Docs"
    assert resource.resource_type == ResourceType.RAG
    assert len(resource.items) == 2


def test_context_resource_select_deselect():
    """ContextResource select/deselect operations work."""
    items = [
        ResourceItem(id="d1", content="Doc 1", token_count=10),
        ResourceItem(id="d2", content="Doc 2", token_count=20),
        ResourceItem(id="d3", content="Doc 3", token_count=30),
    ]
    resource = ContextResource(
        name="Docs",
        resource_type=ResourceType.RAG,
        items=items,
    )

    # Initially empty
    assert len(resource.selected_ids) == 0

    # Select one
    resource.select(["d1"])
    assert "d1" in resource.selected_ids
    assert len(resource.selected_ids) == 1

    # Select another
    resource.select(["d2"])
    assert "d1" in resource.selected_ids
    assert "d2" in resource.selected_ids
    assert len(resource.selected_ids) == 2

    # Deselect one
    resource.deselect(["d1"])
    assert "d1" not in resource.selected_ids
    assert "d2" in resource.selected_ids
    assert len(resource.selected_ids) == 1


def test_context_resource_select_all_clear():
    """ContextResource select_all and clear_selection work."""
    items = [
        ResourceItem(id="d1", content="Doc 1", token_count=10),
        ResourceItem(id="d2", content="Doc 2", token_count=20),
    ]
    resource = ContextResource(
        name="Docs",
        resource_type=ResourceType.RAG,
        items=items,
    )

    # Select all
    resource.select_all()
    assert len(resource.selected_ids) == 2
    assert "d1" in resource.selected_ids
    assert "d2" in resource.selected_ids

    # Clear selection
    resource.clear_selection()
    assert len(resource.selected_ids) == 0


def test_context_resource_to_components():
    """ContextResource.to_components() returns correct components."""
    items = [
        ResourceItem(id="d1", content="Doc 1", token_count=10, score=0.9),
        ResourceItem(id="d2", content="Doc 2", token_count=20, score=0.8),
        ResourceItem(id="d3", content="Doc 3", token_count=30, score=0.7),
    ]
    resource = ContextResource(
        name="Docs",
        resource_type=ResourceType.RAG,
        items=items,
    )

    # Select only d1 and d3
    resource.select(["d1", "d3"])

    components = resource.to_components()
    assert len(components) == 2

    # Check component properties
    comp_ids = [c.id for c in components]
    assert "d1" in comp_ids
    assert "d3" in comp_ids

    # Check type matches
    for comp in components:
        assert comp.type == ComponentType.RAG


def test_context_resource_from_items():
    """ContextResource.from_items factory method works."""
    item_dicts = [
        {"id": "d1", "content": "Doc 1"},
        {"id": "d2", "content": "Doc 2", "token_count": 10, "score": 0.9},
    ]
    resource = ContextResource.from_items(
        items=item_dicts,
        resource_type=ResourceType.EXAMPLE,
        name="Examples",
    )

    assert resource.name == "Examples"
    assert resource.resource_type == ResourceType.EXAMPLE
    assert len(resource.items) == 2
    assert resource.items[0].id == "d1"
    assert resource.items[1].score == 0.9


def test_context_resource_selected_items_property():
    """ContextResource.selected_items returns only selected items."""
    items = [
        ResourceItem(id="d1", content="Doc 1", token_count=10),
        ResourceItem(id="d2", content="Doc 2", token_count=20),
        ResourceItem(id="d3", content="Doc 3", token_count=30),
    ]
    resource = ContextResource(
        name="Docs",
        resource_type=ResourceType.RAG,
        items=items,
    )

    resource.select(["d2"])
    selected = resource.selected_items

    assert len(selected) == 1
    assert selected[0].id == "d2"


def test_context_resource_total_selected_tokens():
    """ContextResource.total_selected_tokens is correct."""
    items = [
        ResourceItem(id="d1", content="Doc 1", token_count=10),
        ResourceItem(id="d2", content="Doc 2", token_count=20),
        ResourceItem(id="d3", content="Doc 3", token_count=30),
    ]
    resource = ContextResource(
        name="Docs",
        resource_type=ResourceType.RAG,
        items=items,
    )

    resource.select(["d1", "d3"])
    assert resource.total_selected_tokens == 40
