"""Base tracer abstract class for provider tracers."""

from abc import ABC, abstractmethod
from typing import Optional

from context_engineering_dashboard.core.trace import ContextTrace


class BaseTracer(ABC):
    """Abstract base class for provider tracers.

    Subclasses implement __enter__/__exit__ for use as context managers.
    After exiting, access the captured trace via the `result` property.
    """

    def __init__(self, context_limit: Optional[int] = None, **kwargs: object) -> None:
        self._context_limit = context_limit
        self._trace: Optional[ContextTrace] = None

    @property
    def result(self) -> Optional[ContextTrace]:
        """The captured trace, available after exiting the context manager."""
        return self._trace

    @abstractmethod
    def __enter__(self) -> "BaseTracer": ...

    @abstractmethod
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...
