from __future__ import annotations
"""Base operator interface for the Volcano execution model."""

from abc import ABC, abstractmethod
from typing import List, Optional

from executor.core.batch import VectorBatch
from storage.types import DataType


class Operator(ABC):
    """Abstract pull-based operator."""

    def __init__(self) -> None:
        self.children: List[Operator] = []

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def next_batch(self) -> Optional[VectorBatch]: ...

    @abstractmethod
    def close(self) -> None: ...

    def output_schema(self) -> List[tuple[str, DataType]]:
        """Return [(column_name, dtype), ...].  May be called before open()."""
        return []

    def explain(self, indent: int = 0) -> str:
        name = type(self).__name__
        prefix = '  ' * indent
        lines = [f"{prefix}{name}"]
        for child in self.children:
            lines.append(child.explain(indent + 1))
        return '\n'.join(lines)
