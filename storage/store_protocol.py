from __future__ import annotations
"""存储引擎统一协议。TableStore 和 LSMStore 都符合此协议。
扫描算子通过此协议访问数据，不依赖具体存储实现。"""
from typing import Any, List, Protocol, Set, runtime_checkable


@runtime_checkable
class StoreProtocol(Protocol):
    """所有存储引擎的接口契约。"""

    @property
    def schema(self) -> Any:
        """表 schema。"""
        ...

    @property
    def row_count(self) -> int:
        """逻辑行数（不含已删除行）。"""
        ...

    def append_row(self, row: list) -> None:
        """追加一行。"""
        ...

    def append_rows(self, rows: List[list]) -> int:
        """批量追加，返回追加行数。"""
        ...

    def read_all_rows(self) -> List[list]:
        """读取所有活跃行。"""
        ...

    def read_rows_by_indices(self,
                             indices: List[int]) -> List[list]:
        """按逻辑索引读取行。"""
        ...

    def get_chunk_count(self) -> int:
        """chunk 数量。"""
        ...

    def get_column_chunks(self,
                          column_name: str) -> list:
        """获取指定列的所有 ColumnChunk。"""
        ...

    def truncate(self) -> None:
        """清空全部数据。"""
        ...

    def delete_rows(self,
                    indices: Set[int]) -> int:
        """按逻辑索引删除行，返回实际删除行数。"""
        ...

    def update_rows(self, indices: Set[int],
                    col_idx: int,
                    new_values: dict) -> int:
        """更新指定行的指定列，返回实际更新行数。"""
        ...
