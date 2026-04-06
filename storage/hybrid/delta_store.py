from __future__ import annotations
"""Delta Store — 行式追加写缓冲。
论文: Grund et al., 2010 "HYRISE"

注意：当前版本中 TableStore 未使用 DeltaStore（功能由
_deleted_global + append 机制覆盖）。
此模块保留供 LSM 层和未来版本使用。"""
from typing import Any, Dict, List, Optional, Set
from metal.bitmap import Bitmap


class DeltaStore:
    """追加写缓冲。按插入顺序存储行。支持逻辑删除（tombstone）。"""

    __slots__ = ('_rows', '_column_names',
                 '_delete_bitmap', '_next_id')

    def __init__(self, column_names: List[str]) -> None:
        self._column_names = column_names
        self._rows: List[list] = []
        self._delete_bitmap = Bitmap(0)
        self._next_id = 0

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def active_row_count(self) -> int:
        """未被标记删除的行数。"""
        return self.row_count - self._delete_bitmap.popcount()

    def insert(self, row: list) -> int:
        """追加行，返回行 ID。"""
        row_id = self._next_id
        self._rows.append(list(row))
        self._delete_bitmap.ensure_capacity(row_id + 1)
        self._next_id += 1
        return row_id

    def mark_deleted(self, row_id: int) -> None:
        """标记删除（tombstone）。"""
        if 0 <= row_id < len(self._rows):
            self._delete_bitmap.set_bit(row_id)

    def is_deleted(self, row_id: int) -> bool:
        return self._delete_bitmap.get_bit(row_id)

    def get_row(self, row_id: int) -> Optional[list]:
        if row_id < 0 or row_id >= len(self._rows):
            return None
        if self._delete_bitmap.get_bit(row_id):
            return None
        return self._rows[row_id]

    def scan_active(self) -> List[list]:
        """扫描所有未删除行。"""
        result = []
        for i, row in enumerate(self._rows):
            if not self._delete_bitmap.get_bit(i):
                result.append(row)
        return result

    def drain(self) -> List[list]:
        """提取全部活跃行并清空。合并（merge）时使用。"""
        active = self.scan_active()
        self.clear()
        return active

    def clear(self) -> None:
        self._rows.clear()
        self._delete_bitmap = Bitmap(0)
        self._next_id = 0
