from __future__ import annotations
"""External Sort — 磁盘排序。
[P09] run 文件使用 pickle 二进制格式替代 JSON。"""
import os
import pickle
import tempfile
from typing import Any, Iterator, List, Optional
from structures.tournament_tree import LoserTree


class ExternalSort:
    """磁盘排序。数据量超过内存限制时使用。"""

    def __init__(self, memory_limit: int = 64 * 1024 * 1024,
                 temp_dir: Optional[str] = None) -> None:
        self._memory_limit = memory_limit
        self._temp_dir = temp_dir or tempfile.gettempdir()
        self._run_files: list = []

    def sort(self, rows: list,
             key_fn: Any = None) -> list:
        """排序。内存不足时溢写到磁盘。"""
        estimated_mem = len(rows) * 200
        if estimated_mem <= self._memory_limit:
            rows.sort(key=key_fn)
            return rows

        # 阶段 1：生成排序 run
        chunk_size = max(1, self._memory_limit // 200)
        self._run_files = []
        for start in range(0, len(rows), chunk_size):
            end = min(start + chunk_size, len(rows))
            chunk = rows[start:end]
            chunk.sort(key=key_fn)
            self._write_run(chunk)

        # 阶段 2：K 路归并
        return self._merge_runs(key_fn)

    def _write_run(self, sorted_chunk: list) -> None:
        """[P09] 二进制 run 文件。"""
        fd, path = tempfile.mkstemp(
            suffix='.run', dir=self._temp_dir)
        try:
            with os.fdopen(fd, 'wb') as f:
                for row in sorted_chunk:
                    data = pickle.dumps(row, protocol=4)
                    f.write(len(data).to_bytes(4, 'little'))
                    f.write(data)
        except Exception:
            os.close(fd)
            raise
        self._run_files.append(path)

    def _merge_runs(self, key_fn) -> list:
        if not self._run_files:
            return []
        file_handles = []
        iterators = []
        for path in self._run_files:
            fh = open(path, 'rb')
            file_handles.append(fh)
            iterators.append(self._run_iterator(fh))
        try:
            if key_fn:
                tree = LoserTree(iterators, key_fn=key_fn)
            else:
                tree = LoserTree(iterators)
            result = tree.merge_all()
        finally:
            for fh in file_handles:
                fh.close()
            self._cleanup()
        return result

    @staticmethod
    def _run_iterator(fh) -> Iterator:
        """[P09] 二进制 run 迭代器。"""
        while True:
            len_bytes = fh.read(4)
            if len(len_bytes) < 4:
                break
            data_len = int.from_bytes(len_bytes, 'little')
            data = fh.read(data_len)
            if len(data) < data_len:
                break
            yield pickle.loads(data)

    def _cleanup(self) -> None:
        for path in self._run_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._run_files = []
