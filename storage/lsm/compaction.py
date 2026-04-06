from __future__ import annotations
"""LSM Compaction — 合并多个 SSTable 为一个。
N 路归并，同 key 取最新版本，tombstone 被丢弃。
先写新 SSTable，由调用方更新 MANIFEST 后再删旧文件。"""
from typing import Any, Iterator, List, Optional, Tuple
from storage.lsm.sstable import SSTableWriter, SSTableReader


class Compactor:
    """SSTable 合并器。"""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir

    def compact(self, sstables: List[SSTableReader],
                output_path: str,
                columns: List[str]) -> SSTableReader:
        """合并 N 个 SSTable 为 1 个。
        返回新 SSTable，调用方负责更新 MANIFEST 后删旧文件。"""
        writer = SSTableWriter(output_path, columns)

        # N 路归并
        iters = [st.scan() for st in sstables]
        heads: List[Optional[Tuple[Any, Any]]] = []
        for it in iters:
            try:
                heads.append(next(it))
            except StopIteration:
                heads.append(None)

        while True:
            min_key = None
            min_idx = -1
            for i, head in enumerate(heads):
                if head is None:
                    continue
                k = head[0]
                if min_key is None:
                    min_key = k
                    min_idx = i
                else:
                    try:
                        if k < min_key:
                            min_key = k
                            min_idx = i
                    except TypeError:
                        pass

            if min_idx == -1:
                break

            # 同 key 取最后一个版本
            final_row = None
            for i, head in enumerate(heads):
                if head is None:
                    continue
                if head[0] == min_key:
                    final_row = head[1]
                    try:
                        heads[i] = next(iters[i])
                    except StopIteration:
                        heads[i] = None

            # 跳过 tombstone
            if final_row is not None:
                writer.add(min_key, final_row)

        writer.finish()
        return SSTableReader(output_path)
