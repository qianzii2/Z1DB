from __future__ import annotations
"""LSM Manifest — 跟踪活跃 SSTable 及其层级。
[FIX-S02] 使用原子 rename 防止崩溃时 MANIFEST 损坏。"""
import json
import os
from typing import Any, Dict, List, Optional
from storage.lsm.sstable import SSTableReader


class Manifest:
    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._levels: Dict[int, List[str]] = {0: [], 1: [], 2: []}
        self._next_id = 0
        self._load()

    def add_sstable(self, level: int, path: str) -> None:
        if level not in self._levels:
            self._levels[level] = []
        self._levels[level].append(path)
        self._save()

    def remove_sstable(self, level: int, path: str) -> None:
        if level in self._levels and path in self._levels[level]:
            self._levels[level].remove(path)
            self._save()

    def get_sstables(self, level: int) -> List[SSTableReader]:
        if level not in self._levels:
            return []
        result = []
        for path in self._levels[level]:
            if os.path.exists(path):
                result.append(SSTableReader(path))
        return result

    def all_sstables(self) -> List[SSTableReader]:
        result = []
        for level in sorted(self._levels.keys()):
            result.extend(self.get_sstables(level))
        return result

    def next_sstable_path(self, level: int) -> str:
        self._next_id += 1
        return os.path.join(
            self._data_dir,
            f'sst_L{level}_{self._next_id}.z1ss')

    def level_count(self, level: int) -> int:
        return len(self._levels.get(level, []))

    def _save(self) -> None:
        """[FIX-S02] 原子保存：写临时文件 → fsync → rename。"""
        path = os.path.join(self._data_dir, 'MANIFEST.json')
        tmp_path = path + '.tmp'
        os.makedirs(self._data_dir, exist_ok=True)
        data = {
            'levels': {str(k): v for k, v in self._levels.items()},
            'next_id': self._next_id,
        }
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)  # 原子替换
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load(self) -> None:
        path = os.path.join(self._data_dir, 'MANIFEST.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._levels = {
                    int(k): v
                    for k, v in data.get('levels', {}).items()}
                self._next_id = data.get('next_id', 0)
            except Exception:
                pass  # 损坏时用默认空状态
