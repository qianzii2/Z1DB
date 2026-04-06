from __future__ import annotations
"""临时文件包装 — 溢写操作用。保持文件句柄打开避免反复open/close。"""
import json
import os
from typing import Any, Iterator, List


class TempFile:
    """追加写入的临时文件。"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._count = 0
        self._fh = open(path, 'w')
        self._closed = False

    def write_row(self, row: list) -> None:
        """写入一行（保持句柄打开）。"""
        self._fh.write(json.dumps(row, default=str) + '\n')
        self._count += 1

    def write_rows(self, rows: List[list]) -> None:
        """批量写入多行。"""
        for row in rows:
            self._fh.write(json.dumps(row, default=str) + '\n')
        self._count += len(rows)

    def flush(self) -> None:
        """强制刷盘。"""
        if not self._closed:
            self._fh.flush()

    def close(self) -> None:
        """关闭写句柄。后续只能读。"""
        if not self._closed:
            self._fh.close()
            self._closed = True

    def read_all(self) -> List[list]:
        """读取全部行。如果还在写入状态会先关闭。"""
        self.close()
        rows = []
        try:
            with open(self._path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except FileNotFoundError:
            pass
        return rows

    def iterator(self) -> Iterator[list]:
        """流式读取。"""
        self.close()
        try:
            with open(self._path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        except FileNotFoundError:
            pass

    @property
    def count(self) -> int:
        return self._count

    def delete(self) -> None:
        """删除文件。"""
        self.close()
        try:
            os.unlink(self._path)
        except OSError:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

class BinaryTempFile:
    """二进制追加写入临时文件。比 JSON 快 5-10x。
    格式：每行 = [row_len:4B][pickled_row]"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._count = 0
        self._fh = open(path, 'wb')
        self._closed = False

    def write_row(self, row: list) -> None:
        import pickle
        data = pickle.dumps(row, protocol=4)
        self._fh.write(len(data).to_bytes(4, 'little'))
        self._fh.write(data)
        self._count += 1

    def write_rows(self, rows) -> None:
        for row in rows:
            self.write_row(row)

    def flush(self) -> None:
        if not self._closed:
            self._fh.flush()

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def read_all(self) -> list:
        self.close()
        import pickle
        rows = []
        try:
            with open(self._path, 'rb') as f:
                while True:
                    len_bytes = f.read(4)
                    if len(len_bytes) < 4:
                        break
                    data_len = int.from_bytes(len_bytes, 'little')
                    data = f.read(data_len)
                    if len(data) < data_len:
                        break
                    rows.append(pickle.loads(data))
        except FileNotFoundError:
            pass
        return rows

    @property
    def count(self) -> int:
        return self._count

    def delete(self) -> None:
        self.close()
        import os
        try:
            os.unlink(self._path)
        except OSError:
            pass