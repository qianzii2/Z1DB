from __future__ import annotations

"""LSM Compaction — merge multiple SSTables into one.
Removes tombstones and duplicate keys during merge."""
import os
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple
from storage.lsm.sstable import SSTableWriter, SSTableReader


class Compactor:
    """Merges multiple SSTables into a single sorted SSTable."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir

    def compact(self, sstables: List[SSTableReader],
                output_path: str, columns: List[str]) -> SSTableReader:
        """Merge N SSTables into 1, removing tombstones and duplicates.

        Uses a simple N-way merge (latest key wins for duplicates).
        Tombstones (row=None) are removed in the final output.
        """
        writer = SSTableWriter(output_path, columns)

        # N-way merge using iterators
        iters: List[Iterator] = [st.scan() for st in sstables]
        heads: List[Optional[Tuple[Any, Any]]] = []
        for it in iters:
            try:
                heads.append(next(it))
            except StopIteration:
                heads.append(None)

        while True:
            # Find minimum key among all heads
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
                break  # All iterators exhausted

            # Collect all entries with this key (last one wins)
            final_row = None
            for i, head in enumerate(heads):
                if head is None:
                    continue
                if head[0] == min_key:
                    final_row = head[1]  # Latest version wins
                    try:
                        heads[i] = next(iters[i])
                    except StopIteration:
                        heads[i] = None

            # Skip tombstones
            if final_row is not None:
                writer.add(min_key, final_row)

        meta = writer.finish()

        # Delete old SSTables
        for st in sstables:
            st.delete_files()

        return SSTableReader(output_path)
