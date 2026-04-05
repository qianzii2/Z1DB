from __future__ import annotations
"""Pattern-Defeating Quicksort — adaptive sort O(n log n) worst case.
Paper: Peters, 2021. Detects sorted/reverse/duplicates patterns."""
from typing import Any, Callable, List, Optional
import random


def pdqsort(arr: list, lo: int = 0, hi: int = -1,
            key: Optional[Callable] = None, reverse: bool = False) -> None:
    """In-place adaptive sort. Falls back to heapsort if recursion too deep."""
    if hi == -1:
        hi = len(arr) - 1
    if hi <= lo:
        return
    _pdqsort_impl(arr, lo, hi, 2 * (hi - lo + 1).bit_length(), key, reverse)


def _pdqsort_impl(arr: list, lo: int, hi: int, depth: int,
                   key: Optional[Callable], reverse: bool) -> None:
    while hi - lo > 24:
        if depth <= 0:
            _heapsort(arr, lo, hi, key, reverse)
            return

        # Check if already sorted
        if _is_sorted(arr, lo, hi, key, reverse):
            return

        # Check if reverse sorted
        if _is_sorted(arr, lo, hi, key, not reverse):
            _reverse_range(arr, lo, hi)
            return

        # Ninther pivot selection (median of medians of 3)
        pivot_idx = _ninther(arr, lo, hi, key, reverse)
        arr[lo], arr[pivot_idx] = arr[pivot_idx], arr[lo]

        # Dutch flag 3-way partition
        lt, gt = _three_way_partition(arr, lo, hi, key, reverse)

        depth -= 1
        # Recurse on smaller partition, iterate on larger (tail call opt)
        if lt - lo < hi - gt:
            _pdqsort_impl(arr, lo, lt - 1, depth, key, reverse)
            lo = gt + 1
        else:
            _pdqsort_impl(arr, gt + 1, hi, depth, key, reverse)
            hi = lt - 1

    _insertion_sort(arr, lo, hi, key, reverse)


def _key_val(arr: list, i: int, key: Optional[Callable]) -> Any:
    return key(arr[i]) if key else arr[i]


def _less(a: Any, b: Any, reverse: bool) -> bool:
    try:
        if reverse:
            return a > b
        return a < b
    except TypeError:
        return str(a) < str(b) if not reverse else str(a) > str(b)


def _is_sorted(arr: list, lo: int, hi: int,
               key: Optional[Callable], reverse: bool) -> bool:
    for i in range(lo, min(lo + 8, hi)):  # Check first 8 pairs
        a = _key_val(arr, i, key)
        b = _key_val(arr, i + 1, key)
        if _less(b, a, reverse):
            return False
    # Full check only if first 8 passed
    for i in range(lo, hi):
        a = _key_val(arr, i, key)
        b = _key_val(arr, i + 1, key)
        if _less(b, a, reverse):
            return False
    return True


def _reverse_range(arr: list, lo: int, hi: int) -> None:
    while lo < hi:
        arr[lo], arr[hi] = arr[hi], arr[lo]
        lo += 1; hi -= 1


def _ninther(arr: list, lo: int, hi: int,
             key: Optional[Callable], reverse: bool) -> int:
    """Median of medians of 3 — better pivot than median of 3."""
    n = hi - lo + 1
    s = n // 8
    a = _med3(arr, lo, lo + s, lo + 2 * s, key, reverse)
    b = _med3(arr, lo + 3 * s, lo + 4 * s, lo + 5 * s, key, reverse)
    c = _med3(arr, hi - 2 * s, hi - s, hi, key, reverse)
    return _med3_idx(arr, a, b, c, key, reverse)


def _med3(arr: list, a: int, b: int, c: int,
          key: Optional[Callable], reverse: bool) -> int:
    return _med3_idx(arr, a, b, c, key, reverse)


def _med3_idx(arr: list, a: int, b: int, c: int,
              key: Optional[Callable], reverse: bool) -> int:
    va, vb, vc = _key_val(arr, a, key), _key_val(arr, b, key), _key_val(arr, c, key)
    if _less(va, vb, reverse):
        if _less(vb, vc, reverse):
            return b
        return a if _less(va, vc, reverse) else c  # corrected: c if va >= vc
    else:
        if _less(va, vc, reverse):
            return a
        return b if _less(vb, vc, reverse) else c


def _three_way_partition(arr: list, lo: int, hi: int,
                         key: Optional[Callable], reverse: bool) -> tuple:
    """Dutch flag partition: <pivot | ==pivot | >pivot"""
    pivot = _key_val(arr, lo, key)
    lt = lo
    gt = hi
    i = lo + 1
    while i <= gt:
        vi = _key_val(arr, i, key)
        if _less(vi, pivot, reverse):
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1; i += 1
        elif _less(pivot, vi, reverse):
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
        else:
            i += 1
    return lt, gt


def _insertion_sort(arr: list, lo: int, hi: int,
                    key: Optional[Callable], reverse: bool) -> None:
    for i in range(lo + 1, hi + 1):
        tmp = arr[i]
        k = _key_val(arr, i, key)
        j = i - 1
        while j >= lo and _less(k, _key_val(arr, j, key), reverse):
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = tmp


def _heapsort(arr: list, lo: int, hi: int,
              key: Optional[Callable], reverse: bool) -> None:
    """Fallback heapsort guarantees O(n log n) worst case."""
    n = hi - lo + 1

    def sift_down(start: int, end: int) -> None:
        root = start
        while True:
            child = 2 * (root - lo) + 1 + lo
            if child > end:
                break
            if child + 1 <= end and _less(
                    _key_val(arr, child, key),
                    _key_val(arr, child + 1, key), reverse):
                child += 1
            if _less(_key_val(arr, root, key),
                     _key_val(arr, child, key), reverse):
                arr[root], arr[child] = arr[child], arr[root]
                root = child
            else:
                break

    # Build heap
    for i in range((n - 2) // 2 + lo, lo - 1, -1):
        sift_down(i, hi)
    # Extract
    for end in range(hi, lo, -1):
        arr[lo], arr[end] = arr[end], arr[lo]
        sift_down(lo, end - 1)
