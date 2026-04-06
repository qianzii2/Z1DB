from __future__ import annotations
"""Pattern-Defeating Quicksort — 自适应 O(n log n)。
[修复] _is_sorted 只在顶层调用一次（O(n)），递归中不再调用。
原问题：递归每层 O(n) × O(log n) 层 = O(n² log n)。"""
from typing import Any, Callable, List, Optional


def pdqsort(arr: list, lo: int = 0, hi: int = -1,
            key: Optional[Callable] = None,
            reverse: bool = False) -> None:
    if hi == -1:
        hi = len(arr) - 1
    if hi <= lo:
        return
    # 顶层一次性检查（O(n)，只执行一次）
    if _is_sorted_full(arr, lo, hi, key, reverse):
        return
    if _is_sorted_full(arr, lo, hi, key, not reverse):
        _reverse_range(arr, lo, hi)
        return
    # 进入递归（不再检查排序状态）
    _pdqsort_impl(
        arr, lo, hi,
        2 * (hi - lo + 1).bit_length(), key, reverse)


def _pdqsort_impl(arr, lo, hi, depth, key, reverse):
    """递归体 — 无 _is_sorted 调用。"""
    while hi - lo > 24:
        if depth <= 0:
            _heapsort(arr, lo, hi, key, reverse)
            return
        # ❌ 原代码此处有 _is_sorted(arr, lo, hi, ...) — 已删除
        pivot_idx = _ninther(arr, lo, hi, key, reverse)
        arr[lo], arr[pivot_idx] = arr[pivot_idx], arr[lo]
        lt, gt = _three_way_partition(arr, lo, hi, key, reverse)
        depth -= 1
        if lt - lo < hi - gt:
            _pdqsort_impl(arr, lo, lt - 1, depth, key, reverse)
            lo = gt + 1
        else:
            _pdqsort_impl(arr, gt + 1, hi, depth, key, reverse)
            hi = lt - 1
    _insertion_sort(arr, lo, hi, key, reverse)


def _key_val(arr, i, key):
    return key(arr[i]) if key else arr[i]


def _less(a, b, reverse):
    try:
        return a > b if reverse else a < b
    except TypeError:
        return (str(a) > str(b) if reverse else str(a) < str(b))


def _is_sorted_full(arr, lo, hi, key, reverse):
    """完整排序检查。O(n) 但全项目只在 pdqsort 顶层调用一次。"""
    for i in range(lo, hi):
        a = _key_val(arr, i, key)
        b = _key_val(arr, i + 1, key)
        if _less(b, a, reverse):
            return False
    return True


def _reverse_range(arr, lo, hi):
    while lo < hi:
        arr[lo], arr[hi] = arr[hi], arr[lo]
        lo += 1
        hi -= 1


def _ninther(arr, lo, hi, key, reverse):
    n = hi - lo + 1
    s = n // 8
    a = _med3_idx(arr, lo, lo + s, lo + 2 * s, key, reverse)
    b = _med3_idx(arr, lo + 3 * s, lo + 4 * s, lo + 5 * s, key, reverse)
    c = _med3_idx(arr, hi - 2 * s, hi - s, hi, key, reverse)
    return _med3_idx(arr, a, b, c, key, reverse)


def _med3_idx(arr, a, b, c, key, reverse):
    va = _key_val(arr, a, key)
    vb = _key_val(arr, b, key)
    vc = _key_val(arr, c, key)
    if _less(va, vb, reverse):
        if _less(vb, vc, reverse):
            return b
        return a if _less(va, vc, reverse) else c
    else:
        if _less(va, vc, reverse):
            return a
        return b if _less(vb, vc, reverse) else c


def _three_way_partition(arr, lo, hi, key, reverse):
    pivot = _key_val(arr, lo, key)
    lt = lo
    gt = hi
    i = lo + 1
    while i <= gt:
        vi = _key_val(arr, i, key)
        if _less(vi, pivot, reverse):
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif _less(pivot, vi, reverse):
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
        else:
            i += 1
    return lt, gt


def _insertion_sort(arr, lo, hi, key, reverse):
    for i in range(lo + 1, hi + 1):
        tmp = arr[i]
        k = _key_val(arr, i, key)
        j = i - 1
        while j >= lo and _less(k, _key_val(arr, j, key), reverse):
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = tmp


def _heapsort(arr, lo, hi, key, reverse):
    n = hi - lo + 1

    def sift_down(start, end):
        root = start
        while True:
            child = 2 * (root - lo) + 1 + lo
            if child > end:
                break
            if (child + 1 <= end
                    and _less(
                        _key_val(arr, child, key),
                        _key_val(arr, child + 1, key),
                        reverse)):
                child += 1
            if _less(
                    _key_val(arr, root, key),
                    _key_val(arr, child, key),
                    reverse):
                arr[root], arr[child] = arr[child], arr[root]
                root = child
            else:
                break

    for i in range((n - 2) // 2 + lo, lo - 1, -1):
        sift_down(i, hi)
    for end in range(hi, lo, -1):
        arr[lo], arr[end] = arr[end], arr[lo]
        sift_down(lo, end - 1)
