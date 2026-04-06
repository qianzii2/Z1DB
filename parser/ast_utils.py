from __future__ import annotations
"""AST 工具函数 — 聚合/窗口检测。
validator / simple_planner / integrated_planner 共用此模块，避免重复实现。"""
import dataclasses
from typing import Any
from parser.ast import AggregateCall, AliasExpr

try:
    from parser.ast import WindowCall
except ImportError:
    WindowCall = None


def contains_agg(node: Any) -> bool:
    """检查表达式树中是否包含聚合函数调用。
    窗口函数内的聚合不算（WindowCall 内部不递归）。"""
    if isinstance(node, AggregateCall):
        return True
    if WindowCall and isinstance(node, WindowCall):
        return False  # 窗口函数不算聚合
    if isinstance(node, AliasExpr):
        return contains_agg(node.expr)
    if isinstance(node, tuple):
        return any(contains_agg(i) for i in node)
    if (node is None
            or not dataclasses.is_dataclass(node)
            or isinstance(node, type)):
        return False
    for f in dataclasses.fields(node):
        c = getattr(node, f.name)
        if isinstance(c, list):
            if any(contains_agg(i) for i in c):
                return True
        elif contains_agg(c):
            return True
    return False


def contains_window(node: Any) -> bool:
    """检查表达式树中是否包含窗口函数调用。"""
    if WindowCall and isinstance(node, WindowCall):
        return True
    if isinstance(node, AliasExpr):
        return contains_window(node.expr)
    if (node is None
            or not dataclasses.is_dataclass(node)
            or isinstance(node, type)):
        return False
    for f in dataclasses.fields(node):
        c = getattr(node, f.name)
        if isinstance(c, list):
            if any(contains_window(i) for i in c):
                return True
        elif contains_window(c):
            return True
    return False
