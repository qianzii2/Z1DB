from __future__ import annotations
"""运算符优先级 — Pratt 解析器使用。数值越大绑定越紧。"""
from enum import IntEnum


class Precedence(IntEnum):
    LOWEST = 0         # 最低优先级
    OR = 1             # OR
    AND = 2            # AND
    NOT_PREFIX = 3     # NOT（前缀）
    COMPARISON = 4     # = != < > <= >= IS IN BETWEEN LIKE
    IS = 4             # IS NULL / IS NOT NULL
    CONCAT = 5         # ||
    ADDITION = 6       # + -
    MULTIPLY = 7       # * / %
    UNARY = 8          # 一元 +/-
    HIGHEST = 9        # 最高优先级
