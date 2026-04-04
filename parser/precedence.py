from __future__ import annotations
"""Operator precedence levels for Pratt parsing."""

from enum import IntEnum


class Precedence(IntEnum):
    LOWEST = 0
    OR = 1
    AND = 2
    NOT_PREFIX = 3
    COMPARISON = 4
    IS = 4
    CONCAT = 5
    ADDITION = 6
    MULTIPLY = 7
    UNARY = 8
    HIGHEST = 9
