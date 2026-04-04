from __future__ import annotations
"""Unified error hierarchy for Z1DB."""


class Z1Error(Exception):
    """Base exception for all Z1DB errors."""

    def __init__(self, message: str, error_code: int = 0):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class ParseError(Z1Error):
    """Lexer / parser errors."""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(message)


class SemanticError(Z1Error):
    pass


class TypeMismatchError(Z1Error):
    def __init__(self, message: str, expected: str = '', actual: str = ''):
        self.expected = expected
        self.actual = actual
        super().__init__(message)


class ExecutionError(Z1Error):
    pass


class TableNotFoundError(Z1Error):
    def __init__(self, table_name: str):
        self.table_name = table_name
        super().__init__(f"table '{table_name}' not found")


class ColumnNotFoundError(Z1Error):
    def __init__(self, column_name: str):
        self.column_name = column_name
        super().__init__(f"column '{column_name}' not found")


class DuplicateError(Z1Error):
    pass


class DivisionByZeroError(Z1Error):
    def __init__(self, message: str = 'division by zero'):
        super().__init__(message)


class NumericOverflowError(Z1Error):
    def __init__(self, message: str = 'numeric overflow'):
        super().__init__(message)


class MemoryLimitError(Z1Error):
    pass


class RecursionLimitError(Z1Error):
    pass
