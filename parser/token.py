from __future__ import annotations
"""Token types and the Token dataclass."""

from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    # -- SQL reserved keywords -----------------------------------------
    SELECT = 'SELECT'
    FROM = 'FROM'
    WHERE = 'WHERE'
    INSERT = 'INSERT'
    INTO = 'INTO'
    VALUES = 'VALUES'
    CREATE = 'CREATE'
    TABLE = 'TABLE'
    DROP = 'DROP'
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'
    ORDER = 'ORDER'
    BY = 'BY'
    ASC = 'ASC'
    DESC = 'DESC'
    LIMIT = 'LIMIT'
    OFFSET = 'OFFSET'
    NULL = 'NULL'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    IF = 'IF'
    EXISTS = 'EXISTS'
    IS = 'IS'
    AS = 'AS'
    NULLS = 'NULLS'
    FIRST = 'FIRST'
    LAST = 'LAST'
    PRIMARY = 'PRIMARY'
    KEY = 'KEY'
    GROUP = 'GROUP'
    HAVING = 'HAVING'
    DISTINCT = 'DISTINCT'
    JOIN = 'JOIN'
    ON = 'ON'

    # -- Type-name keywords (unreserved — may be used as identifiers) --
    INT = 'INT'
    INTEGER = 'INTEGER'
    BIGINT = 'BIGINT'
    FLOAT_KW = 'FLOAT'
    DOUBLE = 'DOUBLE'
    REAL = 'REAL'
    BOOLEAN = 'BOOLEAN'
    BOOL = 'BOOL'
    VARCHAR = 'VARCHAR'
    TEXT_KW = 'TEXT'
    DATE_KW = 'DATE'
    TIMESTAMP = 'TIMESTAMP'

    # -- Literals ------------------------------------------------------
    INTEGER_LIT = 'INTEGER_LIT'
    FLOAT_LIT = 'FLOAT_LIT'
    STRING = 'STRING'
    IDENTIFIER = 'IDENTIFIER'

    # -- Operators -----------------------------------------------------
    PLUS = '+'
    MINUS = '-'
    STAR = '*'
    SLASH = '/'
    PERCENT = '%'
    EQUAL = '='
    NOT_EQUAL = '!='
    LESS = '<'
    GREATER = '>'
    LESS_EQUAL = '<='
    GREATER_EQUAL = '>='
    PIPE_PIPE = '||'

    # -- Delimiters ----------------------------------------------------
    LPAREN = '('
    RPAREN = ')'
    COMMA = ','
    DOT = '.'
    SEMICOLON = ';'

    # -- Special -------------------------------------------------------
    EOF = 'EOF'


@dataclass
class Token:
    type: TokenType
    value: str
    line: int = 1
    col: int = 1
