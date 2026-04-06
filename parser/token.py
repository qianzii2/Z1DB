from __future__ import annotations
"""Token 类型和数据类。每个 TokenType 对应一个 SQL 关键字或符号。"""
from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    # ═══ DML ═══
    SELECT = 'SELECT'
    FROM = 'FROM'
    WHERE = 'WHERE'
    INSERT = 'INSERT'
    INTO = 'INTO'
    VALUES = 'VALUES'
    UPDATE = 'UPDATE'
    DELETE = 'DELETE'
    SET = 'SET'

    # ═══ DDL ═══
    CREATE = 'CREATE'
    TABLE = 'TABLE'
    DROP = 'DROP'
    ALTER = 'ALTER'
    ADD = 'ADD'
    COLUMN = 'COLUMN'
    RENAME = 'RENAME'
    TO = 'TO'
    INDEX = 'INDEX'
    UNIQUE = 'UNIQUE'

    # ═══ 逻辑运算 ═══
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'

    # ═══ 排序/分页 ═══
    ORDER = 'ORDER'
    BY = 'BY'
    ASC = 'ASC'
    DESC = 'DESC'
    LIMIT = 'LIMIT'
    OFFSET = 'OFFSET'
    NULLS = 'NULLS'
    FIRST = 'FIRST'
    LAST = 'LAST'

    # ═══ 字面量/常量 ═══
    NULL = 'NULL'
    TRUE = 'TRUE'
    FALSE = 'FALSE'

    # ═══ 条件/存在 ═══
    IF = 'IF'
    EXISTS = 'EXISTS'
    IS = 'IS'
    AS = 'AS'

    # ═══ 约束 ═══
    PRIMARY = 'PRIMARY'
    KEY = 'KEY'

    # ═══ 分组/聚合 ═══
    GROUP = 'GROUP'
    HAVING = 'HAVING'
    DISTINCT = 'DISTINCT'

    # ═══ JOIN ═══
    JOIN = 'JOIN'
    ON = 'ON'
    USING = 'USING'
    NATURAL = 'NATURAL'
    INNER = 'INNER'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    FULL = 'FULL'
    OUTER = 'OUTER'
    CROSS = 'CROSS'

    # ═══ CASE ═══
    CASE = 'CASE'
    WHEN = 'WHEN'
    THEN = 'THEN'
    ELSE = 'ELSE'
    END = 'END'

    # ═══ 类型转换/谓词 ═══
    CAST = 'CAST'
    IN = 'IN'
    BETWEEN = 'BETWEEN'
    LIKE = 'LIKE'
    ESCAPE = 'ESCAPE'

    # ═══ 集合操作 ═══
    UNION = 'UNION'
    INTERSECT = 'INTERSECT'
    EXCEPT = 'EXCEPT'
    ALL = 'ALL'

    # ═══ 窗口函数 ═══
    OVER = 'OVER'
    PARTITION = 'PARTITION'
    ROWS = 'ROWS'
    RANGE = 'RANGE'
    UNBOUNDED = 'UNBOUNDED'
    PRECEDING = 'PRECEDING'
    FOLLOWING = 'FOLLOWING'
    CURRENT = 'CURRENT'
    ROW = 'ROW'

    # ═══ 其他语句 ═══
    EXPLAIN = 'EXPLAIN'
    COPY = 'COPY'
    WITH = 'WITH'
    VACUUM = 'VACUUM'       # [M05] VACUUM 命令

    # ═══ 数据类型关键字 ═══
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

    # ═══ 字面量 ═══
    INTEGER_LIT = 'INTEGER_LIT'
    FLOAT_LIT = 'FLOAT_LIT'
    STRING = 'STRING'
    IDENTIFIER = 'IDENTIFIER'

    # ═══ 运算符 ═══
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

    # ═══ 标点 ═══
    LPAREN = '('
    RPAREN = ')'
    COMMA = ','
    DOT = '.'
    SEMICOLON = ';'

    # ═══ 结束 ═══
    EOF = 'EOF'


@dataclass
class Token:
    """词法单元。"""
    type: TokenType
    value: str
    line: int = 1
    col: int = 1
