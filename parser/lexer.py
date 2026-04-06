from __future__ import annotations
"""SQL 词法分析器。将 SQL 文本分割为 Token 序列。
支持：关键字、标识符、数字、字符串、双引号标识符、行注释、块注释。"""
from typing import Dict, Optional
from parser.token import Token, TokenType
from utils.errors import ParseError

# SQL 关键字 → TokenType 映射
KEYWORDS: Dict[str, TokenType] = {
    'SELECT': TokenType.SELECT, 'FROM': TokenType.FROM,
    'WHERE': TokenType.WHERE,
    'INSERT': TokenType.INSERT, 'INTO': TokenType.INTO,
    'VALUES': TokenType.VALUES,
    'CREATE': TokenType.CREATE, 'TABLE': TokenType.TABLE,
    'DROP': TokenType.DROP,
    'AND': TokenType.AND, 'OR': TokenType.OR,
    'NOT': TokenType.NOT,
    'ORDER': TokenType.ORDER, 'BY': TokenType.BY,
    'ASC': TokenType.ASC, 'DESC': TokenType.DESC,
    'LIMIT': TokenType.LIMIT, 'OFFSET': TokenType.OFFSET,
    'NULL': TokenType.NULL, 'TRUE': TokenType.TRUE,
    'FALSE': TokenType.FALSE,
    'IF': TokenType.IF, 'EXISTS': TokenType.EXISTS,
    'IS': TokenType.IS, 'AS': TokenType.AS,
    'NULLS': TokenType.NULLS, 'FIRST': TokenType.FIRST,
    'LAST': TokenType.LAST,
    'PRIMARY': TokenType.PRIMARY, 'KEY': TokenType.KEY,
    'GROUP': TokenType.GROUP, 'HAVING': TokenType.HAVING,
    'DISTINCT': TokenType.DISTINCT,
    'JOIN': TokenType.JOIN, 'ON': TokenType.ON,
    'USING': TokenType.USING, 'NATURAL': TokenType.NATURAL,
    'INNER': TokenType.INNER, 'LEFT': TokenType.LEFT,
    'RIGHT': TokenType.RIGHT, 'FULL': TokenType.FULL,
    'OUTER': TokenType.OUTER, 'CROSS': TokenType.CROSS,
    'UPDATE': TokenType.UPDATE, 'DELETE': TokenType.DELETE,
    'SET': TokenType.SET,
    'CASE': TokenType.CASE, 'WHEN': TokenType.WHEN,
    'THEN': TokenType.THEN, 'ELSE': TokenType.ELSE,
    'END': TokenType.END,
    'CAST': TokenType.CAST, 'IN': TokenType.IN,
    'BETWEEN': TokenType.BETWEEN, 'LIKE': TokenType.LIKE,
    'ESCAPE': TokenType.ESCAPE,
    'UNION': TokenType.UNION, 'INTERSECT': TokenType.INTERSECT,
    'EXCEPT': TokenType.EXCEPT, 'ALL': TokenType.ALL,
    'OVER': TokenType.OVER, 'PARTITION': TokenType.PARTITION,
    'ROWS': TokenType.ROWS, 'RANGE': TokenType.RANGE,
    'UNBOUNDED': TokenType.UNBOUNDED,
    'PRECEDING': TokenType.PRECEDING,
    'FOLLOWING': TokenType.FOLLOWING,
    'CURRENT': TokenType.CURRENT, 'ROW': TokenType.ROW,
    'EXPLAIN': TokenType.EXPLAIN, 'ALTER': TokenType.ALTER,
    'ADD': TokenType.ADD, 'COLUMN': TokenType.COLUMN,
    'RENAME': TokenType.RENAME, 'TO': TokenType.TO,
    'INDEX': TokenType.INDEX, 'UNIQUE': TokenType.UNIQUE,
    'COPY': TokenType.COPY, 'WITH': TokenType.WITH,
    'VACUUM': TokenType.VACUUM,
    'INT': TokenType.INT, 'INTEGER': TokenType.INTEGER,
    'BIGINT': TokenType.BIGINT,
    'FLOAT': TokenType.FLOAT_KW, 'DOUBLE': TokenType.DOUBLE,
    'REAL': TokenType.REAL,
    'BOOLEAN': TokenType.BOOLEAN, 'BOOL': TokenType.BOOL,
    'VARCHAR': TokenType.VARCHAR, 'TEXT': TokenType.TEXT_KW,
    'DATE': TokenType.DATE_KW, 'TIMESTAMP': TokenType.TIMESTAMP,
}

# 单字符符号 → TokenType
_SINGLE: Dict[str, TokenType] = {
    '+': TokenType.PLUS, '-': TokenType.MINUS,
    '*': TokenType.STAR, '/': TokenType.SLASH,
    '%': TokenType.PERCENT, '=': TokenType.EQUAL,
    '<': TokenType.LESS, '>': TokenType.GREATER,
    '(': TokenType.LPAREN, ')': TokenType.RPAREN,
    ',': TokenType.COMMA, '.': TokenType.DOT,
    ';': TokenType.SEMICOLON,
}


class Lexer:
    """SQL 词法分析器。逐字符扫描，输出 Token 列表。"""

    def __init__(self, source: str) -> None:
        self._s = source
        self._p = 0
        self._ln = 1
        self._co = 1
        self._c: Optional[str] = (
            source[0] if source else None)

    def tokenize(self) -> list[Token]:
        """扫描全部 Token，末尾附加 EOF。"""
        t: list[Token] = []
        while True:
            self._skip()
            if self._c is None:
                break
            t.append(self._next())
        t.append(Token(TokenType.EOF, '', self._ln, self._co))
        return t

    def _adv(self):
        """推进一个字符。"""
        if self._c == '\n':
            self._ln += 1
            self._co = 1
        else:
            self._co += 1
        self._p += 1
        self._c = (self._s[self._p]
                    if self._p < len(self._s) else None)

    def _pk(self) -> Optional[str]:
        """前瞻一个字符。"""
        n = self._p + 1
        return self._s[n] if n < len(self._s) else None

    def _skip(self):
        """跳过空白和注释。"""
        while self._c is not None:
            if self._c.isspace():
                self._adv()
            elif self._c == '-' and self._pk() == '-':
                # 行注释
                while self._c is not None and self._c != '\n':
                    self._adv()
            elif self._c == '/' and self._pk() == '*':
                # 块注释（支持嵌套）
                self._adv()
                self._adv()
                d = 1
                while self._c is not None and d > 0:
                    if self._c == '/' and self._pk() == '*':
                        d += 1
                        self._adv()
                    elif self._c == '*' and self._pk() == '/':
                        d -= 1
                        self._adv()
                    self._adv()
                if d > 0:
                    raise ParseError(
                        "unterminated comment",
                        self._ln, self._co)
            else:
                break

    def _next(self) -> Token:
        """识别下一个 Token。"""
        ch = self._c
        assert ch is not None
        if ch.isdigit():
            return self._num()
        if ch.isalpha() or ch == '_':
            return self._ident()
        if ch == "'":
            return self._str()
        if ch == '"':
            return self._quoted_ident()
        # 双字符运算符
        nx = self._pk()
        pr = ch + (nx or '')
        ln, co = self._ln, self._co
        if pr == '<=':
            self._adv(); self._adv()
            return Token(TokenType.LESS_EQUAL, '<=', ln, co)
        if pr == '>=':
            self._adv(); self._adv()
            return Token(TokenType.GREATER_EQUAL, '>=', ln, co)
        if pr == '!=':
            self._adv(); self._adv()
            return Token(TokenType.NOT_EQUAL, '!=', ln, co)
        if pr == '<>':
            self._adv(); self._adv()
            return Token(TokenType.NOT_EQUAL, '!=', ln, co)
        if pr == '||':
            self._adv(); self._adv()
            return Token(TokenType.PIPE_PIPE, '||', ln, co)
        # 单字符运算符
        if ch in _SINGLE:
            tt = _SINGLE[ch]
            self._adv()
            return Token(tt, ch, ln, co)
        raise ParseError(
            f"unexpected: {ch!r}", self._ln, self._co)

    def _num(self):
        """数字字面量（整数或浮点）。"""
        ln, co = self._ln, self._co
        s = self._p
        is_float = False
        while self._c is not None and self._c.isdigit():
            self._adv()
        if (self._c == '.' and self._pk() is not None
                and self._pk().isdigit()):
            is_float = True
            self._adv()
            while self._c is not None and self._c.isdigit():
                self._adv()
        if self._c is not None and self._c in ('e', 'E'):
            nx = self._pk()
            if nx is not None and (nx.isdigit() or nx in ('+', '-')):
                is_float = True
                self._adv()
                if self._c in ('+', '-'):
                    self._adv()
                if self._c is None or not self._c.isdigit():
                    raise ParseError(
                        "invalid exponent", self._ln, self._co)
                while self._c is not None and self._c.isdigit():
                    self._adv()
        text = self._s[s:self._p]
        return Token(
            TokenType.FLOAT_LIT if is_float
            else TokenType.INTEGER_LIT,
            text, ln, co)

    def _str(self):
        """单引号字符串。支持 '' 转义。"""
        ln, co = self._ln, self._co
        self._adv()
        p: list = []
        while True:
            if self._c is None:
                raise ParseError("unterminated string", ln, co)
            if self._c == "'":
                if self._pk() == "'":
                    p.append("'")
                    self._adv()
                    self._adv()
                else:
                    self._adv()
                    break
            else:
                p.append(self._c)
                self._adv()
        return Token(TokenType.STRING, ''.join(p), ln, co)

    def _quoted_ident(self):
        """双引号标识符。"""
        ln, co = self._ln, self._co
        self._adv()
        p: list = []
        while True:
            if self._c is None:
                raise ParseError(
                    "unterminated quoted identifier", ln, co)
            if self._c == '"':
                if self._pk() == '"':
                    p.append('"')
                    self._adv()
                    self._adv()
                else:
                    self._adv()
                    break
            else:
                p.append(self._c)
                self._adv()
        return Token(TokenType.IDENTIFIER, ''.join(p), ln, co)

    def _ident(self):
        """标识符或关键字。"""
        ln, co = self._ln, self._co
        s = self._p
        while (self._c is not None
               and (self._c.isalnum() or self._c == '_')):
            self._adv()
        t = self._s[s:self._p]
        u = t.upper()
        if u in KEYWORDS:
            return Token(KEYWORDS[u], u, ln, co)
        return Token(TokenType.IDENTIFIER, t.lower(), ln, co)
