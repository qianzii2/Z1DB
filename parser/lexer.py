from __future__ import annotations
"""SQL lexer — tokenises a source string."""

from typing import Dict, Optional

from parser.token import Token, TokenType
from utils.errors import ParseError

# Keyword lookup table (upper-cased source text → TokenType)
KEYWORDS: Dict[str, TokenType] = {
    'SELECT': TokenType.SELECT, 'FROM': TokenType.FROM, 'WHERE': TokenType.WHERE,
    'INSERT': TokenType.INSERT, 'INTO': TokenType.INTO, 'VALUES': TokenType.VALUES,
    'CREATE': TokenType.CREATE, 'TABLE': TokenType.TABLE, 'DROP': TokenType.DROP,
    'AND': TokenType.AND, 'OR': TokenType.OR, 'NOT': TokenType.NOT,
    'ORDER': TokenType.ORDER, 'BY': TokenType.BY, 'ASC': TokenType.ASC,
    'DESC': TokenType.DESC, 'LIMIT': TokenType.LIMIT, 'OFFSET': TokenType.OFFSET,
    'NULL': TokenType.NULL, 'TRUE': TokenType.TRUE, 'FALSE': TokenType.FALSE,
    'IF': TokenType.IF, 'EXISTS': TokenType.EXISTS,
    'IS': TokenType.IS, 'AS': TokenType.AS,
    'NULLS': TokenType.NULLS, 'FIRST': TokenType.FIRST, 'LAST': TokenType.LAST,
    'PRIMARY': TokenType.PRIMARY, 'KEY': TokenType.KEY,
    'GROUP': TokenType.GROUP, 'HAVING': TokenType.HAVING,
    'DISTINCT': TokenType.DISTINCT, 'JOIN': TokenType.JOIN, 'ON': TokenType.ON,
    # Type-name keywords
    'INT': TokenType.INT, 'INTEGER': TokenType.INTEGER, 'BIGINT': TokenType.BIGINT,
    'FLOAT': TokenType.FLOAT_KW, 'DOUBLE': TokenType.DOUBLE, 'REAL': TokenType.REAL,
    'BOOLEAN': TokenType.BOOLEAN, 'BOOL': TokenType.BOOL,
    'VARCHAR': TokenType.VARCHAR, 'TEXT': TokenType.TEXT_KW,
    'DATE': TokenType.DATE_KW, 'TIMESTAMP': TokenType.TIMESTAMP,
}

# Single-character operator / delimiter map
_SINGLE_CHAR: Dict[str, TokenType] = {
    '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR,
    '/': TokenType.SLASH, '%': TokenType.PERCENT, '=': TokenType.EQUAL,
    '<': TokenType.LESS, '>': TokenType.GREATER,
    '(': TokenType.LPAREN, ')': TokenType.RPAREN,
    ',': TokenType.COMMA, '.': TokenType.DOT, ';': TokenType.SEMICOLON,
}


class Lexer:
    """Converts a SQL source string into a list of tokens."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._line = 1
        self._col = 1
        self._current: Optional[str] = source[0] if source else None

    # ------------------------------------------------------------------
    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            self._skip_whitespace_and_comments()
            if self._current is None:
                break
            tok = self._next_token()
            tokens.append(tok)
        tokens.append(Token(TokenType.EOF, '', self._line, self._col))
        return tokens

    # -- character movement --------------------------------------------
    def _advance(self) -> None:
        if self._current == '\n':
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        self._pos += 1
        self._current = self._source[self._pos] if self._pos < len(self._source) else None

    def _peek(self) -> Optional[str]:
        nxt = self._pos + 1
        return self._source[nxt] if nxt < len(self._source) else None

    # ------------------------------------------------------------------
    def _skip_whitespace_and_comments(self) -> None:
        while self._current is not None:
            if self._current.isspace():
                self._advance()
            elif self._current == '-' and self._peek() == '-':
                # line comment
                while self._current is not None and self._current != '\n':
                    self._advance()
            elif self._current == '/' and self._peek() == '*':
                self._advance()  # /
                self._advance()  # *
                depth = 1
                while self._current is not None and depth > 0:
                    if self._current == '/' and self._peek() == '*':
                        depth += 1
                        self._advance()
                    elif self._current == '*' and self._peek() == '/':
                        depth -= 1
                        self._advance()
                    self._advance()
                if depth > 0:
                    raise ParseError("unterminated block comment", self._line, self._col)
            else:
                break

    # ------------------------------------------------------------------
    def _next_token(self) -> Token:
        assert self._current is not None
        ch = self._current

        if ch.isdigit():
            return self._scan_number()
        if ch.isalpha() or ch == '_':
            return self._scan_identifier_or_keyword()
        if ch == "'":
            return self._scan_string()

        # Two-character operators
        nxt = self._peek()
        pair = ch + (nxt or '')
        line, col = self._line, self._col

        if pair == '<=':
            self._advance(); self._advance()
            return Token(TokenType.LESS_EQUAL, '<=', line, col)
        if pair == '>=':
            self._advance(); self._advance()
            return Token(TokenType.GREATER_EQUAL, '>=', line, col)
        if pair == '!=':
            self._advance(); self._advance()
            return Token(TokenType.NOT_EQUAL, '!=', line, col)
        if pair == '<>':
            self._advance(); self._advance()
            return Token(TokenType.NOT_EQUAL, '!=', line, col)  # normalise
        if pair == '||':
            self._advance(); self._advance()
            return Token(TokenType.PIPE_PIPE, '||', line, col)

        # Single-character
        if ch in _SINGLE_CHAR:
            tt = _SINGLE_CHAR[ch]
            self._advance()
            return Token(tt, ch, line, col)

        raise ParseError(f"unexpected character: {ch!r}", self._line, self._col)

    # ------------------------------------------------------------------
    def _scan_number(self) -> Token:
        line, col = self._line, self._col
        start = self._pos
        while self._current is not None and self._current.isdigit():
            self._advance()
        if self._current == '.' and self._peek() is not None and self._peek().isdigit():  # type: ignore[union-attr]
            self._advance()  # consume '.'
            while self._current is not None and self._current.isdigit():
                self._advance()
            text = self._source[start:self._pos]
            return Token(TokenType.FLOAT_LIT, text, line, col)
        text = self._source[start:self._pos]
        return Token(TokenType.INTEGER_LIT, text, line, col)

    def _scan_string(self) -> Token:
        line, col = self._line, self._col
        self._advance()  # opening '
        parts: list[str] = []
        while True:
            if self._current is None:
                raise ParseError("unterminated string literal", line, col)
            if self._current == "'":
                if self._peek() == "'":
                    parts.append("'")
                    self._advance(); self._advance()
                else:
                    self._advance()  # closing '
                    break
            else:
                parts.append(self._current)
                self._advance()
        return Token(TokenType.STRING, ''.join(parts), line, col)

    def _scan_identifier_or_keyword(self) -> Token:
        line, col = self._line, self._col
        start = self._pos
        while self._current is not None and (self._current.isalnum() or self._current == '_'):
            self._advance()
        text = self._source[start:self._pos]
        upper = text.upper()
        if upper in KEYWORDS:
            return Token(KEYWORDS[upper], upper, line, col)
        return Token(TokenType.IDENTIFIER, text.lower(), line, col)
