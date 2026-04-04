from __future__ import annotations
"""SQL lexer."""
from typing import Dict, Optional
from parser.token import Token, TokenType
from utils.errors import ParseError

KEYWORDS: Dict[str, TokenType] = {
    'SELECT':TokenType.SELECT,'FROM':TokenType.FROM,'WHERE':TokenType.WHERE,
    'INSERT':TokenType.INSERT,'INTO':TokenType.INTO,'VALUES':TokenType.VALUES,
    'CREATE':TokenType.CREATE,'TABLE':TokenType.TABLE,'DROP':TokenType.DROP,
    'AND':TokenType.AND,'OR':TokenType.OR,'NOT':TokenType.NOT,
    'ORDER':TokenType.ORDER,'BY':TokenType.BY,'ASC':TokenType.ASC,'DESC':TokenType.DESC,
    'LIMIT':TokenType.LIMIT,'OFFSET':TokenType.OFFSET,
    'NULL':TokenType.NULL,'TRUE':TokenType.TRUE,'FALSE':TokenType.FALSE,
    'IF':TokenType.IF,'EXISTS':TokenType.EXISTS,'IS':TokenType.IS,'AS':TokenType.AS,
    'NULLS':TokenType.NULLS,'FIRST':TokenType.FIRST,'LAST':TokenType.LAST,
    'PRIMARY':TokenType.PRIMARY,'KEY':TokenType.KEY,
    'GROUP':TokenType.GROUP,'HAVING':TokenType.HAVING,'DISTINCT':TokenType.DISTINCT,
    'JOIN':TokenType.JOIN,'ON':TokenType.ON,
    'INNER':TokenType.INNER,'LEFT':TokenType.LEFT,'RIGHT':TokenType.RIGHT,
    'FULL':TokenType.FULL,'OUTER':TokenType.OUTER,'CROSS':TokenType.CROSS,
    'UPDATE':TokenType.UPDATE,'DELETE':TokenType.DELETE,'SET':TokenType.SET,
    'CASE':TokenType.CASE,'WHEN':TokenType.WHEN,'THEN':TokenType.THEN,
    'ELSE':TokenType.ELSE,'END':TokenType.END,
    'CAST':TokenType.CAST,'IN':TokenType.IN,'BETWEEN':TokenType.BETWEEN,
    'LIKE':TokenType.LIKE,'ESCAPE':TokenType.ESCAPE,
    'UNION':TokenType.UNION,'INTERSECT':TokenType.INTERSECT,'EXCEPT':TokenType.EXCEPT,
    'ALL':TokenType.ALL,
    'INT':TokenType.INT,'INTEGER':TokenType.INTEGER,'BIGINT':TokenType.BIGINT,
    'FLOAT':TokenType.FLOAT_KW,'DOUBLE':TokenType.DOUBLE,'REAL':TokenType.REAL,
    'BOOLEAN':TokenType.BOOLEAN,'BOOL':TokenType.BOOL,
    'VARCHAR':TokenType.VARCHAR,'TEXT':TokenType.TEXT_KW,
    'DATE':TokenType.DATE_KW,'TIMESTAMP':TokenType.TIMESTAMP,
}
_SINGLE_CHAR: Dict[str, TokenType] = {
    '+':TokenType.PLUS,'-':TokenType.MINUS,'*':TokenType.STAR,
    '/':TokenType.SLASH,'%':TokenType.PERCENT,'=':TokenType.EQUAL,
    '<':TokenType.LESS,'>':TokenType.GREATER,
    '(':TokenType.LPAREN,')':TokenType.RPAREN,
    ',':TokenType.COMMA,'.':TokenType.DOT,';':TokenType.SEMICOLON,
}

class Lexer:
    def __init__(self, source: str) -> None:
        self._source = source; self._pos = 0; self._line = 1; self._col = 1
        self._current: Optional[str] = source[0] if source else None

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            self._skip_ws()
            if self._current is None: break
            tokens.append(self._next_token())
        tokens.append(Token(TokenType.EOF, '', self._line, self._col))
        return tokens

    def _advance(self) -> None:
        if self._current == '\n': self._line += 1; self._col = 1
        else: self._col += 1
        self._pos += 1
        self._current = self._source[self._pos] if self._pos < len(self._source) else None

    def _peek(self) -> Optional[str]:
        n = self._pos + 1
        return self._source[n] if n < len(self._source) else None

    def _skip_ws(self) -> None:
        while self._current is not None:
            if self._current.isspace(): self._advance()
            elif self._current == '-' and self._peek() == '-':
                while self._current is not None and self._current != '\n': self._advance()
            elif self._current == '/' and self._peek() == '*':
                self._advance(); self._advance(); d = 1
                while self._current is not None and d > 0:
                    if self._current == '/' and self._peek() == '*': d += 1; self._advance()
                    elif self._current == '*' and self._peek() == '/': d -= 1; self._advance()
                    self._advance()
                if d > 0: raise ParseError("unterminated block comment", self._line, self._col)
            else: break

    def _next_token(self) -> Token:
        ch = self._current; assert ch is not None
        if ch.isdigit(): return self._scan_number()
        if ch.isalpha() or ch == '_': return self._scan_ident()
        if ch == "'": return self._scan_string()
        nxt = self._peek(); pair = ch + (nxt or ''); ln, co = self._line, self._col
        if pair == '<=': self._advance(); self._advance(); return Token(TokenType.LESS_EQUAL,'<=',ln,co)
        if pair == '>=': self._advance(); self._advance(); return Token(TokenType.GREATER_EQUAL,'>=',ln,co)
        if pair == '!=': self._advance(); self._advance(); return Token(TokenType.NOT_EQUAL,'!=',ln,co)
        if pair == '<>': self._advance(); self._advance(); return Token(TokenType.NOT_EQUAL,'!=',ln,co)
        if pair == '||': self._advance(); self._advance(); return Token(TokenType.PIPE_PIPE,'||',ln,co)
        if ch in _SINGLE_CHAR:
            tt = _SINGLE_CHAR[ch]; self._advance(); return Token(tt, ch, ln, co)
        raise ParseError(f"unexpected character: {ch!r}", self._line, self._col)

    def _scan_number(self) -> Token:
        ln, co = self._line, self._col; s = self._pos
        while self._current is not None and self._current.isdigit(): self._advance()
        if self._current == '.' and self._peek() is not None and self._peek().isdigit():
            self._advance()
            while self._current is not None and self._current.isdigit(): self._advance()
            return Token(TokenType.FLOAT_LIT, self._source[s:self._pos], ln, co)
        return Token(TokenType.INTEGER_LIT, self._source[s:self._pos], ln, co)

    def _scan_string(self) -> Token:
        ln, co = self._line, self._col; self._advance(); parts: list[str] = []
        while True:
            if self._current is None: raise ParseError("unterminated string", ln, co)
            if self._current == "'":
                if self._peek() == "'": parts.append("'"); self._advance(); self._advance()
                else: self._advance(); break
            else: parts.append(self._current); self._advance()
        return Token(TokenType.STRING, ''.join(parts), ln, co)

    def _scan_ident(self) -> Token:
        ln, co = self._line, self._col; s = self._pos
        while self._current is not None and (self._current.isalnum() or self._current == '_'): self._advance()
        text = self._source[s:self._pos]; upper = text.upper()
        if upper in KEYWORDS: return Token(KEYWORDS[upper], upper, ln, co)
        return Token(TokenType.IDENTIFIER, text.lower(), ln, co)
