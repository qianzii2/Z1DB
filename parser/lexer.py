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
    'OVER':TokenType.OVER,'PARTITION':TokenType.PARTITION,'ROWS':TokenType.ROWS,
    'RANGE':TokenType.RANGE,'UNBOUNDED':TokenType.UNBOUNDED,
    'PRECEDING':TokenType.PRECEDING,'FOLLOWING':TokenType.FOLLOWING,
    'CURRENT':TokenType.CURRENT,'ROW':TokenType.ROW,
    'INT':TokenType.INT,'INTEGER':TokenType.INTEGER,'BIGINT':TokenType.BIGINT,
    'FLOAT':TokenType.FLOAT_KW,'DOUBLE':TokenType.DOUBLE,'REAL':TokenType.REAL,
    'BOOLEAN':TokenType.BOOLEAN,'BOOL':TokenType.BOOL,
    'VARCHAR':TokenType.VARCHAR,'TEXT':TokenType.TEXT_KW,
    'DATE':TokenType.DATE_KW,'TIMESTAMP':TokenType.TIMESTAMP,
}
_SINGLE: Dict[str, TokenType] = {
    '+':TokenType.PLUS,'-':TokenType.MINUS,'*':TokenType.STAR,
    '/':TokenType.SLASH,'%':TokenType.PERCENT,'=':TokenType.EQUAL,
    '<':TokenType.LESS,'>':TokenType.GREATER,
    '(':TokenType.LPAREN,')':TokenType.RPAREN,
    ',':TokenType.COMMA,'.':TokenType.DOT,';':TokenType.SEMICOLON,
}

class Lexer:
    def __init__(self, source: str) -> None:
        self._s = source; self._p = 0; self._ln = 1; self._co = 1
        self._c: Optional[str] = source[0] if source else None
    def tokenize(self) -> list[Token]:
        t: list[Token] = []
        while True:
            self._skip()
            if self._c is None: break
            t.append(self._next())
        t.append(Token(TokenType.EOF,'',self._ln,self._co)); return t
    def _adv(self) -> None:
        if self._c == '\n': self._ln += 1; self._co = 1
        else: self._co += 1
        self._p += 1; self._c = self._s[self._p] if self._p < len(self._s) else None
    def _pk(self) -> Optional[str]:
        n = self._p+1; return self._s[n] if n < len(self._s) else None
    def _skip(self) -> None:
        while self._c is not None:
            if self._c.isspace(): self._adv()
            elif self._c == '-' and self._pk() == '-':
                while self._c is not None and self._c != '\n': self._adv()
            elif self._c == '/' and self._pk() == '*':
                self._adv(); self._adv(); d = 1
                while self._c is not None and d > 0:
                    if self._c == '/' and self._pk() == '*': d += 1; self._adv()
                    elif self._c == '*' and self._pk() == '/': d -= 1; self._adv()
                    self._adv()
                if d > 0: raise ParseError("unterminated comment",self._ln,self._co)
            else: break
    def _next(self) -> Token:
        ch = self._c; assert ch
        if ch.isdigit(): return self._num()
        if ch.isalpha() or ch == '_': return self._ident()
        if ch == "'": return self._str()
        nx = self._pk(); pr = ch+(nx or ''); ln,co = self._ln,self._co
        if pr == '<=': self._adv();self._adv(); return Token(TokenType.LESS_EQUAL,'<=',ln,co)
        if pr == '>=': self._adv();self._adv(); return Token(TokenType.GREATER_EQUAL,'>=',ln,co)
        if pr == '!=': self._adv();self._adv(); return Token(TokenType.NOT_EQUAL,'!=',ln,co)
        if pr == '<>': self._adv();self._adv(); return Token(TokenType.NOT_EQUAL,'!=',ln,co)
        if pr == '||': self._adv();self._adv(); return Token(TokenType.PIPE_PIPE,'||',ln,co)
        if ch in _SINGLE: tt = _SINGLE[ch]; self._adv(); return Token(tt,ch,ln,co)
        raise ParseError(f"unexpected: {ch!r}",self._ln,self._co)
    def _num(self) -> Token:
        ln,co = self._ln,self._co; s = self._p
        while self._c is not None and self._c.isdigit(): self._adv()
        if self._c == '.' and self._pk() is not None and self._pk().isdigit():
            self._adv()
            while self._c is not None and self._c.isdigit(): self._adv()
            return Token(TokenType.FLOAT_LIT,self._s[s:self._p],ln,co)
        return Token(TokenType.INTEGER_LIT,self._s[s:self._p],ln,co)
    def _str(self) -> Token:
        ln,co = self._ln,self._co; self._adv(); p: list[str] = []
        while True:
            if self._c is None: raise ParseError("unterminated string",ln,co)
            if self._c == "'":
                if self._pk() == "'": p.append("'");self._adv();self._adv()
                else: self._adv(); break
            else: p.append(self._c); self._adv()
        return Token(TokenType.STRING,''.join(p),ln,co)
    def _ident(self) -> Token:
        ln,co = self._ln,self._co; s = self._p
        while self._c is not None and (self._c.isalnum() or self._c == '_'): self._adv()
        t = self._s[s:self._p]; u = t.upper()
        if u in KEYWORDS: return Token(KEYWORDS[u],u,ln,co)
        return Token(TokenType.IDENTIFIER,t.lower(),ln,co)
