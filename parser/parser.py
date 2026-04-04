from __future__ import annotations
"""Recursive-descent + Pratt expression parser."""
from typing import List, Optional
from parser.ast import (
    AggregateCall, AliasExpr, Assignment, BetweenExpr, BinaryExpr, CaseExpr,
    CastExpr, ColumnDef, ColumnRef, CreateTableStmt, DeleteStmt, DropTableStmt,
    FromClause, FunctionCall, GroupByClause, InExpr, InsertStmt, IsNullExpr,
    JoinClause, LikeExpr, Literal, SelectStmt, SortKey, StarExpr, SubqueryExpr,
    TableRef, TypeName, UnaryExpr, UpdateStmt,
)
from parser.precedence import Precedence
from parser.token import Token, TokenType
from storage.types import DataType
from utils.errors import ParseError

_UNRESERVED_KW = frozenset({
    TokenType.INT, TokenType.INTEGER, TokenType.BIGINT,
    TokenType.FLOAT_KW, TokenType.DOUBLE, TokenType.REAL,
    TokenType.BOOLEAN, TokenType.BOOL, TokenType.VARCHAR, TokenType.TEXT_KW,
    TokenType.DATE_KW, TokenType.TIMESTAMP, TokenType.FIRST, TokenType.LAST,
    TokenType.KEY,
})
_INFIX_PREC = {
    TokenType.OR: Precedence.OR, TokenType.AND: Precedence.AND,
    TokenType.EQUAL: Precedence.COMPARISON, TokenType.NOT_EQUAL: Precedence.COMPARISON,
    TokenType.LESS: Precedence.COMPARISON, TokenType.GREATER: Precedence.COMPARISON,
    TokenType.LESS_EQUAL: Precedence.COMPARISON, TokenType.GREATER_EQUAL: Precedence.COMPARISON,
    TokenType.IS: Precedence.IS, TokenType.IN: Precedence.COMPARISON,
    TokenType.BETWEEN: Precedence.COMPARISON, TokenType.LIKE: Precedence.COMPARISON,
    TokenType.PIPE_PIPE: Precedence.CONCAT,
    TokenType.PLUS: Precedence.ADDITION, TokenType.MINUS: Precedence.ADDITION,
    TokenType.STAR: Precedence.MULTIPLY, TokenType.SLASH: Precedence.MULTIPLY,
    TokenType.PERCENT: Precedence.MULTIPLY,
}
_AGG_NAMES = frozenset({'COUNT','SUM','AVG','MIN','MAX'})

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens; self._pos = 0; self._current = tokens[0]

    def _advance(self) -> None:
        self._pos += 1
        if self._pos < len(self._tokens): self._current = self._tokens[self._pos]

    def _peek(self) -> Token:
        n = self._pos + 1
        return self._tokens[n] if n < len(self._tokens) else self._tokens[-1]

    def _expect(self, tt: TokenType) -> Token:
        if self._current.type != tt:
            raise ParseError(f"expected {tt.value}, got {self._current.value!r}", self._current.line, self._current.col)
        tok = self._current; self._advance(); return tok

    def _expect_integer(self) -> int:
        return int(self._expect(TokenType.INTEGER_LIT).value)

    def _accept_identifier(self) -> str:
        if self._current.type == TokenType.IDENTIFIER:
            v = self._current.value; self._advance(); return v
        if self._current.type in _UNRESERVED_KW:
            v = self._current.value.lower(); self._advance(); return v
        raise ParseError(f"expected identifier, got {self._current.value!r}", self._current.line, self._current.col)

    def parse(self) -> object:
        stmt = self._parse_statement()
        if self._current.type == TokenType.SEMICOLON: self._advance()
        if self._current.type != TokenType.EOF:
            raise ParseError(f"extra tokens: {self._current.value!r}", self._current.line, self._current.col)
        return stmt

    def _parse_statement(self) -> object:
        tt = self._current.type
        if tt == TokenType.SELECT: return self._parse_select()
        if tt == TokenType.INSERT: return self._parse_insert()
        if tt == TokenType.UPDATE: return self._parse_update()
        if tt == TokenType.DELETE: return self._parse_delete()
        if tt == TokenType.CREATE: self._advance(); self._expect(TokenType.TABLE); return self._parse_create_table()
        if tt == TokenType.DROP: self._advance(); self._expect(TokenType.TABLE); return self._parse_drop_table()
        raise ParseError(f"unexpected: {self._current.value!r}", self._current.line, self._current.col)

    # -- SELECT --------------------------------------------------------
    def _parse_select(self) -> SelectStmt:
        self._expect(TokenType.SELECT)
        distinct = False
        if self._current.type == TokenType.DISTINCT: distinct = True; self._advance()
        sl = [self._parse_select_item()]
        while self._current.type == TokenType.COMMA: self._advance(); sl.append(self._parse_select_item())
        fc = self._parse_from() if self._current.type == TokenType.FROM else None
        w = None
        if self._current.type == TokenType.WHERE: self._advance(); w = self._parse_expression()
        gb = self._parse_group_by() if self._current.type == TokenType.GROUP else None
        hv = None
        if self._current.type == TokenType.HAVING: self._advance(); hv = self._parse_expression()
        ob: list = []
        if self._current.type == TokenType.ORDER: ob = self._parse_order_by()
        lm = None
        if self._current.type == TokenType.LIMIT: self._advance(); lm = self._parse_expression()
        of = None
        if self._current.type == TokenType.OFFSET: self._advance(); of = self._parse_expression()
        return SelectStmt(distinct=distinct, select_list=sl, from_clause=fc, where=w,
                          group_by=gb, having=hv, order_by=ob, limit=lm, offset=of)

    def _parse_select_item(self) -> object:
        expr = self._parse_expression()
        if self._current.type == TokenType.AS:
            self._advance(); return AliasExpr(expr=expr, alias=self._accept_identifier())
        if self._current.type == TokenType.IDENTIFIER:
            a = self._current.value; self._advance(); return AliasExpr(expr=expr, alias=a)
        return expr

    def _parse_from(self) -> FromClause:
        self._expect(TokenType.FROM)
        tref = self._parse_table_ref()
        joins: list = []
        while self._current.type in (TokenType.JOIN, TokenType.INNER, TokenType.LEFT,
                                      TokenType.RIGHT, TokenType.CROSS, TokenType.FULL, TokenType.COMMA):
            if self._current.type == TokenType.COMMA:
                self._advance()
                t2 = self._parse_table_ref()
                joins.append(JoinClause(join_type='CROSS', table=t2, on=None))
                continue
            joins.append(self._parse_join())
        return FromClause(table=tref, joins=joins)

    def _parse_table_ref(self) -> TableRef:
        if self._current.type == TokenType.LPAREN:
            # Subquery — not fully supported, skip for now
            pass
        name = self._accept_identifier()
        alias: Optional[str] = None
        if self._current.type == TokenType.AS:
            self._advance(); alias = self._accept_identifier()
        elif self._current.type == TokenType.IDENTIFIER:
            alias = self._current.value; self._advance()
        return TableRef(name=name, alias=alias)

    def _parse_join(self) -> JoinClause:
        jt = 'INNER'
        if self._current.type == TokenType.INNER: self._advance(); jt = 'INNER'
        elif self._current.type == TokenType.LEFT: self._advance(); jt = 'LEFT'
        elif self._current.type == TokenType.RIGHT: self._advance(); jt = 'RIGHT'
        elif self._current.type == TokenType.CROSS: self._advance(); jt = 'CROSS'
        elif self._current.type == TokenType.FULL: self._advance(); jt = 'FULL'
        if self._current.type == TokenType.OUTER: self._advance()
        self._expect(TokenType.JOIN)
        t = self._parse_table_ref()
        on = None
        if self._current.type == TokenType.ON:
            self._advance(); on = self._parse_expression()
        return JoinClause(join_type=jt, table=t, on=on)

    def _parse_group_by(self) -> GroupByClause:
        self._expect(TokenType.GROUP); self._expect(TokenType.BY)
        keys = [self._parse_expression()]
        while self._current.type == TokenType.COMMA: self._advance(); keys.append(self._parse_expression())
        return GroupByClause(keys=keys)

    def _parse_order_by(self) -> list:
        self._expect(TokenType.ORDER); self._expect(TokenType.BY)
        keys = [self._parse_sort_key()]
        while self._current.type == TokenType.COMMA: self._advance(); keys.append(self._parse_sort_key())
        return keys

    def _parse_sort_key(self) -> SortKey:
        expr = self._parse_expression()
        d = 'ASC'
        if self._current.type == TokenType.ASC: self._advance()
        elif self._current.type == TokenType.DESC: d = 'DESC'; self._advance()
        ns: Optional[str] = None
        if self._current.type == TokenType.NULLS:
            self._advance()
            if self._current.type == TokenType.FIRST: ns = 'NULLS_FIRST'; self._advance()
            elif self._current.type == TokenType.LAST: ns = 'NULLS_LAST'; self._advance()
            else: raise ParseError(f"expected FIRST/LAST", self._current.line, self._current.col)
        return SortKey(expr=expr, direction=d, nulls=ns)

    # -- INSERT --------------------------------------------------------
    def _parse_insert(self) -> InsertStmt:
        self._expect(TokenType.INSERT); self._expect(TokenType.INTO)
        table = self._accept_identifier()
        cols: Optional[List[str]] = None
        if self._current.type == TokenType.LPAREN:
            self._advance(); cols = [self._accept_identifier()]
            while self._current.type == TokenType.COMMA: self._advance(); cols.append(self._accept_identifier())
            self._expect(TokenType.RPAREN)
        self._expect(TokenType.VALUES)
        rows = [self._parse_value_row()]
        while self._current.type == TokenType.COMMA: self._advance(); rows.append(self._parse_value_row())
        return InsertStmt(table=table, columns=cols, values=rows)

    def _parse_value_row(self) -> list:
        self._expect(TokenType.LPAREN)
        exprs = [self._parse_expression()]
        while self._current.type == TokenType.COMMA: self._advance(); exprs.append(self._parse_expression())
        self._expect(TokenType.RPAREN); return exprs

    # -- UPDATE --------------------------------------------------------
    def _parse_update(self) -> UpdateStmt:
        self._expect(TokenType.UPDATE)
        table = self._accept_identifier()
        self._expect(TokenType.SET)
        assigns = [self._parse_assignment()]
        while self._current.type == TokenType.COMMA: self._advance(); assigns.append(self._parse_assignment())
        w = None
        if self._current.type == TokenType.WHERE: self._advance(); w = self._parse_expression()
        return UpdateStmt(table=table, assignments=assigns, where=w)

    def _parse_assignment(self) -> Assignment:
        col = self._accept_identifier()
        self._expect(TokenType.EQUAL)
        val = self._parse_expression()
        return Assignment(column=col, value=val)

    # -- DELETE --------------------------------------------------------
    def _parse_delete(self) -> DeleteStmt:
        self._expect(TokenType.DELETE); self._expect(TokenType.FROM)
        table = self._accept_identifier()
        w = None
        if self._current.type == TokenType.WHERE: self._advance(); w = self._parse_expression()
        return DeleteStmt(table=table, where=w)

    # -- CREATE / DROP -------------------------------------------------
    def _parse_create_table(self) -> CreateTableStmt:
        ine = False
        if self._current.type == TokenType.IF:
            self._advance(); self._expect(TokenType.NOT); self._expect(TokenType.EXISTS); ine = True
        table = self._accept_identifier(); self._expect(TokenType.LPAREN)
        cols = [self._parse_column_def()]
        while self._current.type == TokenType.COMMA: self._advance(); cols.append(self._parse_column_def())
        self._expect(TokenType.RPAREN)
        return CreateTableStmt(table=table, columns=cols, if_not_exists=ine)

    def _parse_column_def(self) -> ColumnDef:
        name = self._accept_identifier(); tn = self._parse_type_name()
        nullable = True; pk = False
        while True:
            if self._current.type == TokenType.NOT: self._advance(); self._expect(TokenType.NULL); nullable = False
            elif self._current.type == TokenType.NULL: self._advance(); nullable = True
            elif self._current.type == TokenType.PRIMARY: self._advance(); self._expect(TokenType.KEY); pk = True
            else: break
        return ColumnDef(name=name, type_name=tn, nullable=nullable, primary_key=pk)

    def _parse_type_name(self) -> TypeName:
        if self._current.type in _UNRESERVED_KW: n = self._current.value.upper(); self._advance()
        elif self._current.type == TokenType.IDENTIFIER: n = self._current.value.upper(); self._advance()
        else: raise ParseError(f"expected type name", self._current.line, self._current.col)
        params: list[int] = []
        if self._current.type == TokenType.LPAREN:
            self._advance(); params.append(self._expect_integer()); self._expect(TokenType.RPAREN)
        return TypeName(name=n, params=params)

    def _parse_drop_table(self) -> DropTableStmt:
        ie = False
        if self._current.type == TokenType.IF: self._advance(); self._expect(TokenType.EXISTS); ie = True
        return DropTableStmt(table=self._accept_identifier(), if_exists=ie)

    # ═══ Pratt expression parser ═══
    def _parse_expression(self, min_prec: int = Precedence.LOWEST) -> object:
        left = self._parse_prefix()
        while self._get_infix_prec() > min_prec:
            left = self._parse_infix(left)
        return left

    def _parse_prefix(self) -> object:
        tt = self._current.type
        if tt == TokenType.NOT: self._advance(); return UnaryExpr(op='NOT', operand=self._parse_expression(Precedence.NOT_PREFIX))
        if tt == TokenType.MINUS: self._advance(); return UnaryExpr(op='-', operand=self._parse_expression(Precedence.UNARY))
        if tt == TokenType.PLUS: self._advance(); return UnaryExpr(op='+', operand=self._parse_expression(Precedence.UNARY))
        if tt in (TokenType.INTEGER_LIT, TokenType.FLOAT_LIT, TokenType.STRING,
                  TokenType.TRUE, TokenType.FALSE, TokenType.NULL):
            return self._parse_literal()
        if tt == TokenType.IDENTIFIER or tt in _UNRESERVED_KW: return self._parse_identifier_expr()
        if tt == TokenType.CASE: return self._parse_case()
        if tt == TokenType.CAST: return self._parse_cast()
        if tt == TokenType.LPAREN:
            self._advance()
            # Check for subquery
            if self._current.type == TokenType.SELECT:
                sq = self._parse_select()
                self._expect(TokenType.RPAREN)
                return SubqueryExpr(query=sq)
            expr = self._parse_expression(); self._expect(TokenType.RPAREN); return expr
        if tt == TokenType.STAR: self._advance(); return StarExpr()
        raise ParseError(f"unexpected in expression: {self._current.value!r}", self._current.line, self._current.col)

    def _parse_literal(self) -> Literal:
        tok = self._current; self._advance()
        if tok.type == TokenType.INTEGER_LIT:
            v = int(tok.value)
            return Literal(value=v, inferred_type=DataType.INT if v <= 2_147_483_647 else DataType.BIGINT)
        if tok.type == TokenType.FLOAT_LIT: return Literal(value=float(tok.value), inferred_type=DataType.DOUBLE)
        if tok.type == TokenType.STRING: return Literal(value=tok.value, inferred_type=DataType.VARCHAR)
        if tok.type == TokenType.TRUE: return Literal(value=True, inferred_type=DataType.BOOLEAN)
        if tok.type == TokenType.FALSE: return Literal(value=False, inferred_type=DataType.BOOLEAN)
        if tok.type == TokenType.NULL: return Literal(value=None, inferred_type=DataType.UNKNOWN)
        raise ParseError(f"unexpected literal: {tok.type}", tok.line, tok.col)

    def _parse_identifier_expr(self) -> object:
        name = self._current.value
        if self._current.type in _UNRESERVED_KW: name = name.lower()
        self._advance()
        if self._current.type == TokenType.LPAREN: return self._parse_function_call(name)
        if self._current.type == TokenType.DOT:
            self._advance(); col = self._accept_identifier(); return ColumnRef(table=name, column=col)
        return ColumnRef(table=None, column=name)

    def _parse_function_call(self, name: str) -> object:
        self._expect(TokenType.LPAREN); upper = name.upper()
        if upper in _AGG_NAMES:
            if self._current.type == TokenType.STAR:
                self._advance(); self._expect(TokenType.RPAREN)
                return AggregateCall(name=upper, args=[StarExpr()])
            dist = False
            if self._current.type == TokenType.DISTINCT: dist = True; self._advance()
            args = [self._parse_expression()]; self._expect(TokenType.RPAREN)
            return AggregateCall(name=upper, args=args, distinct=dist)
        args: list = []
        if self._current.type != TokenType.RPAREN:
            args.append(self._parse_expression())
            while self._current.type == TokenType.COMMA: self._advance(); args.append(self._parse_expression())
        self._expect(TokenType.RPAREN)
        return FunctionCall(name=name, args=args)

    def _parse_case(self) -> CaseExpr:
        self._expect(TokenType.CASE)
        operand = None
        if self._current.type != TokenType.WHEN:
            operand = self._parse_expression()
        whens: list = []
        while self._current.type == TokenType.WHEN:
            self._advance(); cond = self._parse_expression()
            self._expect(TokenType.THEN); result = self._parse_expression()
            whens.append((cond, result))
        else_expr = None
        if self._current.type == TokenType.ELSE:
            self._advance(); else_expr = self._parse_expression()
        self._expect(TokenType.END)
        return CaseExpr(operand=operand, when_clauses=whens, else_expr=else_expr)

    def _parse_cast(self) -> CastExpr:
        self._expect(TokenType.CAST); self._expect(TokenType.LPAREN)
        expr = self._parse_expression()
        self._expect(TokenType.AS)
        tn = self._parse_type_name()
        self._expect(TokenType.RPAREN)
        return CastExpr(expr=expr, type_name=tn)

    # -- infix ---------------------------------------------------------
    def _get_infix_prec(self) -> int:
        tt = self._current.type
        if tt == TokenType.NOT:
            nxt = self._peek()
            if nxt.type in (TokenType.IN, TokenType.BETWEEN, TokenType.LIKE):
                return Precedence.COMPARISON
            return Precedence.LOWEST
        return _INFIX_PREC.get(tt, Precedence.LOWEST)

    def _parse_infix(self, left: object) -> object:
        tt = self._current.type
        if tt == TokenType.IS: return self._parse_is_expr(left)
        if tt == TokenType.IN: return self._parse_in_expr(left, False)
        if tt == TokenType.BETWEEN: return self._parse_between_expr(left, False)
        if tt == TokenType.LIKE: return self._parse_like_expr(left, False)
        if tt == TokenType.NOT:
            nxt = self._peek()
            if nxt.type == TokenType.IN: self._advance(); return self._parse_in_expr(left, True)
            if nxt.type == TokenType.BETWEEN: self._advance(); return self._parse_between_expr(left, True)
            if nxt.type == TokenType.LIKE: self._advance(); return self._parse_like_expr(left, True)
        prec = _INFIX_PREC[tt]
        op = self._current.value; self._advance()
        op_map = {'AND':'AND','OR':'OR','=':'=','!=':'!=','<':'<','>':'>','<=':'<=','>=':'>=',
                  '+':'+','-':'-','*':'*','/':'/','%':'%','||':'||'}
        right = self._parse_expression(prec)
        return BinaryExpr(op=op_map.get(op, op), left=left, right=right)

    def _parse_is_expr(self, left: object) -> IsNullExpr:
        self._expect(TokenType.IS); neg = False
        if self._current.type == TokenType.NOT: neg = True; self._advance()
        self._expect(TokenType.NULL)
        return IsNullExpr(expr=left, negated=neg)

    def _parse_in_expr(self, left: object, negated: bool) -> InExpr:
        self._expect(TokenType.IN); self._expect(TokenType.LPAREN)
        if self._current.type == TokenType.SELECT:
            sq = self._parse_select(); self._expect(TokenType.RPAREN)
            return InExpr(expr=left, values=[SubqueryExpr(query=sq)], negated=negated)
        vals = [self._parse_expression()]
        while self._current.type == TokenType.COMMA: self._advance(); vals.append(self._parse_expression())
        self._expect(TokenType.RPAREN)
        return InExpr(expr=left, values=vals, negated=negated)

    def _parse_between_expr(self, left: object, negated: bool) -> BetweenExpr:
        self._expect(TokenType.BETWEEN)
        low = self._parse_expression(Precedence.COMPARISON)
        self._expect(TokenType.AND)
        high = self._parse_expression(Precedence.COMPARISON)
        return BetweenExpr(expr=left, low=low, high=high, negated=negated)

    def _parse_like_expr(self, left: object, negated: bool) -> LikeExpr:
        self._expect(TokenType.LIKE)
        pattern = self._parse_expression(Precedence.COMPARISON)
        return LikeExpr(expr=left, pattern=pattern, negated=negated)
