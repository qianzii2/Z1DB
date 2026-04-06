from __future__ import annotations
"""Pratt 表达式解析器。
[RP1] 通过 ParserInterface 协议与 Parser 交互，不直接访问私有属性。"""
from typing import Any, List, Optional, Protocol, runtime_checkable
from parser.ast import *
from parser.precedence import Precedence
from parser.token import Token, TokenType
from storage.types import DataType

_AGGS = frozenset({
    'COUNT','SUM','AVG','MIN','MAX','STDDEV','STDDEV_POP',
    'VARIANCE','VAR_POP','MEDIAN','MODE','ARRAY_AGG','STRING_AGG',
    'COUNT_DISTINCT','SUM_DISTINCT','AVG_DISTINCT',
    'PERCENTILE_CONT','PERCENTILE_DISC',
    'APPROX_COUNT_DISTINCT','APPROX_PERCENTILE','APPROX_TOP_K','GROUPING'})
_WINFUNCS = frozenset({
    'ROW_NUMBER','RANK','DENSE_RANK','NTILE','PERCENT_RANK',
    'CUME_DIST','LAG','LEAD','FIRST_VALUE','LAST_VALUE','NTH_VALUE'})
_NOARG_FUNCS = frozenset({'current_date','current_timestamp','now'})

_INFIX = {
    TokenType.OR: Precedence.OR, TokenType.AND: Precedence.AND,
    TokenType.EQUAL: Precedence.COMPARISON, TokenType.NOT_EQUAL: Precedence.COMPARISON,
    TokenType.LESS: Precedence.COMPARISON, TokenType.GREATER: Precedence.COMPARISON,
    TokenType.LESS_EQUAL: Precedence.COMPARISON, TokenType.GREATER_EQUAL: Precedence.COMPARISON,
    TokenType.IS: Precedence.IS, TokenType.IN: Precedence.COMPARISON,
    TokenType.BETWEEN: Precedence.COMPARISON, TokenType.LIKE: Precedence.COMPARISON,
    TokenType.PIPE_PIPE: Precedence.CONCAT,
    TokenType.PLUS: Precedence.ADDITION, TokenType.MINUS: Precedence.ADDITION,
    TokenType.STAR: Precedence.MULTIPLY, TokenType.SLASH: Precedence.MULTIPLY,
    TokenType.PERCENT: Precedence.MULTIPLY}

_OP_MAP = {
    'AND':'AND','OR':'OR','=':'=','!=':'!=','<':'<','>':'>',
    '<=':'<=','>=':'>=','+':'+','-':'-','*':'*','/':'/','%':'%','||':'||'}

_UNRESERVED = frozenset({
    TokenType.INT, TokenType.INTEGER, TokenType.BIGINT, TokenType.FLOAT_KW,
    TokenType.DOUBLE, TokenType.REAL, TokenType.BOOLEAN, TokenType.BOOL,
    TokenType.VARCHAR, TokenType.TEXT_KW, TokenType.DATE_KW, TokenType.TIMESTAMP,
    TokenType.FIRST, TokenType.LAST, TokenType.KEY, TokenType.PARTITION,
    TokenType.ROWS, TokenType.RANGE, TokenType.UNBOUNDED,
    TokenType.PRECEDING, TokenType.FOLLOWING, TokenType.CURRENT,
    TokenType.ROW, TokenType.OVER, TokenType.COLUMN, TokenType.RENAME,
    TokenType.TO, TokenType.ADD, TokenType.INDEX, TokenType.UNIQUE,
    TokenType.USING, TokenType.NATURAL})


@runtime_checkable
class ParserInterface(Protocol):
    """[RP1] Parser 暴露给 ExprParser 的正式接口。"""
    def advance(self) -> None: ...
    def current_token(self) -> Token: ...
    def peek_token(self) -> Token: ...
    def expect(self, tt: TokenType) -> Token: ...
    def expect_int(self) -> int: ...
    def accept_identifier(self) -> str: ...
    def parse_type_name(self) -> Any: ...
    def parse_select(self) -> Any: ...


class ExprParser:
    """Pratt 表达式解析器。通过 ParserInterface 与 Parser 交互。"""

    def __init__(self, parser: ParserInterface) -> None:
        self._p = parser

    def parse_expression(self, min_prec=Precedence.LOWEST):
        left = self._parse_prefix()
        while self._infix_precedence() > min_prec:
            left = self._parse_infix(left)
        return left

    def parse_sort_key(self):
        expr = self.parse_expression()
        direction = 'ASC'
        if self._p.current_token().type == TokenType.ASC: self._p.advance()
        elif self._p.current_token().type == TokenType.DESC: direction = 'DESC'; self._p.advance()
        nulls = None
        if self._p.current_token().type == TokenType.NULLS:
            self._p.advance()
            if self._p.current_token().type == TokenType.FIRST: nulls = 'NULLS_FIRST'; self._p.advance()
            elif self._p.current_token().type == TokenType.LAST: nulls = 'NULLS_LAST'; self._p.advance()
        return SortKey(expr=expr, direction=direction, nulls=nulls)

    # ═══ 前缀 ═══

    def _parse_prefix(self):
        tt = self._p.current_token().type
        if tt == TokenType.NOT:
            self._p.advance()
            return UnaryExpr(op='NOT', operand=self.parse_expression(Precedence.NOT_PREFIX))
        if tt == TokenType.MINUS:
            self._p.advance()
            return UnaryExpr(op='-', operand=self.parse_expression(Precedence.UNARY))
        if tt == TokenType.PLUS:
            self._p.advance()
            return UnaryExpr(op='+', operand=self.parse_expression(Precedence.UNARY))
        if tt in (TokenType.INTEGER_LIT, TokenType.FLOAT_LIT, TokenType.STRING,
                  TokenType.TRUE, TokenType.FALSE, TokenType.NULL):
            return self._parse_literal()
        if tt == TokenType.IDENTIFIER or tt in _UNRESERVED:
            return self._parse_identifier_expr()
        if tt == TokenType.CASE: return self._parse_case()
        if tt == TokenType.CAST: return self._parse_cast()
        if tt == TokenType.EXISTS:
            self._p.advance(); self._p.expect(TokenType.LPAREN)
            sq = self._p.parse_select(); self._p.expect(TokenType.RPAREN)
            return ExistsExpr(query=sq)
        if tt == TokenType.LPAREN:
            self._p.advance()
            if self._p.current_token().type == TokenType.SELECT:
                sq = self._p.parse_select(); self._p.expect(TokenType.RPAREN)
                return SubqueryExpr(query=sq)
            e = self.parse_expression(); self._p.expect(TokenType.RPAREN); return e
        if tt == TokenType.STAR: self._p.advance(); return StarExpr()
        from utils.errors import ParseError
        t = self._p.current_token()
        raise ParseError(f"意外: {t.value!r}", t.line, t.col)

    def _parse_literal(self):
        t = self._p.current_token(); self._p.advance()
        if t.type == TokenType.INTEGER_LIT:
            v = int(t.value)
            return Literal(value=v, inferred_type=DataType.INT if v <= 2_147_483_647 else DataType.BIGINT)
        if t.type == TokenType.FLOAT_LIT: return Literal(value=float(t.value), inferred_type=DataType.DOUBLE)
        if t.type == TokenType.STRING: return Literal(value=t.value, inferred_type=DataType.VARCHAR)
        if t.type == TokenType.TRUE: return Literal(value=True, inferred_type=DataType.BOOLEAN)
        if t.type == TokenType.FALSE: return Literal(value=False, inferred_type=DataType.BOOLEAN)
        return Literal(value=None, inferred_type=DataType.UNKNOWN)

    def _parse_identifier_expr(self):
        name = self._p.current_token().value
        if self._p.current_token().type in _UNRESERVED: name = name.lower()
        self._p.advance()
        if name.lower() in _NOARG_FUNCS:
            if self._p.current_token().type == TokenType.LPAREN:
                self._p.advance(); self._p.expect(TokenType.RPAREN)
            return FunctionCall(name=name.upper(), args=[])
        if self._p.current_token().type == TokenType.LPAREN:
            return self._parse_function_call(name)
        if self._p.current_token().type == TokenType.DOT:
            self._p.advance(); return ColumnRef(table=name, column=self._p.accept_identifier())
        return ColumnRef(table=None, column=name)

    def _parse_function_call(self, name):
        self._p.expect(TokenType.LPAREN); upper = name.upper()
        if upper in _AGGS:
            if self._p.current_token().type == TokenType.STAR:
                self._p.advance(); self._p.expect(TokenType.RPAREN)
                return self._maybe_window(AggregateCall(name=upper, args=[StarExpr()]))
            distinct = False
            if self._p.current_token().type == TokenType.DISTINCT: distinct = True; self._p.advance()
            args = self._parse_arg_list()
            self._p.expect(TokenType.RPAREN)
            return self._maybe_window(AggregateCall(name=upper, args=args, distinct=distinct))
        if upper in _WINFUNCS:
            args = self._parse_arg_list() if self._p.current_token().type != TokenType.RPAREN else []
            self._p.expect(TokenType.RPAREN)
            return self._maybe_window(FunctionCall(name=upper, args=args))
        args = self._parse_arg_list() if self._p.current_token().type != TokenType.RPAREN else []
        self._p.expect(TokenType.RPAREN)
        return self._maybe_window(FunctionCall(name=name, args=args))

    def _parse_arg_list(self) -> list:
        """[RP2] 统一参数列表解析。"""
        args = [self.parse_expression()]
        while self._p.current_token().type == TokenType.COMMA:
            self._p.advance(); args.append(self.parse_expression())
        return args

    def _maybe_window(self, func):
        if self._p.current_token().type != TokenType.OVER: return func
        self._p.expect(TokenType.OVER); self._p.expect(TokenType.LPAREN)
        pb = []
        if self._p.current_token().type == TokenType.PARTITION:
            self._p.advance(); self._p.expect(TokenType.BY)
            pb = self._parse_arg_list()
        oby = []
        if self._p.current_token().type == TokenType.ORDER:
            self._p.expect(TokenType.ORDER); self._p.expect(TokenType.BY)
            oby.append(self.parse_sort_key())
            while self._p.current_token().type == TokenType.COMMA:
                self._p.advance(); oby.append(self.parse_sort_key())
        frame = None
        if self._p.current_token().type in (TokenType.ROWS, TokenType.RANGE):
            frame = self._parse_frame()
        self._p.expect(TokenType.RPAREN)
        return WindowCall(func=func, partition_by=pb, order_by=oby, frame=frame)

    def _parse_frame(self):
        mode = self._p.current_token().value; self._p.advance()
        if self._p.current_token().type == TokenType.BETWEEN:
            self._p.advance(); s = self._parse_bound()
            self._p.expect(TokenType.AND); e = self._parse_bound()
            return WindowFrame(mode=mode, start=s, end=e)
        return WindowFrame(mode=mode, start=self._parse_bound(),
                           end=FrameBound(type='CURRENT_ROW'))

    def _parse_bound(self):
        if self._p.current_token().type == TokenType.UNBOUNDED:
            self._p.advance()
            if self._p.current_token().type == TokenType.PRECEDING:
                self._p.advance(); return FrameBound(type='UNBOUNDED_PRECEDING')
            self._p.expect(TokenType.FOLLOWING); return FrameBound(type='UNBOUNDED_FOLLOWING')
        if self._p.current_token().type == TokenType.CURRENT:
            self._p.advance(); self._p.expect(TokenType.ROW); return FrameBound(type='CURRENT_ROW')
        if self._p.current_token().type == TokenType.INTEGER_LIT:
            n = int(self._p.current_token().value); self._p.advance()
            if self._p.current_token().type == TokenType.PRECEDING:
                self._p.advance(); return FrameBound(type='N_PRECEDING', offset=n)
            self._p.expect(TokenType.FOLLOWING); return FrameBound(type='N_FOLLOWING', offset=n)
        from utils.errors import ParseError
        t = self._p.current_token()
        raise ParseError(f"期望帧边界", t.line, t.col)

    def _parse_case(self):
        self._p.expect(TokenType.CASE)
        op = None
        if self._p.current_token().type != TokenType.WHEN: op = self.parse_expression()
        ws = []
        while self._p.current_token().type == TokenType.WHEN:
            self._p.advance(); c = self.parse_expression()
            self._p.expect(TokenType.THEN); r = self.parse_expression()
            ws.append((c, r))
        el = None
        if self._p.current_token().type == TokenType.ELSE:
            self._p.advance(); el = self.parse_expression()
        self._p.expect(TokenType.END)
        return CaseExpr(operand=op, when_clauses=ws, else_expr=el)

    def _parse_cast(self):
        self._p.expect(TokenType.CAST); self._p.expect(TokenType.LPAREN)
        e = self.parse_expression(); self._p.expect(TokenType.AS)
        t = self._p.parse_type_name(); self._p.expect(TokenType.RPAREN)
        return CastExpr(expr=e, type_name=t)

    # ═══ 中缀 ═══

    def _infix_precedence(self):
        tt = self._p.current_token().type
        if tt == TokenType.NOT:
            nxt = self._p.peek_token()
            if nxt.type in (TokenType.IN, TokenType.BETWEEN, TokenType.LIKE):
                return Precedence.COMPARISON
            return Precedence.LOWEST
        return _INFIX.get(tt, Precedence.LOWEST)

    def _parse_infix(self, left):
        tt = self._p.current_token().type
        if tt == TokenType.IS: return self._parse_is(left)
        if tt == TokenType.IN: return self._parse_in(left, False)
        if tt == TokenType.BETWEEN: return self._parse_between(left, False)
        if tt == TokenType.LIKE: return self._parse_like(left, False)
        if tt == TokenType.NOT:
            nxt = self._p.peek_token()
            if nxt.type == TokenType.IN: self._p.advance(); return self._parse_in(left, True)
            if nxt.type == TokenType.BETWEEN: self._p.advance(); return self._parse_between(left, True)
            if nxt.type == TokenType.LIKE: self._p.advance(); return self._parse_like(left, True)
        prec = _INFIX[tt]; op = self._p.current_token().value; self._p.advance()
        return BinaryExpr(op=_OP_MAP.get(op, op), left=left, right=self.parse_expression(prec))

    def _parse_is(self, left):
        self._p.expect(TokenType.IS); neg = False
        if self._p.current_token().type == TokenType.NOT: neg = True; self._p.advance()
        self._p.expect(TokenType.NULL)
        return IsNullExpr(expr=left, negated=neg)

    def _parse_in(self, left, negated):
        self._p.expect(TokenType.IN); self._p.expect(TokenType.LPAREN)
        if self._p.current_token().type == TokenType.SELECT:
            sq = self._p.parse_select(); self._p.expect(TokenType.RPAREN)
            return InExpr(expr=left, values=[SubqueryExpr(query=sq)], negated=negated)
        vs = self._parse_arg_list()
        self._p.expect(TokenType.RPAREN)
        return InExpr(expr=left, values=vs, negated=negated)

    def _parse_between(self, left, negated):
        self._p.expect(TokenType.BETWEEN)
        lo = self.parse_expression(Precedence.COMPARISON)
        self._p.expect(TokenType.AND)
        return BetweenExpr(expr=left, low=lo,
                           high=self.parse_expression(Precedence.COMPARISON),
                           negated=negated)

    def _parse_like(self, left, negated):
        self._p.expect(TokenType.LIKE)
        return LikeExpr(expr=left,
                        pattern=self.parse_expression(Precedence.COMPARISON),
                        negated=negated)
