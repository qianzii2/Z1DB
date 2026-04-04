from __future__ import annotations
"""Recursive-descent + Pratt expression parser."""

from typing import List, Optional

from parser.ast import (
    AggregateCall, AliasExpr, BinaryExpr, ColumnDef, ColumnRef,
    CreateTableStmt, DropTableStmt, FromClause, FunctionCall,
    GroupByClause, InsertStmt, IsNullExpr, Literal, SelectStmt,
    SortKey, StarExpr, TableRef, TypeName, UnaryExpr,
)
from parser.precedence import Precedence
from parser.token import Token, TokenType
from storage.types import DataType
from utils.errors import ParseError

# Keywords that may double as identifiers (unreserved)
_UNRESERVED_KEYWORDS = frozenset({
    TokenType.INT, TokenType.INTEGER, TokenType.BIGINT,
    TokenType.FLOAT_KW, TokenType.DOUBLE, TokenType.REAL,
    TokenType.BOOLEAN, TokenType.BOOL,
    TokenType.VARCHAR, TokenType.TEXT_KW,
    TokenType.DATE_KW, TokenType.TIMESTAMP,
    TokenType.FIRST, TokenType.LAST, TokenType.KEY,
})

# Infix precedence table
_INFIX_PREC = {
    TokenType.OR: Precedence.OR,
    TokenType.AND: Precedence.AND,
    TokenType.EQUAL: Precedence.COMPARISON,
    TokenType.NOT_EQUAL: Precedence.COMPARISON,
    TokenType.LESS: Precedence.COMPARISON,
    TokenType.GREATER: Precedence.COMPARISON,
    TokenType.LESS_EQUAL: Precedence.COMPARISON,
    TokenType.GREATER_EQUAL: Precedence.COMPARISON,
    TokenType.IS: Precedence.IS,
    TokenType.PIPE_PIPE: Precedence.CONCAT,
    TokenType.PLUS: Precedence.ADDITION,
    TokenType.MINUS: Precedence.ADDITION,
    TokenType.STAR: Precedence.MULTIPLY,
    TokenType.SLASH: Precedence.MULTIPLY,
    TokenType.PERCENT: Precedence.MULTIPLY,
}

_AGG_NAMES = frozenset({'COUNT', 'SUM', 'AVG', 'MIN', 'MAX'})


class Parser:
    """Turns a list of tokens into an AST."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._current: Token = tokens[0]

    # -- movement ------------------------------------------------------
    def _advance(self) -> None:
        self._pos += 1
        if self._pos < len(self._tokens):
            self._current = self._tokens[self._pos]

    def _peek(self) -> Token:
        nxt = self._pos + 1
        if nxt < len(self._tokens):
            return self._tokens[nxt]
        return self._tokens[-1]

    def _expect(self, tt: TokenType) -> Token:
        if self._current.type != tt:
            raise ParseError(
                f"expected {tt.value}, got {self._current.value!r}",
                self._current.line, self._current.col,
            )
        tok = self._current
        self._advance()
        return tok

    def _expect_integer(self) -> int:
        tok = self._expect(TokenType.INTEGER_LIT)
        return int(tok.value)

    def _accept_identifier(self) -> str:
        if self._current.type == TokenType.IDENTIFIER:
            val = self._current.value
            self._advance()
            return val
        if self._current.type in _UNRESERVED_KEYWORDS:
            val = self._current.value.lower()
            self._advance()
            return val
        raise ParseError(
            f"expected identifier, got {self._current.value!r}",
            self._current.line, self._current.col,
        )

    # -- entry point ---------------------------------------------------
    def parse(self) -> object:
        stmt = self._parse_statement()
        if self._current.type == TokenType.SEMICOLON:
            self._advance()
        if self._current.type != TokenType.EOF:
            raise ParseError(
                f"extra tokens after statement: {self._current.value!r}",
                self._current.line, self._current.col,
            )
        return stmt

    # -- statements ----------------------------------------------------
    def _parse_statement(self) -> object:
        tt = self._current.type
        if tt == TokenType.SELECT:
            return self._parse_select()
        if tt == TokenType.INSERT:
            return self._parse_insert()
        if tt == TokenType.CREATE:
            self._advance()
            self._expect(TokenType.TABLE)
            return self._parse_create_table()
        if tt == TokenType.DROP:
            self._advance()
            self._expect(TokenType.TABLE)
            return self._parse_drop_table()
        raise ParseError(
            f"unexpected token: {self._current.value!r}",
            self._current.line, self._current.col,
        )

    # -- SELECT --------------------------------------------------------
    def _parse_select(self) -> SelectStmt:
        self._expect(TokenType.SELECT)

        distinct = False
        if self._current.type == TokenType.DISTINCT:
            distinct = True
            self._advance()

        select_list = [self._parse_select_item()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            select_list.append(self._parse_select_item())

        from_clause: Optional[FromClause] = None
        if self._current.type == TokenType.FROM:
            from_clause = self._parse_from()

        where = None
        if self._current.type == TokenType.WHERE:
            self._advance()
            where = self._parse_expression()

        group_by: Optional[GroupByClause] = None
        if self._current.type == TokenType.GROUP:
            group_by = self._parse_group_by()

        having = None
        if self._current.type == TokenType.HAVING:
            self._advance()
            having = self._parse_expression()

        order_by: list = []
        if self._current.type == TokenType.ORDER:
            order_by = self._parse_order_by()

        limit = None
        if self._current.type == TokenType.LIMIT:
            self._advance()
            limit = self._parse_expression()

        offset = None
        if self._current.type == TokenType.OFFSET:
            self._advance()
            offset = self._parse_expression()

        return SelectStmt(
            distinct=distinct, select_list=select_list,
            from_clause=from_clause, where=where,
            group_by=group_by, having=having,
            order_by=order_by, limit=limit, offset=offset,
        )

    def _parse_select_item(self) -> object:
        expr = self._parse_expression()
        if self._current.type == TokenType.AS:
            self._advance()
            alias = self._accept_identifier()
            return AliasExpr(expr=expr, alias=alias)
        # implicit alias: identifier immediately following an expression
        if self._current.type == TokenType.IDENTIFIER:
            alias = self._current.value
            self._advance()
            return AliasExpr(expr=expr, alias=alias)
        return expr

    def _parse_from(self) -> FromClause:
        self._expect(TokenType.FROM)
        table_name = self._accept_identifier()
        alias: Optional[str] = None
        if self._current.type == TokenType.AS:
            self._advance()
            alias = self._accept_identifier()
        elif (self._current.type == TokenType.IDENTIFIER
              and self._current.type != TokenType.JOIN):
            alias = self._current.value
            self._advance()

        # Detect JOIN and give a clear error
        if self._current.type == TokenType.JOIN:
            raise ParseError(
                "JOIN is not yet supported",
                self._current.line, self._current.col,
            )

        return FromClause(table=TableRef(name=table_name, alias=alias), joins=[])

    def _parse_group_by(self) -> GroupByClause:
        self._expect(TokenType.GROUP)
        self._expect(TokenType.BY)
        keys = [self._parse_expression()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            keys.append(self._parse_expression())
        return GroupByClause(keys=keys)

    def _parse_order_by(self) -> list:
        self._expect(TokenType.ORDER)
        self._expect(TokenType.BY)
        keys = [self._parse_sort_key()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            keys.append(self._parse_sort_key())
        return keys

    def _parse_sort_key(self) -> SortKey:
        expr = self._parse_expression()
        direction = 'ASC'
        if self._current.type == TokenType.ASC:
            self._advance()
        elif self._current.type == TokenType.DESC:
            direction = 'DESC'
            self._advance()
        nulls: Optional[str] = None
        if self._current.type == TokenType.NULLS:
            self._advance()
            if self._current.type == TokenType.FIRST:
                nulls = 'NULLS_FIRST'
                self._advance()
            elif self._current.type == TokenType.LAST:
                nulls = 'NULLS_LAST'
                self._advance()
            else:
                raise ParseError(
                    f"expected FIRST or LAST after NULLS, got {self._current.value!r}",
                    self._current.line, self._current.col,
                )
        return SortKey(expr=expr, direction=direction, nulls=nulls)

    # -- INSERT --------------------------------------------------------
    def _parse_insert(self) -> InsertStmt:
        self._expect(TokenType.INSERT)
        self._expect(TokenType.INTO)
        table = self._accept_identifier()

        columns: Optional[List[str]] = None
        if self._current.type == TokenType.LPAREN:
            self._advance()
            columns = [self._accept_identifier()]
            while self._current.type == TokenType.COMMA:
                self._advance()
                columns.append(self._accept_identifier())
            self._expect(TokenType.RPAREN)

        self._expect(TokenType.VALUES)
        rows = [self._parse_value_row()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            rows.append(self._parse_value_row())

        return InsertStmt(table=table, columns=columns, values=rows)

    def _parse_value_row(self) -> list:
        self._expect(TokenType.LPAREN)
        exprs = [self._parse_expression()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            exprs.append(self._parse_expression())
        self._expect(TokenType.RPAREN)
        return exprs

    # -- CREATE TABLE --------------------------------------------------
    def _parse_create_table(self) -> CreateTableStmt:
        if_not_exists = False
        if self._current.type == TokenType.IF:
            self._advance()
            self._expect(TokenType.NOT)
            self._expect(TokenType.EXISTS)
            if_not_exists = True

        table = self._accept_identifier()
        self._expect(TokenType.LPAREN)
        cols = [self._parse_column_def()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            cols.append(self._parse_column_def())
        self._expect(TokenType.RPAREN)
        return CreateTableStmt(table=table, columns=cols, if_not_exists=if_not_exists)

    def _parse_column_def(self) -> ColumnDef:
        name = self._accept_identifier()
        type_name = self._parse_type_name()
        nullable = True
        primary_key = False
        while True:
            if self._current.type == TokenType.NOT:
                self._advance()
                self._expect(TokenType.NULL)
                nullable = False
            elif self._current.type == TokenType.NULL:
                self._advance()
                nullable = True
            elif self._current.type == TokenType.PRIMARY:
                self._advance()
                self._expect(TokenType.KEY)
                primary_key = True
            else:
                break
        return ColumnDef(name=name, type_name=type_name, nullable=nullable, primary_key=primary_key)

    def _parse_type_name(self) -> TypeName:
        if self._current.type in _UNRESERVED_KEYWORDS:
            name = self._current.value.upper()
            self._advance()
        elif self._current.type == TokenType.IDENTIFIER:
            name = self._current.value.upper()
            self._advance()
        else:
            raise ParseError(
                f"expected type name, got {self._current.value!r}",
                self._current.line, self._current.col,
            )
        params: list[int] = []
        if self._current.type == TokenType.LPAREN:
            self._advance()
            params.append(self._expect_integer())
            self._expect(TokenType.RPAREN)
        return TypeName(name=name, params=params)

    # -- DROP TABLE ----------------------------------------------------
    def _parse_drop_table(self) -> DropTableStmt:
        if_exists = False
        if self._current.type == TokenType.IF:
            self._advance()
            self._expect(TokenType.EXISTS)
            if_exists = True
        table = self._accept_identifier()
        return DropTableStmt(table=table, if_exists=if_exists)

    # ═══════════════════════════════════════════════════════════════════
    # Pratt expression parser
    # ═══════════════════════════════════════════════════════════════════

    def _parse_expression(self, min_prec: int = Precedence.LOWEST) -> object:
        left = self._parse_prefix()
        while self._get_infix_precedence() > min_prec:
            left = self._parse_infix(left)
        return left

    def _parse_prefix(self) -> object:
        tt = self._current.type
        if tt == TokenType.NOT:
            self._advance()
            operand = self._parse_expression(Precedence.NOT_PREFIX)
            return UnaryExpr(op='NOT', operand=operand)
        if tt == TokenType.MINUS:
            self._advance()
            operand = self._parse_expression(Precedence.UNARY)
            return UnaryExpr(op='-', operand=operand)
        if tt == TokenType.PLUS:
            self._advance()
            operand = self._parse_expression(Precedence.UNARY)
            return UnaryExpr(op='+', operand=operand)
        if tt in (TokenType.INTEGER_LIT, TokenType.FLOAT_LIT,
                  TokenType.STRING, TokenType.TRUE, TokenType.FALSE, TokenType.NULL):
            return self._parse_literal()
        if tt == TokenType.IDENTIFIER or tt in _UNRESERVED_KEYWORDS:
            return self._parse_identifier_expr()
        if tt == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr
        if tt == TokenType.STAR:
            self._advance()
            return StarExpr()
        raise ParseError(
            f"unexpected token in expression: {self._current.value!r}",
            self._current.line, self._current.col,
        )

    def _parse_literal(self) -> Literal:
        tok = self._current
        self._advance()
        if tok.type == TokenType.INTEGER_LIT:
            val = int(tok.value)
            if val <= 2_147_483_647:
                return Literal(value=val, inferred_type=DataType.INT)
            return Literal(value=val, inferred_type=DataType.BIGINT)
        if tok.type == TokenType.FLOAT_LIT:
            return Literal(value=float(tok.value), inferred_type=DataType.DOUBLE)
        if tok.type == TokenType.STRING:
            return Literal(value=tok.value, inferred_type=DataType.VARCHAR)
        if tok.type == TokenType.TRUE:
            return Literal(value=True, inferred_type=DataType.BOOLEAN)
        if tok.type == TokenType.FALSE:
            return Literal(value=False, inferred_type=DataType.BOOLEAN)
        if tok.type == TokenType.NULL:
            return Literal(value=None, inferred_type=DataType.UNKNOWN)
        raise ParseError(f"unexpected literal type: {tok.type}", tok.line, tok.col)

    def _parse_identifier_expr(self) -> object:
        name = self._current.value
        if self._current.type in _UNRESERVED_KEYWORDS:
            name = name.lower()
        self._advance()
        if self._current.type == TokenType.LPAREN:
            return self._parse_function_call(name)
        if self._current.type == TokenType.DOT:
            self._advance()
            col = self._accept_identifier()
            return ColumnRef(table=name, column=col)
        return ColumnRef(table=None, column=name)

    def _parse_function_call(self, name: str) -> object:
        self._expect(TokenType.LPAREN)
        upper = name.upper()
        if upper in _AGG_NAMES:
            if self._current.type == TokenType.STAR:
                self._advance()
                self._expect(TokenType.RPAREN)
                return AggregateCall(name=upper, args=[StarExpr()])
            distinct = False
            if self._current.type == TokenType.DISTINCT:
                distinct = True
                self._advance()
            args = [self._parse_expression()]
            self._expect(TokenType.RPAREN)
            return AggregateCall(name=upper, args=args, distinct=distinct)
        # Scalar function
        args: list = []
        if self._current.type != TokenType.RPAREN:
            args.append(self._parse_expression())
            while self._current.type == TokenType.COMMA:
                self._advance()
                args.append(self._parse_expression())
        self._expect(TokenType.RPAREN)
        return FunctionCall(name=name, args=args)

    # -- infix ---------------------------------------------------------
    def _get_infix_precedence(self) -> int:
        return _INFIX_PREC.get(self._current.type, Precedence.LOWEST)

    def _parse_infix(self, left: object) -> object:
        tt = self._current.type
        if tt == TokenType.IS:
            return self._parse_is_expression(left)
        prec = _INFIX_PREC[tt]
        op_value = self._current.value
        self._advance()
        op_map = {
            'AND': 'AND', 'OR': 'OR',
            '=': '=', '!=': '!=', '<': '<', '>': '>',
            '<=': '<=', '>=': '>=',
            '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
            '||': '||',
        }
        op = op_map.get(op_value, op_value)
        right = self._parse_expression(prec)
        return BinaryExpr(op=op, left=left, right=right)

    def _parse_is_expression(self, left: object) -> IsNullExpr:
        self._expect(TokenType.IS)
        negated = False
        if self._current.type == TokenType.NOT:
            negated = True
            self._advance()
        self._expect(TokenType.NULL)
        return IsNullExpr(expr=left, negated=negated)
