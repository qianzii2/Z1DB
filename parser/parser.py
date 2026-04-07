from __future__ import annotations
"""SQL 解析器 — 语句级解析。表达式解析委托到 expr_parser.py。
[B07] CTE 列名解析安全性改进。
[M05] 新增 VACUUM 语句解析。"""
from typing import List, Optional
from parser.ast import *
from parser.precedence import Precedence
from parser.expr_parser import ExprParser, _UNRESERVED
from parser.token import Token, TokenType
from storage.types import DataType
from utils.errors import ParseError


class Parser:
    """SQL 解析器。将 Token 序列转换为 AST。"""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._current = tokens[0]
        self._expr = ExprParser(self)

    # ═══ 基础方法 ═══

    def _advance(self):
        self._pos += 1
        self._current = (
            self._tokens[self._pos]
            if self._pos < len(self._tokens)
            else self._tokens[-1])

    # ═══ ParserInterface 实现（供 ExprParser 回调）═══

    def advance(self) -> None:
        self._advance()

    def current_token(self) -> Token:
        return self._current

    def peek_token(self) -> Token:
        return self._peek()

    def expect(self, tt: TokenType) -> Token:
        return self._expect(tt)

    def expect_int(self) -> int:
        return self._expect_int()

    def accept_identifier(self) -> str:
        return self._accept_identifier()

    def parse_type_name(self) -> Any:
        return self._parse_type_name()

    def parse_select(self) -> Any:
        return self._parse_select()

    def _peek(self) -> Token:
        n = self._pos + 1
        return (self._tokens[n]
                if n < len(self._tokens)
                else self._tokens[-1])

    def _expect(self, tt) -> Token:
        if self._current.type != tt:
            raise ParseError(
                f"期望 {tt.value}，实际 {self._current.value!r}",
                self._current.line, self._current.col)
        t = self._current
        self._advance()
        return t

    def _expect_int(self) -> int:
        return int(self._expect(TokenType.INTEGER_LIT).value)

    def _accept_identifier(self) -> str:
        if self._current.type == TokenType.IDENTIFIER:
            v = self._current.value
            self._advance()
            return v
        if self._current.type in _UNRESERVED:
            v = self._current.value.lower()
            self._advance()
            return v
        raise ParseError(
            f"期望标识符，实际 {self._current.value!r}",
            self._current.line, self._current.col)

    def _parse_type_name(self):
        if (self._current.type in _UNRESERVED
                or self._current.type == TokenType.IDENTIFIER):
            n = self._current.value.upper()
            self._advance()
        else:
            raise ParseError(
                f"期望类型名",
                self._current.line, self._current.col)
        p = []
        if self._current.type == TokenType.LPAREN:
            self._advance()
            p.append(self._expect_int())
            self._expect(TokenType.RPAREN)
        return TypeName(name=n, params=p)

    # ═══ 顶层入口 ═══

    def parse(self):
        stmt = self._parse_statement()
        while self._current.type in (
                TokenType.UNION, TokenType.INTERSECT,
                TokenType.EXCEPT):
            stmt = self._parse_set_operation(stmt)
        # Handle ORDER BY / LIMIT after set operations
        if isinstance(stmt, SetOperationStmt):
            if self._current.type in (TokenType.ORDER, TokenType.LIMIT, TokenType.OFFSET):
                order_by = []
                limit = None
                offset = None
                if self._current.type == TokenType.ORDER:
                    order_by = self._parse_order_by()
                if self._current.type == TokenType.LIMIT:
                    self._advance()
                    limit = self._expr.parse_expression()
                if self._current.type == TokenType.OFFSET:
                    self._advance()
                    offset = self._expr.parse_expression()
                # Wrap: SELECT * FROM (set_op) __setop ORDER BY ... LIMIT ...
                from parser.ast import FromClause, TableRef
                stmt = SelectStmt(
                    select_list=[StarExpr()],
                    from_clause=FromClause(
                        table=TableRef(name='__setop', alias='__setop', subquery=stmt),
                        joins=[]),
                    order_by=order_by,
                    limit=limit,
                    offset=offset,
                )
        if self._current.type == TokenType.SEMICOLON:
            self._advance()
        if self._current.type != TokenType.EOF:
            raise ParseError(
                f"多余 token: {self._current.value!r}",
                self._current.line, self._current.col)
        return stmt

    def _parse_statement(self):
        tt = self._current.type
        if tt == TokenType.WITH or (
                tt == TokenType.IDENTIFIER
                and self._current.value == 'with'):
            return self._parse_with_cte()
        if tt == TokenType.SELECT:
            return self._parse_select()
        if tt == TokenType.INSERT:
            return self._parse_insert()
        if tt == TokenType.UPDATE:
            return self._parse_update()
        if tt == TokenType.DELETE:
            return self._parse_delete()
        if tt == TokenType.COPY:
            return self._parse_copy()
        if tt == TokenType.VACUUM:
            return self._parse_vacuum()
        if tt == TokenType.CREATE:
            self._advance()
            if self._current.type == TokenType.INDEX:
                self._advance()
                return self._parse_create_index(False)
            if self._current.type == TokenType.UNIQUE:
                self._advance()
                self._expect(TokenType.INDEX)
                return self._parse_create_index(True)
            self._expect(TokenType.TABLE)
            return self._parse_create_table()
        if tt == TokenType.DROP:
            self._advance()
            if self._current.type == TokenType.INDEX:
                self._advance()
                return self._parse_drop_index()
            self._expect(TokenType.TABLE)
            return self._parse_drop_table()
        if tt == TokenType.EXPLAIN:
            return self._parse_explain()
        if tt == TokenType.ALTER:
            return self._parse_alter()
        raise ParseError(
            f"意外: {self._current.value!r}",
            self._current.line, self._current.col)

    # ═══ SET 操作 ═══

    def _parse_set_operation(self, left):
        op = self._current.value
        self._advance()
        all_ = False
        if self._current.type == TokenType.ALL:
            all_ = True
            self._advance()
        right = self._parse_select_no_order()
        while self._current.type in (
                TokenType.UNION, TokenType.INTERSECT,
                TokenType.EXCEPT):
            right = self._parse_set_operation(right)
        return SetOperationStmt(
            op=op, all=all_, left=left, right=right)

    def _parse_select_no_order(self):
        """Parse SELECT without trailing ORDER BY/LIMIT (for set operations)."""
        self._expect(TokenType.SELECT)
        distinct = False
        if self._current.type == TokenType.DISTINCT:
            distinct = True
            self._advance()
        select_list = [self._parse_select_item()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            select_list.append(self._parse_select_item())
        from_clause = (self._parse_from()
                       if self._current.type == TokenType.FROM
                       else None)
        where = None
        if self._current.type == TokenType.WHERE:
            self._advance()
            where = self._expr.parse_expression()
        group_by = (self._parse_group_by()
                    if self._current.type == TokenType.GROUP
                    else None)
        having = None
        if self._current.type == TokenType.HAVING:
            self._advance()
            having = self._expr.parse_expression()
        # No ORDER BY / LIMIT here — they apply to the set operation
        return SelectStmt(
            distinct=distinct, select_list=select_list,
            from_clause=from_clause, where=where,
            group_by=group_by, having=having)

    # ═══ SELECT ═══

    def _parse_select(self):
        self._expect(TokenType.SELECT)
        distinct = False
        if self._current.type == TokenType.DISTINCT:
            distinct = True
            self._advance()
        select_list = [self._parse_select_item()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            select_list.append(self._parse_select_item())
        from_clause = (self._parse_from()
                       if self._current.type == TokenType.FROM
                       else None)
        where = None
        if self._current.type == TokenType.WHERE:
            self._advance()
            where = self._expr.parse_expression()
        group_by = (self._parse_group_by()
                    if self._current.type == TokenType.GROUP
                    else None)
        having = None
        if self._current.type == TokenType.HAVING:
            self._advance()
            having = self._expr.parse_expression()
        order_by = (self._parse_order_by()
                    if self._current.type == TokenType.ORDER
                    else [])
        limit = None
        if self._current.type == TokenType.LIMIT:
            self._advance()
            limit = self._expr.parse_expression()
        offset = None
        if self._current.type == TokenType.OFFSET:
            self._advance()
            offset = self._expr.parse_expression()
        return SelectStmt(
            distinct=distinct, select_list=select_list,
            from_clause=from_clause, where=where,
            group_by=group_by, having=having,
            order_by=order_by, limit=limit, offset=offset)

    def _parse_select_item(self):
        e = self._expr.parse_expression()
        if self._current.type == TokenType.AS:
            self._advance()
            return AliasExpr(
                expr=e, alias=self._accept_identifier())
        if self._current.type == TokenType.IDENTIFIER:
            a = self._current.value
            self._advance()
            return AliasExpr(expr=e, alias=a)
        return e

    # ═══ FROM / JOIN ═══

    def _parse_from(self):
        self._expect(TokenType.FROM)
        tref = self._parse_table_ref()
        joins = []
        while self._current.type in (
                TokenType.JOIN, TokenType.INNER,
                TokenType.LEFT, TokenType.RIGHT,
                TokenType.CROSS, TokenType.FULL,
                TokenType.COMMA, TokenType.NATURAL):
            if self._current.type == TokenType.COMMA:
                self._advance()
                joins.append(JoinClause(
                    join_type='CROSS',
                    table=self._parse_table_ref()))
                continue
            joins.append(self._parse_join())
        return FromClause(table=tref, joins=joins)

    def _parse_table_ref(self):
        if self._current.type == TokenType.LPAREN:
            self._advance()
            if self._current.type == TokenType.SELECT:
                sq = self._parse_select()
                self._expect(TokenType.RPAREN)
                a = None
                if self._current.type == TokenType.AS:
                    self._advance()
                    a = self._accept_identifier()
                elif self._current.type == TokenType.IDENTIFIER:
                    a = self._current.value
                    self._advance()
                return TableRef(
                    name=a or '__subquery', alias=a,
                    subquery=sq)
        name = self._accept_identifier()
        # 表函数（如 generate_series(1, 10)）
        if self._current.type == TokenType.LPAREN:
            func_args = self._parse_func_args()
            a = None
            if self._current.type == TokenType.AS:
                self._advance()
                a = self._accept_identifier()
            elif self._current.type == TokenType.IDENTIFIER:
                a = self._current.value
                self._advance()
            return TableRef(name=name, alias=a,
                            func_args=func_args)
        a = None
        if self._current.type == TokenType.AS:
            self._advance()
            a = self._accept_identifier()
        elif self._current.type == TokenType.IDENTIFIER:
            a = self._current.value
            self._advance()
        return TableRef(name=name, alias=a)

    def _parse_func_args(self):
        self._expect(TokenType.LPAREN)
        args = []
        if self._current.type != TokenType.RPAREN:
            args.append(self._expr.parse_expression())
            while self._current.type == TokenType.COMMA:
                self._advance()
                args.append(self._expr.parse_expression())
        self._expect(TokenType.RPAREN)
        return args

    def _parse_join(self):
        natural = False
        if self._current.type == TokenType.NATURAL:
            natural = True
            self._advance()
        jt = 'INNER'
        if self._current.type == TokenType.INNER:
            self._advance()
        elif self._current.type == TokenType.LEFT:
            self._advance(); jt = 'LEFT'
        elif self._current.type == TokenType.RIGHT:
            self._advance(); jt = 'RIGHT'
        elif self._current.type == TokenType.CROSS:
            self._advance(); jt = 'CROSS'
        elif self._current.type == TokenType.FULL:
            self._advance(); jt = 'FULL'
        if self._current.type == TokenType.OUTER:
            self._advance()
        self._expect(TokenType.JOIN)
        t = self._parse_table_ref()
        on = None
        using_cols = None
        if natural:
            pass  # NATURAL JOIN 不需要 ON/USING
        elif self._current.type == TokenType.ON:
            self._advance()
            on = self._expr.parse_expression()
        elif self._current.type == TokenType.USING:
            self._advance()
            self._expect(TokenType.LPAREN)
            using_cols = [self._accept_identifier()]
            while self._current.type == TokenType.COMMA:
                self._advance()
                using_cols.append(self._accept_identifier())
            self._expect(TokenType.RPAREN)
        return JoinClause(
            join_type=jt, table=t, on=on,
            using=using_cols, natural=natural)

    # ═══ GROUP BY / ORDER BY ═══

    def _parse_group_by(self):
        self._expect(TokenType.GROUP)
        self._expect(TokenType.BY)
        keys = [self._expr.parse_expression()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            keys.append(self._expr.parse_expression())
        return GroupByClause(keys=keys)

    def _parse_order_by(self):
        self._expect(TokenType.ORDER)
        self._expect(TokenType.BY)
        keys = [self._expr.parse_sort_key()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            keys.append(self._expr.parse_sort_key())
        return keys

    # ═══ INSERT / UPDATE / DELETE ═══

    def _parse_insert(self):
        self._expect(TokenType.INSERT)
        self._expect(TokenType.INTO)
        t = self._accept_identifier()
        cols = None
        if self._current.type == TokenType.LPAREN:
            # [B07] 安全区分列名列表 vs 子查询
            saved = self._pos
            saved_c = self._current
            self._advance()
            if self._current.type == TokenType.SELECT:
                # 回退：这是 INSERT INTO t (SELECT ...)
                self._pos = saved
                self._current = saved_c
            else:
                cols = [self._accept_identifier()]
                while self._current.type == TokenType.COMMA:
                    self._advance()
                    cols.append(self._accept_identifier())
                self._expect(TokenType.RPAREN)
        if self._current.type == TokenType.SELECT:
            return InsertStmt(
                table=t, columns=cols, values=[],
                query=self._parse_select())
        self._expect(TokenType.VALUES)
        rows = [self._parse_value_row()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            rows.append(self._parse_value_row())
        return InsertStmt(table=t, columns=cols, values=rows)

    def _parse_value_row(self):
        self._expect(TokenType.LPAREN)
        e = [self._expr.parse_expression()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            e.append(self._expr.parse_expression())
        self._expect(TokenType.RPAREN)
        return e

    def _parse_update(self):
        self._expect(TokenType.UPDATE)
        t = self._accept_identifier()
        self._expect(TokenType.SET)
        a = [self._parse_assignment()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            a.append(self._parse_assignment())
        w = None
        if self._current.type == TokenType.WHERE:
            self._advance()
            w = self._expr.parse_expression()
        return UpdateStmt(table=t, assignments=a, where=w)

    def _parse_assignment(self):
        c = self._accept_identifier()
        self._expect(TokenType.EQUAL)
        return Assignment(
            column=c,
            value=self._expr.parse_expression())

    def _parse_delete(self):
        self._expect(TokenType.DELETE)
        self._expect(TokenType.FROM)
        t = self._accept_identifier()
        w = None
        if self._current.type == TokenType.WHERE:
            self._advance()
            w = self._expr.parse_expression()
        return DeleteStmt(table=t, where=w)

    # ═══ CREATE / DROP / ALTER ═══

    def _parse_create_table(self):
        ine = False
        if self._current.type == TokenType.IF:
            self._advance()
            self._expect(TokenType.NOT)
            self._expect(TokenType.EXISTS)
            ine = True
        t = self._accept_identifier()
        self._expect(TokenType.LPAREN)
        cols = [self._parse_column_def()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            cols.append(self._parse_column_def())
        self._expect(TokenType.RPAREN)
        return CreateTableStmt(
            table=t, columns=cols, if_not_exists=ine)

    def _parse_column_def(self):
        n = self._accept_identifier()
        tn = self._parse_type_name()
        nl = True
        pk = False
        while True:
            if self._current.type == TokenType.NOT:
                self._advance()
                self._expect(TokenType.NULL)
                nl = False
            elif self._current.type == TokenType.NULL:
                self._advance()
                nl = True
            elif self._current.type == TokenType.PRIMARY:
                self._advance()
                self._expect(TokenType.KEY)
                pk = True
            else:
                break
        return ColumnDef(
            name=n, type_name=tn,
            nullable=nl, primary_key=pk)

    def _parse_drop_table(self):
        ie = False
        if self._current.type == TokenType.IF:
            self._advance()
            self._expect(TokenType.EXISTS)
            ie = True
        return DropTableStmt(
            table=self._accept_identifier(), if_exists=ie)

    def _parse_create_index(self, unique):
        ine = False
        if self._current.type == TokenType.IF:
            self._advance()
            self._expect(TokenType.NOT)
            self._expect(TokenType.EXISTS)
            ine = True
        idx = self._accept_identifier()
        self._expect(TokenType.ON)
        tbl = self._accept_identifier()
        self._expect(TokenType.LPAREN)
        cols = [self._accept_identifier()]
        while self._current.type == TokenType.COMMA:
            self._advance()
            cols.append(self._accept_identifier())
        self._expect(TokenType.RPAREN)
        return CreateIndexStmt(
            index_name=idx, table=tbl, columns=cols,
            unique=unique, if_not_exists=ine)

    def _parse_drop_index(self):
        ie = False
        if self._current.type == TokenType.IF:
            self._advance()
            self._expect(TokenType.EXISTS)
            ie = True
        return DropIndexStmt(
            index_name=self._accept_identifier(),
            if_exists=ie)

    def _parse_explain(self):
        self._expect(TokenType.EXPLAIN)
        inner = self._parse_statement()
        while self._current.type in (
                TokenType.UNION, TokenType.INTERSECT,
                TokenType.EXCEPT):
            inner = self._parse_set_operation(inner)
        return ExplainStmt(statement=inner)

    def _parse_alter(self):
        self._expect(TokenType.ALTER)
        self._expect(TokenType.TABLE)
        t = self._accept_identifier()
        if self._current.type == TokenType.ADD:
            self._advance()
            if self._current.type == TokenType.COLUMN:
                self._advance()
            return AlterTableStmt(
                table=t, action='ADD_COLUMN',
                column_def=self._parse_column_def())
        if self._current.type == TokenType.DROP:
            self._advance()
            if self._current.type == TokenType.COLUMN:
                self._advance()
            return AlterTableStmt(
                table=t, action='DROP_COLUMN',
                column_name=self._accept_identifier())
        if self._current.type == TokenType.RENAME:
            self._advance()
            if self._current.type == TokenType.COLUMN:
                self._advance()
            old = self._accept_identifier()
            self._expect(TokenType.TO)
            new = self._accept_identifier()
            return AlterTableStmt(
                table=t, action='RENAME_COLUMN',
                column_name=old, new_name=new)
        raise ParseError(
            f"期望 ADD/DROP/RENAME",
            self._current.line, self._current.col)

    # ═══ COPY ═══

    def _parse_copy(self):
        self._expect(TokenType.COPY)
        table = self._accept_identifier()
        direction = 'FROM'
        if self._current.type == TokenType.FROM:
            self._advance()
        elif self._current.type == TokenType.TO:
            direction = 'TO'
            self._advance()
        else:
            raise ParseError(
                "期望 FROM 或 TO",
                self._current.line, self._current.col)
        file_path = self._expect(TokenType.STRING).value
        has_header = True
        delimiter = ','
        if self._current.type == TokenType.WITH:
            self._advance()
            self._expect(TokenType.LPAREN)
            while self._current.type != TokenType.RPAREN:
                opt = self._current.value.upper()
                self._advance()
                if opt == 'HEADER':
                    val = self._parse_option_value()
                    has_header = val in (
                        'TRUE', '1', 'YES', 'ON')
                elif opt == 'DELIMITER':
                    delimiter = self._expect(
                        TokenType.STRING).value
                if self._current.type == TokenType.COMMA:
                    self._advance()
            self._expect(TokenType.RPAREN)
        return CopyStmt(
            table=table, file_path=file_path,
            direction=direction, has_header=has_header,
            delimiter=delimiter)

    def _parse_option_value(self):
        if self._current.type == TokenType.TRUE:
            self._advance(); return 'TRUE'
        if self._current.type == TokenType.FALSE:
            self._advance(); return 'FALSE'
        if self._current.type == TokenType.IDENTIFIER:
            v = self._current.value.upper()
            self._advance(); return v
        if self._current.type == TokenType.STRING:
            v = self._expect(TokenType.STRING).value
            return v.upper()
        if self._current.type == TokenType.INTEGER_LIT:
            v = self._current.value
            self._advance(); return v
        if self._current.type in _UNRESERVED:
            v = self._current.value.upper()
            self._advance(); return v
        raise ParseError(
            f"期望选项值",
            self._current.line, self._current.col)

    # ═══ VACUUM [M05] ═══

    def _parse_vacuum(self):
        """VACUUM [table_name]"""
        self._expect(TokenType.VACUUM)
        table = None
        if (self._current.type == TokenType.IDENTIFIER
                or self._current.type in _UNRESERVED):
            table = self._accept_identifier()
        return VacuumStmt(table=table)

    # ═══ CTE [B07] ═══

    def _parse_with_cte(self):
        self._advance()
        is_recursive = False
        if (self._current.type == TokenType.IDENTIFIER
                and self._current.value == 'recursive'):
            is_recursive = True
            self._advance()
        ctes = []
        while True:
            cte_name = self._accept_identifier()
            cte_columns = None
            if self._current.type == TokenType.LPAREN:
                # [B07] 安全区分列名列表 vs 子查询：
                # 保存位置尝试解析列名，失败则回退解析子查询
                saved_pos = self._pos
                saved_current = self._current
                self._advance()
                if self._current.type == TokenType.SELECT:
                    # (SELECT ...) — 子查询
                    cte_query = self._parse_select()
                    while self._current.type in (
                            TokenType.UNION, TokenType.INTERSECT,
                            TokenType.EXCEPT):
                        cte_query = self._parse_set_operation(
                            cte_query)
                    self._expect(TokenType.RPAREN)
                    ctes.append((cte_name, cte_query,
                                 is_recursive, cte_columns))
                    if self._current.type == TokenType.COMMA:
                        self._advance()
                        continue
                    else:
                        break
                else:
                    # 尝试解析列名列表
                    try:
                        cte_columns = [self._accept_identifier()]
                        while self._current.type == TokenType.COMMA:
                            self._advance()
                            cte_columns.append(
                                self._accept_identifier())
                        self._expect(TokenType.RPAREN)
                    except ParseError:
                        # 列名解析失败 → 回退
                        self._pos = saved_pos
                        self._current = saved_current
                        cte_columns = None

            if cte_columns is not None or (
                    self._current.type == TokenType.AS):
                self._expect(TokenType.AS)
                self._expect(TokenType.LPAREN)
                cte_query = self._parse_select()
                while self._current.type in (
                        TokenType.UNION, TokenType.INTERSECT,
                        TokenType.EXCEPT):
                    cte_query = self._parse_set_operation(
                        cte_query)
                self._expect(TokenType.RPAREN)
                ctes.append((cte_name, cte_query,
                             is_recursive, cte_columns))
            if self._current.type == TokenType.COMMA:
                self._advance()
            else:
                break

        import dataclasses
        return dataclasses.replace(
            self._parse_select(), ctes=ctes)
