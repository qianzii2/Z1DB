from __future__ import annotations
"""Recursive-descent + Pratt parser — Phase 3: windows, set ops, subqueries."""
from typing import List, Optional
from parser.ast import *
from parser.precedence import Precedence
from parser.token import Token, TokenType
from storage.types import DataType
from utils.errors import ParseError

_UNRESERVED = frozenset({
    TokenType.INT,TokenType.INTEGER,TokenType.BIGINT,TokenType.FLOAT_KW,
    TokenType.DOUBLE,TokenType.REAL,TokenType.BOOLEAN,TokenType.BOOL,
    TokenType.VARCHAR,TokenType.TEXT_KW,TokenType.DATE_KW,TokenType.TIMESTAMP,
    TokenType.FIRST,TokenType.LAST,TokenType.KEY,TokenType.PARTITION,
    TokenType.ROWS,TokenType.RANGE,TokenType.UNBOUNDED,TokenType.PRECEDING,
    TokenType.FOLLOWING,TokenType.CURRENT,TokenType.ROW,TokenType.OVER,
})
_INFIX = {
    TokenType.OR:Precedence.OR,TokenType.AND:Precedence.AND,
    TokenType.EQUAL:Precedence.COMPARISON,TokenType.NOT_EQUAL:Precedence.COMPARISON,
    TokenType.LESS:Precedence.COMPARISON,TokenType.GREATER:Precedence.COMPARISON,
    TokenType.LESS_EQUAL:Precedence.COMPARISON,TokenType.GREATER_EQUAL:Precedence.COMPARISON,
    TokenType.IS:Precedence.IS,TokenType.IN:Precedence.COMPARISON,
    TokenType.BETWEEN:Precedence.COMPARISON,TokenType.LIKE:Precedence.COMPARISON,
    TokenType.PIPE_PIPE:Precedence.CONCAT,
    TokenType.PLUS:Precedence.ADDITION,TokenType.MINUS:Precedence.ADDITION,
    TokenType.STAR:Precedence.MULTIPLY,TokenType.SLASH:Precedence.MULTIPLY,
    TokenType.PERCENT:Precedence.MULTIPLY,
}
_AGGS = frozenset({'COUNT','SUM','AVG','MIN','MAX'})
_WINFUNCS = frozenset({'ROW_NUMBER','RANK','DENSE_RANK','NTILE','PERCENT_RANK',
                        'CUME_DIST','LAG','LEAD','FIRST_VALUE','LAST_VALUE','NTH_VALUE'})

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._t = tokens; self._p = 0; self._c = tokens[0]
    def _adv(self) -> None:
        self._p += 1
        if self._p < len(self._t): self._c = self._t[self._p]
    def _pk(self) -> Token:
        n = self._p+1; return self._t[n] if n < len(self._t) else self._t[-1]
    def _ex(self, tt: TokenType) -> Token:
        if self._c.type != tt:
            raise ParseError(f"expected {tt.value}, got {self._c.value!r}",self._c.line,self._c.col)
        t = self._c; self._adv(); return t
    def _exi(self) -> int: return int(self._ex(TokenType.INTEGER_LIT).value)
    def _aid(self) -> str:
        if self._c.type == TokenType.IDENTIFIER: v = self._c.value; self._adv(); return v
        if self._c.type in _UNRESERVED: v = self._c.value.lower(); self._adv(); return v
        raise ParseError(f"expected identifier, got {self._c.value!r}",self._c.line,self._c.col)

    def parse(self) -> object:
        stmt = self._stmt()
        # Check for set operations
        while self._c.type in (TokenType.UNION, TokenType.INTERSECT, TokenType.EXCEPT):
            stmt = self._parse_set_op(stmt)
        if self._c.type == TokenType.SEMICOLON: self._adv()
        if self._c.type != TokenType.EOF:
            raise ParseError(f"extra tokens: {self._c.value!r}",self._c.line,self._c.col)
        return stmt

    def _parse_set_op(self, left: object) -> SetOperationStmt:
        op = self._c.value; self._adv()
        all_ = False
        if self._c.type == TokenType.ALL: all_ = True; self._adv()
        right = self._parse_select_core()
        # Chain further set ops
        while self._c.type in (TokenType.UNION, TokenType.INTERSECT, TokenType.EXCEPT):
            right = self._parse_set_op(right)
        return SetOperationStmt(op=op, all=all_, left=left, right=right)

    def _stmt(self) -> object:
        tt = self._c.type
        if tt == TokenType.SELECT: return self._parse_select_core()
        if tt == TokenType.INSERT: return self._ins()
        if tt == TokenType.UPDATE: return self._upd()
        if tt == TokenType.DELETE: return self._dlt()
        if tt == TokenType.CREATE: self._adv(); self._ex(TokenType.TABLE); return self._crt()
        if tt == TokenType.DROP: self._adv(); self._ex(TokenType.TABLE); return self._drp()
        raise ParseError(f"unexpected: {self._c.value!r}",self._c.line,self._c.col)

    def _parse_select_core(self) -> SelectStmt:
        self._ex(TokenType.SELECT)
        dist = False
        if self._c.type == TokenType.DISTINCT: dist = True; self._adv()
        sl = [self._si()]
        while self._c.type == TokenType.COMMA: self._adv(); sl.append(self._si())
        fc = self._fr() if self._c.type == TokenType.FROM else None
        w = None
        if self._c.type == TokenType.WHERE: self._adv(); w = self._expr()
        gb = self._gb() if self._c.type == TokenType.GROUP else None
        hv = None
        if self._c.type == TokenType.HAVING: self._adv(); hv = self._expr()
        ob: list = []
        if self._c.type == TokenType.ORDER: ob = self._ob()
        lm = None
        if self._c.type == TokenType.LIMIT: self._adv(); lm = self._expr()
        of = None
        if self._c.type == TokenType.OFFSET: self._adv(); of = self._expr()
        return SelectStmt(distinct=dist,select_list=sl,from_clause=fc,where=w,
                          group_by=gb,having=hv,order_by=ob,limit=lm,offset=of)

    def _si(self) -> object:
        e = self._expr()
        if self._c.type == TokenType.AS: self._adv(); return AliasExpr(expr=e,alias=self._aid())
        if self._c.type == TokenType.IDENTIFIER:
            a = self._c.value; self._adv(); return AliasExpr(expr=e,alias=a)
        return e

    def _fr(self) -> FromClause:
        self._ex(TokenType.FROM)
        tref = self._tref()
        joins: list = []
        while self._c.type in (TokenType.JOIN,TokenType.INNER,TokenType.LEFT,
                                TokenType.RIGHT,TokenType.CROSS,TokenType.FULL,TokenType.COMMA):
            if self._c.type == TokenType.COMMA:
                self._adv(); joins.append(JoinClause(join_type='CROSS',table=self._tref())); continue
            joins.append(self._jn())
        return FromClause(table=tref,joins=joins)

    def _tref(self) -> TableRef:
        if self._c.type == TokenType.LPAREN:
            self._adv()
            if self._c.type == TokenType.SELECT:
                sq = self._parse_select_core()
                self._ex(TokenType.RPAREN)
                alias = None
                if self._c.type == TokenType.AS: self._adv(); alias = self._aid()
                elif self._c.type == TokenType.IDENTIFIER: alias = self._c.value; self._adv()
                return TableRef(name=alias or '__subquery', alias=alias, subquery=sq)
            # Parenthesized table ref — just parse name
        name = self._aid()
        alias: Optional[str] = None
        if self._c.type == TokenType.AS: self._adv(); alias = self._aid()
        elif self._c.type == TokenType.IDENTIFIER: alias = self._c.value; self._adv()
        return TableRef(name=name, alias=alias)

    def _jn(self) -> JoinClause:
        jt = 'INNER'
        if self._c.type == TokenType.INNER: self._adv()
        elif self._c.type == TokenType.LEFT: self._adv(); jt = 'LEFT'
        elif self._c.type == TokenType.RIGHT: self._adv(); jt = 'RIGHT'
        elif self._c.type == TokenType.CROSS: self._adv(); jt = 'CROSS'
        elif self._c.type == TokenType.FULL: self._adv(); jt = 'FULL'
        if self._c.type == TokenType.OUTER: self._adv()
        self._ex(TokenType.JOIN)
        t = self._tref(); on = None
        if self._c.type == TokenType.ON: self._adv(); on = self._expr()
        return JoinClause(join_type=jt,table=t,on=on)

    def _gb(self) -> GroupByClause:
        self._ex(TokenType.GROUP);self._ex(TokenType.BY)
        k = [self._expr()]
        while self._c.type == TokenType.COMMA: self._adv(); k.append(self._expr())
        return GroupByClause(keys=k)

    def _ob(self) -> list:
        self._ex(TokenType.ORDER);self._ex(TokenType.BY)
        k = [self._sk()]
        while self._c.type == TokenType.COMMA: self._adv(); k.append(self._sk())
        return k

    def _sk(self) -> SortKey:
        e = self._expr(); d = 'ASC'
        if self._c.type == TokenType.ASC: self._adv()
        elif self._c.type == TokenType.DESC: d = 'DESC'; self._adv()
        ns: Optional[str] = None
        if self._c.type == TokenType.NULLS:
            self._adv()
            if self._c.type == TokenType.FIRST: ns = 'NULLS_FIRST'; self._adv()
            elif self._c.type == TokenType.LAST: ns = 'NULLS_LAST'; self._adv()
        return SortKey(expr=e,direction=d,nulls=ns)

    def _ins(self) -> InsertStmt:
        self._ex(TokenType.INSERT);self._ex(TokenType.INTO); t = self._aid()
        cols: Optional[List[str]] = None
        if self._c.type == TokenType.LPAREN:
            self._adv(); cols = [self._aid()]
            while self._c.type == TokenType.COMMA: self._adv(); cols.append(self._aid())
            self._ex(TokenType.RPAREN)
        self._ex(TokenType.VALUES)
        rows = [self._vr()]
        while self._c.type == TokenType.COMMA: self._adv(); rows.append(self._vr())
        return InsertStmt(table=t,columns=cols,values=rows)

    def _vr(self) -> list:
        self._ex(TokenType.LPAREN); e = [self._expr()]
        while self._c.type == TokenType.COMMA: self._adv(); e.append(self._expr())
        self._ex(TokenType.RPAREN); return e

    def _upd(self) -> UpdateStmt:
        self._ex(TokenType.UPDATE); t = self._aid(); self._ex(TokenType.SET)
        a = [self._asgn()]
        while self._c.type == TokenType.COMMA: self._adv(); a.append(self._asgn())
        w = None
        if self._c.type == TokenType.WHERE: self._adv(); w = self._expr()
        return UpdateStmt(table=t,assignments=a,where=w)

    def _asgn(self) -> Assignment:
        c = self._aid(); self._ex(TokenType.EQUAL); return Assignment(column=c,value=self._expr())

    def _dlt(self) -> DeleteStmt:
        self._ex(TokenType.DELETE);self._ex(TokenType.FROM); t = self._aid()
        w = None
        if self._c.type == TokenType.WHERE: self._adv(); w = self._expr()
        return DeleteStmt(table=t,where=w)

    def _crt(self) -> CreateTableStmt:
        ine = False
        if self._c.type == TokenType.IF:
            self._adv();self._ex(TokenType.NOT);self._ex(TokenType.EXISTS);ine = True
        t = self._aid(); self._ex(TokenType.LPAREN)
        cols = [self._cd()]
        while self._c.type == TokenType.COMMA: self._adv(); cols.append(self._cd())
        self._ex(TokenType.RPAREN); return CreateTableStmt(table=t,columns=cols,if_not_exists=ine)

    def _cd(self) -> ColumnDef:
        n = self._aid(); tn = self._tn(); nl = True; pk = False
        while True:
            if self._c.type == TokenType.NOT: self._adv();self._ex(TokenType.NULL); nl = False
            elif self._c.type == TokenType.NULL: self._adv(); nl = True
            elif self._c.type == TokenType.PRIMARY: self._adv();self._ex(TokenType.KEY); pk = True
            else: break
        return ColumnDef(name=n,type_name=tn,nullable=nl,primary_key=pk)

    def _tn(self) -> TypeName:
        if self._c.type in _UNRESERVED or self._c.type == TokenType.IDENTIFIER:
            n = self._c.value.upper(); self._adv()
        else: raise ParseError(f"expected type",self._c.line,self._c.col)
        p: list[int] = []
        if self._c.type == TokenType.LPAREN: self._adv(); p.append(self._exi()); self._ex(TokenType.RPAREN)
        return TypeName(name=n,params=p)

    def _drp(self) -> DropTableStmt:
        ie = False
        if self._c.type == TokenType.IF: self._adv();self._ex(TokenType.EXISTS);ie = True
        return DropTableStmt(table=self._aid(),if_exists=ie)

    # ═══ Pratt ═══
    def _expr(self, mp: int = Precedence.LOWEST) -> object:
        l = self._prefix()
        while self._iprec() > mp: l = self._infix(l)
        return l

    def _prefix(self) -> object:
        tt = self._c.type
        if tt == TokenType.NOT: self._adv(); return UnaryExpr(op='NOT',operand=self._expr(Precedence.NOT_PREFIX))
        if tt == TokenType.MINUS: self._adv(); return UnaryExpr(op='-',operand=self._expr(Precedence.UNARY))
        if tt == TokenType.PLUS: self._adv(); return UnaryExpr(op='+',operand=self._expr(Precedence.UNARY))
        if tt in (TokenType.INTEGER_LIT,TokenType.FLOAT_LIT,TokenType.STRING,
                  TokenType.TRUE,TokenType.FALSE,TokenType.NULL):
            return self._lit()
        if tt == TokenType.IDENTIFIER or tt in _UNRESERVED: return self._idexpr()
        if tt == TokenType.CASE: return self._case()
        if tt == TokenType.CAST: return self._cast()
        if tt == TokenType.EXISTS:
            self._adv(); self._ex(TokenType.LPAREN)
            sq = self._parse_select_core(); self._ex(TokenType.RPAREN)
            return ExistsExpr(query=sq)
        if tt == TokenType.NOT:
            self._adv()
            if self._c.type == TokenType.EXISTS:
                self._adv(); self._ex(TokenType.LPAREN)
                sq = self._parse_select_core(); self._ex(TokenType.RPAREN)
                return ExistsExpr(query=sq, negated=True)
        if tt == TokenType.LPAREN:
            self._adv()
            if self._c.type == TokenType.SELECT:
                sq = self._parse_select_core(); self._ex(TokenType.RPAREN)
                return SubqueryExpr(query=sq)
            e = self._expr(); self._ex(TokenType.RPAREN); return e
        if tt == TokenType.STAR: self._adv(); return StarExpr()
        raise ParseError(f"unexpected: {self._c.value!r}",self._c.line,self._c.col)

    def _lit(self) -> Literal:
        t = self._c; self._adv()
        if t.type == TokenType.INTEGER_LIT:
            v = int(t.value)
            return Literal(value=v,inferred_type=DataType.INT if v <= 2_147_483_647 else DataType.BIGINT)
        if t.type == TokenType.FLOAT_LIT: return Literal(value=float(t.value),inferred_type=DataType.DOUBLE)
        if t.type == TokenType.STRING: return Literal(value=t.value,inferred_type=DataType.VARCHAR)
        if t.type == TokenType.TRUE: return Literal(value=True,inferred_type=DataType.BOOLEAN)
        if t.type == TokenType.FALSE: return Literal(value=False,inferred_type=DataType.BOOLEAN)
        return Literal(value=None,inferred_type=DataType.UNKNOWN)

    def _idexpr(self) -> object:
        name = self._c.value
        if self._c.type in _UNRESERVED: name = name.lower()
        self._adv()
        if self._c.type == TokenType.LPAREN: return self._fncall(name)
        if self._c.type == TokenType.DOT:
            self._adv(); return ColumnRef(table=name,column=self._aid())
        return ColumnRef(table=None,column=name)

    def _fncall(self, name: str) -> object:
        self._ex(TokenType.LPAREN); upper = name.upper()
        if upper in _AGGS:
            if self._c.type == TokenType.STAR:
                self._adv(); self._ex(TokenType.RPAREN)
                func = AggregateCall(name=upper,args=[StarExpr()])
                if self._c.type == TokenType.OVER: return self._window(func)
                return func
            dist = False
            if self._c.type == TokenType.DISTINCT: dist = True; self._adv()
            args = [self._expr()]; self._ex(TokenType.RPAREN)
            func = AggregateCall(name=upper,args=args,distinct=dist)
            if self._c.type == TokenType.OVER: return self._window(func)
            return func
        # Window function names
        if upper in _WINFUNCS:
            args: list = []
            if self._c.type != TokenType.RPAREN:
                args.append(self._expr())
                while self._c.type == TokenType.COMMA: self._adv(); args.append(self._expr())
            self._ex(TokenType.RPAREN)
            func = FunctionCall(name=upper,args=args)
            if self._c.type == TokenType.OVER: return self._window(func)
            return func
        # Regular function
        args = []
        if self._c.type != TokenType.RPAREN:
            args.append(self._expr())
            while self._c.type == TokenType.COMMA: self._adv(); args.append(self._expr())
        self._ex(TokenType.RPAREN)
        func = FunctionCall(name=name,args=args)
        if self._c.type == TokenType.OVER: return self._window(func)
        return func

    def _window(self, func: object) -> WindowCall:
        self._ex(TokenType.OVER); self._ex(TokenType.LPAREN)
        pb: list = []
        if self._c.type == TokenType.PARTITION:
            self._adv(); self._ex(TokenType.BY)
            pb.append(self._expr())
            while self._c.type == TokenType.COMMA: self._adv(); pb.append(self._expr())
        oby: list = []
        if self._c.type == TokenType.ORDER:
            self._ex(TokenType.ORDER); self._ex(TokenType.BY)
            oby.append(self._sk())
            while self._c.type == TokenType.COMMA: self._adv(); oby.append(self._sk())
        frame = None
        if self._c.type in (TokenType.ROWS, TokenType.RANGE):
            frame = self._parse_frame()
        self._ex(TokenType.RPAREN)
        return WindowCall(func=func,partition_by=pb,order_by=oby,frame=frame)

    def _parse_frame(self) -> WindowFrame:
        mode = self._c.value; self._adv()  # ROWS or RANGE
        if self._c.type == TokenType.BETWEEN:
            self._adv()
            start = self._parse_bound()
            self._ex(TokenType.AND)
            end = self._parse_bound()
            return WindowFrame(mode=mode, start=start, end=end)
        # Single bound (start only, end = CURRENT ROW)
        start = self._parse_bound()
        return WindowFrame(mode=mode, start=start, end=FrameBound(type='CURRENT_ROW'))

    def _parse_bound(self) -> FrameBound:
        if self._c.type == TokenType.UNBOUNDED:
            self._adv()
            if self._c.type == TokenType.PRECEDING:
                self._adv(); return FrameBound(type='UNBOUNDED_PRECEDING')
            self._ex(TokenType.FOLLOWING); return FrameBound(type='UNBOUNDED_FOLLOWING')
        if self._c.type == TokenType.CURRENT:
            self._adv(); self._ex(TokenType.ROW); return FrameBound(type='CURRENT_ROW')
        if self._c.type == TokenType.INTEGER_LIT:
            n = int(self._c.value); self._adv()
            if self._c.type == TokenType.PRECEDING:
                self._adv(); return FrameBound(type='N_PRECEDING', offset=n)
            self._ex(TokenType.FOLLOWING); return FrameBound(type='N_FOLLOWING', offset=n)
        raise ParseError(f"expected frame bound",self._c.line,self._c.col)

    def _case(self) -> CaseExpr:
        self._ex(TokenType.CASE); op = None
        if self._c.type != TokenType.WHEN: op = self._expr()
        ws: list = []
        while self._c.type == TokenType.WHEN:
            self._adv(); c = self._expr(); self._ex(TokenType.THEN); r = self._expr()
            ws.append((c,r))
        el = None
        if self._c.type == TokenType.ELSE: self._adv(); el = self._expr()
        self._ex(TokenType.END); return CaseExpr(operand=op,when_clauses=ws,else_expr=el)

    def _cast(self) -> CastExpr:
        self._ex(TokenType.CAST);self._ex(TokenType.LPAREN)
        e = self._expr(); self._ex(TokenType.AS); t = self._tn()
        self._ex(TokenType.RPAREN); return CastExpr(expr=e,type_name=t)

    def _iprec(self) -> int:
        if self._c.type == TokenType.NOT:
            n = self._pk()
            if n.type in (TokenType.IN,TokenType.BETWEEN,TokenType.LIKE): return Precedence.COMPARISON
            return Precedence.LOWEST
        return _INFIX.get(self._c.type, Precedence.LOWEST)

    def _infix(self, left: object) -> object:
        tt = self._c.type
        if tt == TokenType.IS: return self._is(left)
        if tt == TokenType.IN: return self._in(left,False)
        if tt == TokenType.BETWEEN: return self._btw(left,False)
        if tt == TokenType.LIKE: return self._lk(left,False)
        if tt == TokenType.NOT:
            n = self._pk()
            if n.type == TokenType.IN: self._adv(); return self._in(left,True)
            if n.type == TokenType.BETWEEN: self._adv(); return self._btw(left,True)
            if n.type == TokenType.LIKE: self._adv(); return self._lk(left,True)
        prec = _INFIX[tt]; op = self._c.value; self._adv()
        m = {'AND':'AND','OR':'OR','=':'=','!=':'!=','<':'<','>':'>','<=':'<=','>=':'>=',
             '+':'+','-':'-','*':'*','/':'/','%':'%','||':'||'}
        return BinaryExpr(op=m.get(op,op),left=left,right=self._expr(prec))

    def _is(self, l: object) -> IsNullExpr:
        self._ex(TokenType.IS); neg = False
        if self._c.type == TokenType.NOT: neg = True; self._adv()
        self._ex(TokenType.NULL); return IsNullExpr(expr=l,negated=neg)

    def _in(self, l: object, neg: bool) -> InExpr:
        self._ex(TokenType.IN); self._ex(TokenType.LPAREN)
        if self._c.type == TokenType.SELECT:
            sq = self._parse_select_core(); self._ex(TokenType.RPAREN)
            return InExpr(expr=l,values=[SubqueryExpr(query=sq)],negated=neg)
        vs = [self._expr()]
        while self._c.type == TokenType.COMMA: self._adv(); vs.append(self._expr())
        self._ex(TokenType.RPAREN); return InExpr(expr=l,values=vs,negated=neg)

    def _btw(self, l: object, neg: bool) -> BetweenExpr:
        self._ex(TokenType.BETWEEN); lo = self._expr(Precedence.COMPARISON)
        self._ex(TokenType.AND); hi = self._expr(Precedence.COMPARISON)
        return BetweenExpr(expr=l,low=lo,high=hi,negated=neg)

    def _lk(self, l: object, neg: bool) -> LikeExpr:
        self._ex(TokenType.LIKE); return LikeExpr(expr=l,pattern=self._expr(Precedence.COMPARISON),negated=neg)
