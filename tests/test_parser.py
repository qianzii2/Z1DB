from __future__ import annotations
from tests.conftest import *
from parser.lexer import Lexer
from parser.parser import Parser
from parser.ast import *

def _parse(sql):
    return Parser(Lexer(sql).tokenize()).parse()

def test_simple_select():
    ast = _parse("SELECT a, b FROM t;")
    assert isinstance(ast, SelectStmt)
    assert len(ast.select_list) == 2

def test_select_where():
    ast = _parse("SELECT * FROM t WHERE x > 5;")
    assert ast.where is not None
    assert isinstance(ast.where, BinaryExpr)

def test_select_order_limit():
    ast = _parse("SELECT * FROM t ORDER BY a DESC LIMIT 10 OFFSET 5;")
    assert len(ast.order_by) == 1
    assert ast.order_by[0].direction == 'DESC'
    assert ast.limit is not None
    assert ast.offset is not None

def test_insert():
    ast = _parse("INSERT INTO t VALUES (1, 'hello');")
    assert isinstance(ast, InsertStmt)
    assert ast.table == 't'
    assert len(ast.values) == 1

def test_insert_columns():
    ast = _parse("INSERT INTO t (a, b) VALUES (1, 2);")
    assert ast.columns == ['a', 'b']

def test_update():
    ast = _parse("UPDATE t SET a = 1, b = 2 WHERE id = 3;")
    assert isinstance(ast, UpdateStmt)
    assert len(ast.assignments) == 2

def test_delete():
    ast = _parse("DELETE FROM t WHERE id = 1;")
    assert isinstance(ast, DeleteStmt)

def test_create_table():
    ast = _parse("CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR(100) NOT NULL);")
    assert isinstance(ast, CreateTableStmt)
    assert len(ast.columns) == 2
    assert ast.columns[0].primary_key == True

def test_drop_table():
    ast = _parse("DROP TABLE IF EXISTS t;")
    assert isinstance(ast, DropTableStmt)
    assert ast.if_exists == True

def test_join():
    ast = _parse("SELECT * FROM a JOIN b ON a.id = b.aid;")
    assert len(ast.from_clause.joins) == 1
    assert ast.from_clause.joins[0].join_type == 'INNER'

def test_left_join():
    ast = _parse("SELECT * FROM a LEFT JOIN b ON a.id = b.id;")
    assert ast.from_clause.joins[0].join_type == 'LEFT'

def test_group_by_having():
    ast = _parse("SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1;")
    assert ast.group_by is not None
    assert ast.having is not None

def test_case_expr():
    ast = _parse("SELECT CASE WHEN a > 1 THEN 'y' ELSE 'n' END FROM t;")
    expr = ast.select_list[0]
    assert isinstance(expr, CaseExpr)

def test_subquery():
    ast = _parse("SELECT * FROM t WHERE a IN (SELECT b FROM t2);")
    assert isinstance(ast.where, InExpr)

def test_cte():
    ast = _parse("WITH c AS (SELECT 1 AS x) SELECT * FROM c;")
    assert len(ast.ctes) == 1

def test_union():
    ast = _parse("SELECT a FROM t1 UNION SELECT b FROM t2;")
    assert isinstance(ast, SetOperationStmt)
    assert ast.op == 'UNION'

def test_window():
    ast = _parse("SELECT ROW_NUMBER() OVER (ORDER BY a) FROM t;")
    expr = ast.select_list[0]
    assert isinstance(expr, WindowCall)

def test_create_index():
    ast = _parse("CREATE UNIQUE INDEX idx ON t (a, b);")
    assert isinstance(ast, CreateIndexStmt)
    assert ast.unique == True

def test_copy():
    ast = _parse("COPY t FROM 'file.csv' WITH (HEADER TRUE, DELIMITER ',');")
    assert isinstance(ast, CopyStmt)
    assert ast.direction == 'FROM'

def test_alter_table():
    ast = _parse("ALTER TABLE t ADD COLUMN x INT;")
    assert isinstance(ast, AlterTableStmt)
    assert ast.action == 'ADD_COLUMN'

def test_explain():
    ast = _parse("EXPLAIN SELECT * FROM t;")
    assert isinstance(ast, ExplainStmt)
