from __future__ import annotations
from tests.conftest import *
from parser.lexer import Lexer
from parser.token import TokenType

def test_basic_select():
    tokens = Lexer("SELECT * FROM t;").tokenize()
    types = [t.type for t in tokens]
    assert TokenType.SELECT in types
    assert TokenType.STAR in types
    assert TokenType.FROM in types
    assert TokenType.EOF == types[-1]

def test_integer_literal():
    tokens = Lexer("42").tokenize()
    assert tokens[0].type == TokenType.INTEGER_LIT
    assert tokens[0].value == '42'

def test_float_literal():
    tokens = Lexer("3.14").tokenize()
    assert tokens[0].type == TokenType.FLOAT_LIT

def test_string_literal():
    tokens = Lexer("'hello world'").tokenize()
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == 'hello world'

def test_escaped_quote():
    tokens = Lexer("'it''s'").tokenize()
    assert tokens[0].value == "it's"

def test_operators():
    tokens = Lexer("a <= b != c").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.LESS_EQUAL in types
    assert TokenType.NOT_EQUAL in types

def test_pipe_pipe():
    tokens = Lexer("a || b").tokenize()
    assert any(t.type == TokenType.PIPE_PIPE for t in tokens)

def test_keywords():
    tokens = Lexer("SELECT INSERT CREATE DROP WHERE GROUP ORDER").tokenize()
    assert tokens[0].type == TokenType.SELECT
    assert tokens[1].type == TokenType.INSERT

def test_line_comment():
    tokens = Lexer("SELECT -- comment\n42").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.SELECT in types
    assert TokenType.INTEGER_LIT in types

def test_block_comment():
    tokens = Lexer("SELECT /* block */ 42").tokenize()
    assert len([t for t in tokens if t.type == TokenType.INTEGER_LIT]) == 1

def test_quoted_identifier():
    tokens = Lexer('"my column"').tokenize()
    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == 'my column'

def test_scientific_notation():
    tokens = Lexer("1.5e10").tokenize()
    assert tokens[0].type == TokenType.FLOAT_LIT
