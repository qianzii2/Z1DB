from __future__ import annotations
from tests.conftest import *
from storage.types import DataType, promote, is_numeric, is_string

def test_promote_int_int():
    assert promote(DataType.INT, DataType.INT) == DataType.INT

def test_promote_int_bigint():
    assert promote(DataType.INT, DataType.BIGINT) == DataType.BIGINT

def test_promote_int_float():
    assert promote(DataType.INT, DataType.FLOAT) == DataType.FLOAT

def test_promote_int_double():
    assert promote(DataType.INT, DataType.DOUBLE) == DataType.DOUBLE

def test_promote_bigint_float():
    # BIGINT + FLOAT → DOUBLE (避免精度丢失)
    assert promote(DataType.BIGINT, DataType.FLOAT) == DataType.DOUBLE

def test_promote_float_double():
    assert promote(DataType.FLOAT, DataType.DOUBLE) == DataType.DOUBLE

def test_promote_varchar_text():
    assert promote(DataType.VARCHAR, DataType.TEXT) == DataType.TEXT

def test_promote_bool_int():
    assert promote(DataType.BOOLEAN, DataType.INT) == DataType.INT

def test_promote_unknown():
    assert promote(DataType.UNKNOWN, DataType.INT) == DataType.INT
    assert promote(DataType.INT, DataType.UNKNOWN) == DataType.INT

def test_promote_incompatible():
    assert_error(lambda: promote(DataType.INT, DataType.VARCHAR))

def test_is_numeric():
    assert is_numeric(DataType.INT)
    assert is_numeric(DataType.BIGINT)
    assert is_numeric(DataType.FLOAT)
    assert is_numeric(DataType.DOUBLE)
    assert is_numeric(DataType.BOOLEAN)
    assert not is_numeric(DataType.VARCHAR)

def test_is_string():
    assert is_string(DataType.VARCHAR)
    assert is_string(DataType.TEXT)
    assert not is_string(DataType.INT)
