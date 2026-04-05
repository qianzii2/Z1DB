
# Z1DB
**纯 Python 零依赖 OLAP 列式数据库引擎**
使用 Python 3.8+ 标准库。
## 快速开始
### 安装
```bash
git clone https://github.com/yourname/z1db.git
cd z1db
# 无需 pip install — 零依赖
python main.py
```
### 基本使用
```sql
z1db> CREATE TABLE users (id INT, name VARCHAR(50), age INT);
OK

z1db> INSERT INTO users VALUES (1,'Alice',30),(2,'Bob',25),(3,'Carol',35);
Inserted 3 rows

z1db> SELECT name, age FROM users WHERE age > 28 ORDER BY age DESC;
┌───────┬─────┐
│ name  │ age │
├───────┼─────┤
│ Carol │ 35  │
│ Alice │ 30  │
└───────┴─────┘
2 rows (0.001 sec)

z1db> SELECT COUNT(*), AVG(age), MAX(age) FROM users;
┌──────────┬──────────┬──────────┐
│ COUNT(*) │ AVG(age) │ MAX(age) │
├──────────┼──────────┼──────────┤
│ 3        │ 30.0     │ 35       │
└──────────┴──────────┴──────────┘

z1db> SELECT name, ROW_NUMBER() OVER (ORDER BY age DESC) AS rank FROM users;
┌───────┬──────┐
│ name  │ rank │
├───────┼──────┤
│ Carol │ 1    │
│ Alice │ 2    │
│ Bob   │ 3    │
└───────┴──────┘
```

### 高级功能

```sql
-- 递归 CTE: 生成序列
WITH RECURSIVE seq(x) AS (
    SELECT 1 UNION ALL SELECT x + 1 FROM seq WHERE x < 10
) SELECT x FROM seq;

-- JOIN
SELECT u.name, o.amount
FROM users u INNER JOIN orders o ON u.id = o.user_id;

-- 窗口函数
SELECT name, age,
  SUM(age) OVER (ORDER BY age ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running
FROM users;

-- 子查询
SELECT name FROM users WHERE id IN (SELECT user_id FROM orders WHERE amount > 100);

-- 近似聚合 (HyperLogLog)
SELECT APPROX_COUNT_DISTINCT(city) FROM big_table;

-- 分析
EXPLAIN SELECT * FROM users WHERE age > 25 ORDER BY name;
```

### 持久化

```bash
# 数据持久化到目录 (二进制列式格式)
python main.py /path/to/data

# 重启后数据自动恢复
python main.py /path/to/data
```

### TCP 服务器

```bash
# 启动服务器
python main.py --server 5433

# Python 客户端连接
python -c "
from server.protocol import Z1Client
with Z1Client('127.0.0.1', 5433) as c:
    r = c.execute('SELECT 1 + 1;')
    print(r)
"
```

### REPL 增强

```bash
z1db> .color on                              # ANSI 彩色输出
z1db> .profile on                            # 分阶段耗时 (解析/优化/执行)
z1db> .estimate on                           # 执行前行数预估
z1db> .benchmark 100 SELECT COUNT(*) FROM t; # 性能基准 (P50/P95/P99)
z1db> .memory                                # 内存使用统计
z1db> .history                               # 查询历史
z1db> .analyze users                         # 计算列统计信息
z1db> .stats users                           # 显示 NDV/min/max
z1db> .import data.csv mytable               # CSV 导入
z1db> .export out.csv mytable                # CSV 导出
```

## 支持的 SQL

| 语句   | 示例                                                         |
| ------ | ------------------------------------------------------------ |
| DDL    | `CREATE TABLE t (id INT, name VARCHAR(50) NOT NULL)`         |
| DML    | `INSERT` / `UPDATE ... SET ... WHERE` / `DELETE ... WHERE`   |
| SELECT | `WHERE` / `ORDER BY` / `LIMIT` / `OFFSET` / `DISTINCT`       |
| JOIN   | `INNER` / `LEFT` / `RIGHT` / `FULL OUTER` / `CROSS`          |
| 聚合   | `GROUP BY` / `HAVING` / `COUNT(DISTINCT)`                    |
| 窗口   | `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN ...)` |
| CTE    | `WITH t AS (...)` / `WITH RECURSIVE t(x) AS (... UNION ALL ...)` |
| 子查询 | `IN (SELECT ...)` / `EXISTS (SELECT ...)` / 标量子查询       |
| 集合   | `UNION` / `INTERSECT` / `EXCEPT` (ALL)                       |
| 表达式 | `CASE WHEN` / `CAST` / `BETWEEN` / `LIKE` / `IS NULL`        |
| 管理   | `EXPLAIN` / `ALTER TABLE ADD/DROP/RENAME COLUMN`             |

### 内置函数

```sql
标量-数学(19):    ABS CEIL FLOOR ROUND TRUNC POWER SQRT CBRT MOD SIGN
                  LN LOG2 LOG10 LOG EXP GREATEST LEAST RANDOM WIDTH_BUCKET

标量-字符串(28):  UPPER LOWER LENGTH SUBSTR CONCAT CONCAT_WS
                  TRIM LTRIM RTRIM LPAD RPAD LEFT RIGHT
                  REPLACE REVERSE REPEAT POSITION
                  STARTS_WITH ENDS_WITH CONTAINS
                  SPLIT SPLIT_PART INITCAP ASCII CHR
                  REGEXP_MATCH REGEXP_REPLACE REGEXP_EXTRACT

标量-日期(22):    NOW CURRENT_DATE CURRENT_TIMESTAMP
                  YEAR MONTH DAY HOUR MINUTE SECOND EXTRACT
                  DAY_OF_WEEK DAY_OF_YEAR WEEK_OF_YEAR QUARTER
                  DATE_TRUNC DATE_DIFF DATE_ADD DATE_SUB
                  DATE_FORMAT TO_DATE TO_TIMESTAMP EPOCH

标量-条件(12):    CAST TRY_CAST COALESCE NULLIF IF TYPEOF
                  HASH ENCODE DECODE BIT_COUNT MURMUR_HASH (CASE/WHEN)

聚合(22):         COUNT(*) COUNT(col) COUNT(DISTINCT)
                  SUM SUM(DISTINCT) AVG AVG(DISTINCT) MIN MAX
                  STDDEV STDDEV_POP VARIANCE VAR_POP
                  MEDIAN PERCENTILE_CONT PERCENTILE_DISC MODE
                  APPROX_COUNT_DISTINCT APPROX_PERCENTILE APPROX_TOP_K
                  ARRAY_AGG STRING_AGG GROUPING

窗口(16):         ROW_NUMBER RANK DENSE_RANK NTILE PERCENT_RANK CUME_DIST
                  LAG LEAD FIRST_VALUE LAST_VALUE NTH_VALUE
                  SUM/AVG/COUNT/MIN/MAX OVER

数组(17):         ARRAY ARRAY_LENGTH ARRAY_CONTAINS ARRAY_POSITION
                  ARRAY_SLICE ARRAY_APPEND ARRAY_PREPEND ARRAY_REMOVE
                  ARRAY_CONCAT ARRAY_SORT ARRAY_REVERSE ARRAY_DISTINCT
                  ARRAY_FLATTEN ARRAY_INTERSECT ARRAY_UNION ARRAY_EXCEPT ARRAY_JOIN

表函数(4):        EXPLODE POSEXPLODE UNNEST GENERATE_SERIES
相似度(2):        JACCARD_SIMILARITY COSINE_SIMILARITY
```

## 系统要求

- **Python 3.8+**
- Windows / Linux / macOS

## 许可证

MIT
