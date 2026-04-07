from executor.core.operator import Operator
from executor.core.batch import VectorBatch
from catalog.catalog import Catalog
from parser.ast import TableRef


class SystemTableScanOperator(Operator):
    """系统表扫描算子，读取系统表全量数据并转换为 VectorBatch。"""

    def __init__(self, catalog: Catalog, table_name: str):
        super().__init__()
        self._catalog = catalog
        self._table_name = table_name
        self._batch = None
        self._emitted = False

    def output_schema(self):
        """返回输出列的 schema。"""
        schema = self._catalog.get_table(self._table_name)
        return [(col.name, col.dtype) for col in schema.columns]

    def open(self):
        """打开算子，加载系统表数据。"""
        store = self._catalog.get_store(self._table_name)
        schema = self._catalog.get_table(self._table_name)
        all_rows = store.read_all_rows()

        col_names = [c.name for c in schema.columns]
        col_types = [c.dtype for c in schema.columns]

        # 构建 VectorBatch
        self._batch = VectorBatch.from_rows(all_rows, col_names, col_types)
        self._emitted = False

    def next_batch(self) -> VectorBatch:
        """返回下一批数据。系统表一次返回全部。"""
        if self._emitted or self._batch is None:
            return None
        self._emitted = True
        return self._batch

    def close(self):
        """关闭算子，释放资源。"""
        self._batch = None
        self._emitted = False
