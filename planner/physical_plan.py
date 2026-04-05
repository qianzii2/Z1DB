from __future__ import annotations
"""物理计划 — 带代价估算和算法选择的算子树。扩展支持JOIN/GROUP BY。"""
from typing import Any, Dict, List, Optional, Tuple
from planner.cost_model import CostEstimate, CostModel
from planner.cardinality import CardinalityEstimator
from planner.rules import PredicatePushdown, PredicateReorder, TopNPushdown


class PhysicalPlanNode:
    """物理执行计划节点。"""
    __slots__ = ('op_type', 'cost', 'children', 'properties', 'output_rows')

    def __init__(self, op_type: str, cost: CostEstimate,
                 children: Optional[List[PhysicalPlanNode]] = None,
                 properties: Optional[Dict] = None,
                 output_rows: float = 0) -> None:
        self.op_type = op_type
        self.cost = cost
        self.children = children or []
        self.properties = properties or {}
        self.output_rows = output_rows

    def explain(self, indent: int = 0) -> str:
        prefix = '  ' * indent
        props = ', '.join(f'{k}={v}' for k, v in self.properties.items())
        line = (f"{prefix}{self.op_type}"
                f" (rows={self.output_rows:.0f}, cost={self.cost.total:.1f}"
                f"{', ' + props if props else ''})")
        lines = [line]
        for child in self.children:
            lines.append(child.explain(indent + 1))
        return '\n'.join(lines)

    @property
    def total_cost(self) -> float:
        return self.cost.total


class PhysicalPlanner:
    """从AST+统计信息生成优化的物理计划。"""

    def __init__(self, estimator: Optional[CardinalityEstimator] = None) -> None:
        self._estimator = estimator or CardinalityEstimator()

    def plan(self, ast: Any, table_rows: Dict[str, int]) -> PhysicalPlanNode:
        from parser.ast import SelectStmt
        if not isinstance(ast, SelectStmt):
            return PhysicalPlanNode('UNKNOWN', CostEstimate())

        ast = PredicateReorder.apply(ast)

        if ast.from_clause is None:
            return PhysicalPlanNode('DUAL_SCAN', CostEstimate(rows=1), output_rows=1)

        # 构建扫描
        table_name = ast.from_clause.table.name
        rows = table_rows.get(table_name, 1000)
        scan_cost = CostModel.seq_scan(rows)
        plan = PhysicalPlanNode(
            'SEQ_SCAN', scan_cost,
            properties={'table': table_name}, output_rows=rows)

        # JOIN
        if ast.from_clause.joins:
            plan = self._plan_joins(ast, plan, table_rows, rows)

        # 过滤
        if ast.where:
            sel = self._estimator.estimate_selectivity(ast.where, table_name)
            filter_cost = CostModel.filter(scan_cost, sel)
            plan = PhysicalPlanNode(
                'FILTER', filter_cost, [plan],
                properties={'selectivity': f'{sel:.2%}'},
                output_rows=plan.output_rows * sel)

        # GROUP BY
        if ast.group_by:
            group_cols = [str(k) for k in ast.group_by.keys]
            ndv = self._estimator.estimate_group_ndv(table_name, ast.group_by.keys)
            agg_cost = CostModel.hash_agg(plan.cost, ndv)
            plan = PhysicalPlanNode(
                'HASH_AGG', agg_cost, [plan],
                properties={'groups': ndv, 'keys': ','.join(group_cols)},
                output_rows=ndv)

        # 排序或TopN
        if ast.order_by:
            current_rows = plan.output_rows
            if TopNPushdown.should_use_top_n(ast):
                n = TopNPushdown.get_limit_value(ast)
                tn_cost = CostModel.top_n(plan.cost, n)
                plan = PhysicalPlanNode(
                    'TOP_N', tn_cost, [plan],
                    properties={'n': n, 'algorithm': 'HEAP'},
                    output_rows=min(n, current_rows))
            else:
                sort_cost = CostModel.sort(plan.cost)
                algo = 'PDQSORT' if current_rows < 100000 else 'EXTERNAL'
                plan = PhysicalPlanNode(
                    'SORT', sort_cost, [plan],
                    properties={'algorithm': algo},
                    output_rows=current_rows)

        # LIMIT
        if ast.limit is not None:
            from parser.ast import Literal
            if isinstance(ast.limit, Literal) and isinstance(ast.limit.value, int):
                lim = ast.limit.value
                plan = PhysicalPlanNode(
                    'LIMIT', plan.cost, [plan],
                    properties={'limit': lim},
                    output_rows=min(lim, plan.output_rows))

        return plan

    def _plan_joins(self, ast, base_plan, table_rows, base_rows):
        """构建JOIN物理计划。"""
        current = base_plan
        current_rows = base_rows
        for jc in ast.from_clause.joins:
            if jc.table is None:
                continue
            right_name = jc.table.name
            right_rows = table_rows.get(right_name, 1000)
            right_cost = CostModel.seq_scan(right_rows)
            right_plan = PhysicalPlanNode(
                'SEQ_SCAN', right_cost,
                properties={'table': right_name}, output_rows=right_rows)

            if jc.join_type == 'CROSS':
                cross_rows = current_rows * right_rows
                cross_cost = CostModel.nested_loop_join(
                    current.cost, right_cost, 1.0)
                current = PhysicalPlanNode(
                    'CROSS_JOIN', cross_cost, [current, right_plan],
                    output_rows=cross_rows)
                current_rows = cross_rows
            else:
                # 选择最佳JOIN算法
                algo_name, join_cost = CostModel.select_join_algorithm(
                    current.cost, right_cost, selectivity=0.1)
                join_rows = max(1, current_rows * right_rows * 0.1)
                current = PhysicalPlanNode(
                    algo_name, join_cost, [current, right_plan],
                    properties={'type': jc.join_type},
                    output_rows=join_rows)
                current_rows = join_rows
        return current

    def explain_str(self, plan: PhysicalPlanNode) -> str:
        return plan.explain()
