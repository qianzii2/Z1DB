from __future__ import annotations
"""Physical plan — annotated operator tree with cost estimates and chosen algorithms.
Bridges logical AST → physical operator tree with optimization decisions."""
from typing import Any, Dict, List, Optional, Tuple
from planner.cost_model import CostEstimate, CostModel
from planner.cardinality import CardinalityEstimator
from planner.rules import PredicatePushdown, PredicateReorder, TopNPushdown


class PhysicalPlanNode:
    """A node in the physical execution plan."""
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
    """Transforms AST + statistics into an optimized physical plan."""

    def __init__(self, estimator: Optional[CardinalityEstimator] = None) -> None:
        self._estimator = estimator or CardinalityEstimator()

    def plan(self, ast: Any, table_rows: Dict[str, int]) -> PhysicalPlanNode:
        """Generate optimized physical plan from AST."""
        from parser.ast import SelectStmt

        if not isinstance(ast, SelectStmt):
            return PhysicalPlanNode('UNKNOWN', CostEstimate())

        # Apply rules
        ast = PredicateReorder.apply(ast)

        # Build scan
        if ast.from_clause is None:
            return PhysicalPlanNode('DUAL_SCAN', CostEstimate(rows=1), output_rows=1)

        table_name = ast.from_clause.table.name
        rows = table_rows.get(table_name, 1000)
        scan_cost = CostModel.seq_scan(rows)
        plan = PhysicalPlanNode('SEQ_SCAN', scan_cost,
                                properties={'table': table_name}, output_rows=rows)

        # Filter
        if ast.where:
            sel = self._estimator.estimate_selectivity(ast.where, table_name)
            filter_cost = CostModel.filter(scan_cost, sel)
            plan = PhysicalPlanNode('FILTER', filter_cost, [plan],
                                    properties={'selectivity': f'{sel:.2%}'},
                                    output_rows=rows * sel)

        # Sort or TopN
        if ast.order_by:
            current_rows = plan.output_rows
            if TopNPushdown.should_use_top_n(ast):
                n = TopNPushdown.get_limit_value(ast)
                tn_cost = CostModel.top_n(plan.cost, n)
                plan = PhysicalPlanNode('TOP_N', tn_cost, [plan],
                                        properties={'n': n, 'algorithm': 'HEAP'},
                                        output_rows=min(n, current_rows))
            else:
                sort_cost = CostModel.sort(plan.cost)
                algo = 'PDQSORT' if current_rows < 100000 else 'EXTERNAL'
                plan = PhysicalPlanNode('SORT', sort_cost, [plan],
                                        properties={'algorithm': algo},
                                        output_rows=current_rows)

        return plan

    def explain_str(self, plan: PhysicalPlanNode) -> str:
        return plan.explain()
