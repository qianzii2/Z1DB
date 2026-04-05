from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCostModel:
    def test_seq_scan(self):
        from planner.cost_model import CostModel
        c = CostModel.seq_scan(10000)
        assert c.rows == 10000 and c.total > 0

    def test_topn_cheaper(self):
        from planner.cost_model import CostModel
        scan = CostModel.seq_scan(100000)
        full = CostModel.sort(scan)
        topn = CostModel.top_n(scan, 10)
        assert topn.total < full.total

    def test_join_selection(self):
        from planner.cost_model import CostModel
        left = CostModel.seq_scan(10000)
        right = CostModel.seq_scan(100)
        algo, cost = CostModel.select_join_algorithm(left, right)
        assert algo in ('HASH_JOIN', 'NESTED_LOOP', 'SORT_MERGE')


class TestCardinality:
    def test_selectivity(self):
        from planner.cardinality import CardinalityEstimator
        from parser.ast import BinaryExpr, ColumnRef, Literal
        from storage.types import DataType
        ce = CardinalityEstimator()
        eq = BinaryExpr(op='=', left=ColumnRef(column='x'),
                        right=Literal(value=5, inferred_type=DataType.INT))
        sel = ce.estimate_selectivity(eq)
        assert 0 < sel < 1


class TestDPccp:
    def test_star_schema(self):
        from planner.join_reorder import DPccp, JoinGraph
        g = JoinGraph()
        g.add_table('F', 1000000)
        g.add_table('D1', 100)
        g.add_table('D2', 50)
        g.add_edge('F', 'D1')
        g.add_edge('F', 'D2')
        dp = DPccp(g)
        result = dp.optimize()
        assert result.tables == frozenset({'F', 'D1', 'D2'})
        assert result.cost.total > 0


class TestRules:
    def test_predicate_reorder(self):
        from planner.rules import PredicateReorder
        from parser.ast import SelectStmt, BinaryExpr, ColumnRef, Literal
        from storage.types import DataType
        ast = SelectStmt(
            select_list=[ColumnRef(column='x')],
            where=BinaryExpr(op='AND',
                left=BinaryExpr(op='>', left=ColumnRef(column='x'),
                    right=Literal(value=5, inferred_type=DataType.INT)),
                right=BinaryExpr(op='=', left=ColumnRef(column='y'),
                    right=Literal(value=1, inferred_type=DataType.INT))))
        result = PredicateReorder.apply(ast)
        assert result.where is not None

    def test_topn_detection(self):
        from planner.rules import TopNPushdown
        from parser.ast import SelectStmt, SortKey, ColumnRef, Literal
        from storage.types import DataType
        ast = SelectStmt(
            select_list=[ColumnRef(column='x')],
            order_by=[SortKey(expr=ColumnRef(column='x'))],
            limit=Literal(value=10, inferred_type=DataType.INT))
        assert TopNPushdown.should_use_top_n(ast)
        assert TopNPushdown.get_limit_value(ast) == 10


def run_planner_tests():
    classes = [TestCostModel, TestCardinality, TestDPccp, TestRules]
    total = 0; passed = 0; failed = 0
    for cls in classes:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith('test_'): continue
            total += 1
            try:
                getattr(obj, name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{name}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {cls.__name__}.{name}: {e}")
    print(f"\nPlanner: {passed}/{total} passed, {failed} failed")
    return failed


if __name__ == '__main__':
    print("=== Planner Tests ===")
    run_planner_tests()
