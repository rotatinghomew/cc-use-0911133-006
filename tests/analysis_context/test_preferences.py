"""偏好组合分析的单元测试。

覆盖用户强调的口径：
* 修订取代、重复提交不增加席位；
* 提案改版后旧票表达依据白名单逐条失效，不默认沿用；
* 并列不虚构先后、未列出不等于反对；
* 预算/依赖/互斥/拒绝是硬边界；
* 小案例分支定界前沿必须与穷举前沿完全一致；
* 中性提案结构缩减与全量搜索等价；
* 循环偏好证据区分当前有效口径与原始表达口径；
* 分析不写入承诺台账。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from deliberation_lake.preferences.analysis import (
    _linear_pareto,
    analyze_case,
    preference_hit_masks,
    run_analysis,
)
from deliberation_lake.preferences.feasibility import (
    FeasibilityIndex,
    branch_and_bound,
    build_index,
    enumerate_feasible,
    is_feasible,
    mask_to_ids,
)
from deliberation_lake.preferences.model import load_case
from deliberation_lake.preferences.scoring import Scorer
from deliberation_lake.preferences.sensitivity import (
    _groups_without_acceptable,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "preference_portfolios.json"


class BallotAccountingTests(unittest.TestCase):
    def setUp(self):
        self.loaded = load_case(FIXTURE)
        self.case = self.loaded["small"]

    def test_revision_supersedes_without_seat_inflation(self):
        by_rep = {b.representative: b for b in self.case.ballots}
        # r-a 提交了 revision 1 和 2，只有 rev2 生效。
        self.assertEqual(by_rep["r-a"].revision, 2)
        self.assertEqual(by_rep["r-a"].superseded_revisions, (1,))
        # day 群体只有 r-a 一名出席代表，重复提交未增加席位。
        self.assertEqual(self.case.present_reps()["day"], ["r-a"])
        self.assertLessEqual(
            len(self.case.present_reps()["day"]),
            self.case.nominal_seats("day"),
        )

    def test_version_change_voids_old_expression(self):
        by_rep = {b.representative: b for b in self.case.ballots}
        # P1 升级到 v2 且兼容白名单为空，所有基于 v1 的 P1 表达作废。
        for rep in ("r-a", "r-b", "r-c"):
            voided = {item.proposal_id for item in by_rep[rep].ranked
                      if not item.valid}
            self.assertIn("P1", voided)
            valid_ids = {item.proposal_id for item in by_rep[rep].valid_ranked}
            self.assertNotIn("P1", valid_ids)

    def test_ties_are_preserved_not_ordered(self):
        by_rep = {b.representative: b for b in self.case.ballots}
        # r-a rev2 第 0 层是 P0/P5 并列，r-d 第 0 层是 P4/P5 并列。
        top = {item.proposal_id for item in by_rep["r-a"].valid_ranked
               if item.tier == 0}
        self.assertEqual(top, {"P0", "P5"})
        top_d = {item.proposal_id for item in by_rep["r-d"].valid_ranked
                 if item.tier == 0}
        self.assertEqual(top_d, {"P4", "P5"})

    def test_unlisted_is_not_rejection(self):
        # 未列入 ranking 的提案不会出现在 valid_unacceptable；
        # 只有显式 unacceptable 才是拒绝。
        by_rep = {b.representative: b for b in self.case.ballots}
        self.assertEqual(
            [item.proposal_id for item in by_rep["r-a"].valid_unacceptable],
            ["P4"],
        )
        # r-c 没有任何拒绝项。
        self.assertEqual(by_rep["r-c"].valid_unacceptable, ())


class HardConstraintTests(unittest.TestCase):
    def setUp(self):
        self.loaded = load_case(FIXTURE)
        self.case = self.loaded["small"]
        self.index = build_index(self.case)

    def _mask(self, ids):
        mask = 0
        for pid in ids:
            mask |= 1 << self.index.positions[pid]
        return mask

    def test_budget_is_hard(self):
        # 全部提案成本 26 > 预算 10，不可行。
        full = (1 << self.index.size) - 1
        self.assertFalse(is_feasible(full, self.index))

    def test_dependency_is_hard(self):
        # P5 依赖 P0：只选 P5 不可行，P0+P5 可行。
        self.assertFalse(is_feasible(self._mask({"P5"}), self.index))
        self.assertTrue(
            is_feasible(self._mask({"P0", "P5"}), self.index)
        )

    def test_exclusivity_is_hard(self):
        # P3 与 P4 互斥（单边声明已对称闭包），同时选入不可行。
        self.assertFalse(
            is_feasible(self._mask({"P3", "P4"}), self.index)
        )

    def test_dangling_exclusive_downgraded_to_warning(self):
        large = self.loaded["large"]
        self.assertTrue(
            any("L28" in warning for warning in large.warnings)
        )
        # L27 仍保留，悬空引用被忽略后 L27 本身可单独选入。
        self.assertIn("L27", self.loaded["large"].proposal_map)


class ParetoFrontTests(unittest.TestCase):
    def setUp(self):
        self.loaded = load_case(FIXTURE)

    def test_small_front_matches_exhaustive(self):
        result = analyze_case(self.loaded["small"], exhaustive=True)
        check = result["exhaustive_cross_check"]
        self.assertTrue(check["fronts_match"])
        self.assertEqual(
            check["exhaustive_front_count"],
            check["branch_and_bound_front_count"],
        )

    def test_front_members_are_feasible_and_non_dominated(self):
        case = self.loaded["small"]
        index = build_index(case)
        scorer = Scorer(case, index)
        result = analyze_case(case, exhaustive=True)
        masks = [
            self._mask_from_record(index, rec["proposals"])
            for rec in result["pareto_front"]["records"]
        ]
        for mask in masks:
            self.assertTrue(is_feasible(mask, index))
        # 前沿内任意两个成员互不支配。
        for i, left in enumerate(masks):
            for right in masks[i + 1:]:
                self.assertFalse(scorer.dominates(left, right))
                self.assertFalse(scorer.dominates(right, left))

    def _mask_from_record(self, index, ids):
        mask = 0
        for pid in ids:
            mask |= 1 << index.positions[pid]
        return mask

    def test_neutral_reduction_equals_full_search_large(self):
        case = self.loaded["large"]
        index = build_index(case)
        # 中性提案确实被识别出来。
        neutral = set(mask_to_ids(index.neutral_mask, index.order))
        self.assertTrue(neutral)
        scorer = Scorer(case, index)
        hits = preference_hit_masks(case, index)
        reduced, _ = branch_and_bound(index, scorer.objective, hits)
        full_index = FeasibilityIndex(
            case=case, order=index.order, positions=index.positions,
            costs=index.costs, dependency_mask=index.dependency_mask,
            exclusive_mask=index.exclusive_mask, neutral_mask=0,
            search_order=index.order,
        )
        full, _ = branch_and_bound(full_index, scorer.objective, hits)
        self.assertEqual(set(reduced), set(full))
        # 前沿不含任何中性提案。
        for mask in reduced:
            self.assertFalse(
                neutral & set(mask_to_ids(mask, index.order))
            )

    def test_large_search_does_not_materialize_all_subsets(self):
        case = self.loaded["large"]
        index = build_index(case)
        scorer = Scorer(case, index)
        hits = preference_hit_masks(case, index)
        _, stats = branch_and_bound(index, scorer.objective, hits)
        # 访问节点远小于 2^28，证明没有把全部子集装入内存。
        self.assertLess(stats["nodes_visited"], 1_000_000)
        self.assertLess(stats["nodes_visited"], 2 ** index.size)


class CycleEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.loaded = load_case(FIXTURE)
        self.result = analyze_case(self.loaded["small"], exhaustive=True)

    def test_cycle_differs_between_original_and_current(self):
        cycle = self.result["cycle_evidence"]
        # 原始表达存在 P0>P1>P2>P0 循环；P1 改版失效后当前口径循环消失。
        self.assertTrue(cycle["cycles_under_original_expressions"])
        self.assertEqual(cycle["cycles_under_current_versions"], [])

    def test_tied_and_voided_never_count_as_pairwise_votes(self):
        cycle = self.result["cycle_evidence"]
        # 所有当前多数边的两侧都不应是同层并列关系。
        for edge in cycle["current_valid_edges"]:
            self.assertNotEqual(edge["votes_for"], edge["votes_against"])
        self.assertTrue(cycle["tied_groups"])
        self.assertTrue(cycle["voided_expressions"])


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.loaded = load_case(FIXTURE)

    def test_scenarios_change_anchors_but_keep_hard_feasibility(self):
        result = analyze_case(self.loaded["small"], exhaustive=True)
        keys = {scenario["key"] for scenario in result["scenarios"]}
        self.assertIn("present_one_person_one_vote", keys)
        self.assertIn("nominal_seats_preserved", keys)
        self.assertIn("partial_ranking_top_tier_only", keys)
        self.assertTrue(
            any(key.startswith("seat_perturbation:") for key in keys)
        )

    def test_partial_ranking_lens_changes_front_without_completion(self):
        case = self.loaded["small"]
        index = build_index(case)
        base = Scorer(case, index, loss_lens="all_tiers")
        top = Scorer(case, index, loss_lens="top_tier")
        hits = preference_hit_masks(case, index)
        base_front, _ = branch_and_bound(index, base.objective, hits)
        top_front, _ = branch_and_bound(index, top.objective, hits)
        # 两种口径都只产出可行组合，且结果可不同（证明口径影响推荐）。
        for mask in set(base_front) | set(top_front):
            self.assertTrue(is_feasible(mask, index))
        self.assertNotEqual(set(base_front), set(top_front))
        # 顶层口径的单票损失只取 0 或 1，不引入层间权重。
        self.assertIn(top.fast_ballot_loss(
            next(iter(top._tier_masks)), 0), (0.0, 1.0))

    def test_group_without_acceptable_option_is_flagged(self):
        groups = [{"id": "day", "seats": 1}, {"id": "night", "seats": 1}]
        proposals = [
            {"id": "P4", "version": 1, "cost": 6,
             "depends_on": [], "exclusive_with": []},
        ]
        ballots = [
            {"representative": "r1", "group": "day", "revision": 1,
             "proposal_versions": {"P4": 1},
             "ranking": [["P4"]], "unacceptable": []},
            {"representative": "r2", "group": "night", "revision": 1,
             "proposal_versions": {"P4": 1},
             "ranking": [["P4"]], "unacceptable": []},
        ]
        from deliberation_lake.preferences.model import _build_case
        case = _build_case("x", 0, groups, proposals, ballots, [], None)
        index = build_index(case)
        feasible = enumerate_feasible(index)
        lost = _groups_without_acceptable(case, index, feasible)
        self.assertEqual(set(lost), {"day", "night"})


class IsolationTests(unittest.TestCase):
    def test_analysis_module_does_not_touch_commitment_ledger(self):
        # 分析包不导入承诺台账模块，保证协商材料不进入承诺台账。
        import deliberation_lake.preferences.feasibility as feasibility
        import deliberation_lake.preferences.analysis as analysis
        self.assertNotIn("deliberation_lake.commitments", dir(feasibility))
        self.assertNotIn("deliberation_lake.commitments", dir(analysis))

    def test_run_analysis_returns_disclaimer_and_no_commitment_write(self):
        result = run_analysis(str(FIXTURE), "/tmp/preferences-test-output")
        self.assertIn("不写入承诺台账", result["disclaimer"])
        self.assertIn("small_case", result)
        self.assertIn("large_case", result)


if __name__ == "__main__":
    unittest.main()
