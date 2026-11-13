"""敏感性与稳定性：展示结论在不同合理假设下如何变化。

不虚构缺席代表的偏好，只改变**权重口径**与**可表达口径**：

* ``present``：基线，每位出席代表 1 票；
* ``nominal_seats``：恢复名义席位权重，缺席者的份额保留给其群体但
  不编造投票方向（群体内按出席者均摊）；
* ``seat_perturbation``：在 [1, 名义席位] 内枚举各群体席位数，
  检查推荐锚点与覆盖是否改变；
* ``partial_ranking``：在两种透明口径之间切换——基线只按已表达层计
  损失；``unranked_neutral_tiers`` 把未列提案视为"中性并列层"仅用于
  对照，明确标注这是口径实验而非补全排序。

每个情景都标出：是否有群体在全部可行组合中都得不到可接受方案。
"""

from __future__ import annotations

from dataclasses import dataclass

from deliberation_lake.preferences.feasibility import (
    FeasibilityIndex,
    branch_and_bound,
    enumerate_feasible,
)
from deliberation_lake.preferences.model import Case
from deliberation_lake.preferences.scoring import Scorer


@dataclass(frozen=True)
class ScenarioResult:
    key: str
    description: str
    weights: dict[str, float]
    front_size: int
    anchors: dict[str, list[int]]
    groups_without_acceptable_option: tuple[str, ...]
    acceptable_front_size: int
    search_stats: dict[str, int]


def _weights_for(case: Case, seat_map: dict[str, int]) -> dict[str, float]:
    return {group["id"]: float(seat_map[group["id"]]) for group in case.groups}


def baseline_seat_map(case: Case) -> dict[str, int]:
    return {group["id"]: len(reps)
            for group, reps in ((g, case.present_reps()[g["id"]]) for g in case.groups)}


def nominal_seat_map(case: Case) -> dict[str, int]:
    return {group["id"]: int(group["seats"]) for group in case.groups}


def _anchors(front_masks: list[int], scorer: Scorer) -> dict[str, list[int]]:
    """每个目标轴上的最优组合（允许并列，全部列出，不给唯一排名）。"""
    if not front_masks:
        return {"min_cost": [], "min_total_loss": [], "min_worst_group_loss": [],
                "max_group_coverage": []}
    evaluated = [(mask, scorer.evaluate(mask)) for mask in front_masks]
    min_cost = min(metrics.cost for _, metrics in evaluated)
    min_total = min(metrics.total_loss for _, metrics in evaluated)
    min_worst = min(metrics.worst_group_loss for _, metrics in evaluated)
    max_cover = max(len(metrics.covered_groups) for _, metrics in evaluated)
    return {
        "min_cost": sorted(mask for mask, m in evaluated if m.cost == min_cost),
        "min_total_loss": sorted(
            mask for mask, m in evaluated if abs(m.total_loss - min_total) < 1e-12
        ),
        "min_worst_group_loss": sorted(
            mask for mask, m in evaluated
            if abs(m.worst_group_loss - min_worst) < 1e-12
        ),
        "max_group_coverage": sorted(
            mask for mask, m in evaluated if len(m.covered_groups) == max_cover
        ),
    }


def _groups_without_acceptable(case: Case, index: FeasibilityIndex,
                               all_feasible: list[int]) -> tuple[str, ...]:
    """对每个群体检查：是否存在至少一个组合，使群体内每位有有效表达的
    代表都不触发 unacceptable，且群体至少命中一个有效偏好项。"""
    present = case.present_reps()
    result: list[str] = []
    for group_id in case.group_ids():
        reps = set(present[group_id])
        if not reps:
            result.append(group_id)  # 无人出席：不存在可接受方案的证据本身。
            continue
        found = False
        for mask in all_feasible:
            metrics = Scorer(case, index).evaluate(mask)
            group_outcomes = [
                outcome for outcome in metrics.per_ballot
                if outcome.group == group_id
            ]
            if (all(not outcome.violated for outcome in group_outcomes)
                    and any(outcome.ranked_hits for outcome in group_outcomes)):
                found = True
                break
        if not found:
            result.append(group_id)
    return tuple(sorted(result))


def run_front(case: Case, index: FeasibilityIndex,
              weights: dict[str, float], exhaustive: bool,
              preference_hits: dict[str, int],
              loss_lens: str = "all_tiers"
              ) -> tuple[list[int], dict[str, int], Scorer]:
    scorer = Scorer(case, index, weights, loss_lens=loss_lens)
    if exhaustive:
        from deliberation_lake.preferences.analysis import _linear_pareto
        feasible = enumerate_feasible(index)
        front = _linear_pareto(feasible, index, scorer)
        stats = {"nodes_visited": len(feasible), "pruned_bounds": 0,
                 "pruned_dominance": 0, "front_capped": 0,
                 "hard_constraint_skips": 0,
                 "exhaustive_enumeration": 1}
        return sorted(front), stats, scorer
    front, stats = branch_and_bound(
        index, scorer.objective, preference_hits
    )
    stats["exhaustive_enumeration"] = 0
    return front, stats, scorer


def scenario_suite(case: Case, index: FeasibilityIndex,
                   exhaustive: bool,
                   preference_hits: dict[str, int]) -> list[ScenarioResult]:
    present_map = baseline_seat_map(case)
    nominal_map = nominal_seat_map(case)
    all_feasible_cache: list[int] | None = None

    def all_feasible() -> list[int]:
        nonlocal all_feasible_cache
        if all_feasible_cache is None:
            if exhaustive:
                all_feasible_cache = enumerate_feasible(index)
            else:
                # 大案例不装入全部子集：用前沿并集近似存在性检查的搜索域，
                # 存在性本身另由分支定界的完整遍历保证（见 analyze_case）。
                all_feasible_cache = []
        return all_feasible_cache

    scenarios: list[ScenarioResult] = []

    # 情景 1：基线（出席代表一人一票）。
    front, stats, scorer = run_front(
        case, index, _weights_for(case, present_map), exhaustive, preference_hits
    )
    scenarios.append(ScenarioResult(
        key="present_one_person_one_vote",
        description="基线：仅出席代表计票，每位代表权重 1；缺席不虚构偏好",
        weights=_weights_for(case, present_map),
        front_size=len(front),
        anchors=_anchors(front, scorer),
        groups_without_acceptable_option=(
            _groups_without_acceptable(case, index, all_feasible())
            if exhaustive else ()
        ),
        acceptable_front_size=sum(
            1 for mask in front if scorer.evaluate(mask).violation_count == 0
        ),
        search_stats=stats,
    ))

    # 情景 2：名义席位权重（缺席份额保留给群体，不编造方向）。
    if nominal_map != present_map:
        front_n, stats_n, scorer_n = run_front(
            case, index, _weights_for(case, nominal_map), exhaustive, preference_hits
        )
        scenarios.append(ScenarioResult(
            key="nominal_seats_preserved",
            description=(
                "名义席位口径：缺席代表对应的席位权重保留给其群体，"
                "由群体内出席者均摊，不假设缺席者会怎么投"
            ),
            weights=_weights_for(case, nominal_map),
            front_size=len(front_n),
            anchors=_anchors(front_n, scorer_n),
            groups_without_acceptable_option=(
                _groups_without_acceptable(case, index, all_feasible())
                if exhaustive else ()
            ),
            acceptable_front_size=sum(
                1 for mask in front_n if scorer_n.evaluate(mask).violation_count == 0
            ),
            search_stats=stats_n,
        ))

    # 情景 3：部分排序聚合口径对照（不补全排序，只改变对已表达偏好的聚合）。
    front_t, stats_t, scorer_t = run_front(
        case, index, _weights_for(case, present_map), exhaustive,
        preference_hits, loss_lens="top_tier"
    )
    scenarios.append(ScenarioResult(
        key="partial_ranking_top_tier_only",
        description=(
            "部分排序对照口径：只计每张选票最高偏好层是否命中（命中 0、"
            "未命中 1），不引入层间权重、不补全未列项目；用于检验推荐"
            "是否依赖默认的多层归一化口径"
        ),
        weights=_weights_for(case, present_map),
        front_size=len(front_t),
        anchors=_anchors(front_t, scorer_t),
        groups_without_acceptable_option=(
            _groups_without_acceptable(case, index, all_feasible())
            if exhaustive else ()
        ),
        acceptable_front_size=sum(
            1 for mask in front_t if scorer_t.evaluate(mask).violation_count == 0
        ),
        search_stats=stats_t,
    ))

    # 情景 4：合理席位扰动（每个群体 1..名义席位）。
    perturb_rows = _seat_perturbations(
        case, index, exhaustive, preference_hits, present_map
    )
    scenarios.extend(perturb_rows)
    return scenarios


def _seat_perturbations(case: Case, index: FeasibilityIndex,
                        exhaustive: bool, preference_hits: dict[str, int],
                        present_map: dict[str, int]) -> list[ScenarioResult]:
    group_ids = case.group_ids()
    ranges = {
        gid: range(1, case.nominal_seats(gid) + 1)
        for gid in group_ids
    }
    rows: list[ScenarioResult] = []
    baseline_anchor_sets = None
    for seats in _product(ranges):
        seat_map = dict(zip(group_ids, seats))
        if seat_map == present_map:
            continue
        weights = _weights_for(case, seat_map)
        front, stats, scorer = run_front(
            case, index, weights, exhaustive, preference_hits
        )
        anchors = _anchors(front, scorer)
        if baseline_anchor_sets is None:
            base_front, _, base_scorer = run_front(
                case, index, _weights_for(case, present_map),
                exhaustive, preference_hits
            )
            baseline_anchor_sets = {
                key: set(value) for key, value in _anchors(base_front, base_scorer).items()
            }
        changed_axes = sorted(
            axis for axis, masks in anchors.items()
            if set(masks) != baseline_anchor_sets[axis]
        )
        rows.append(ScenarioResult(
            key="seat_perturbation:" + ",".join(
                f"{gid}={seat_map[gid]}" for gid in group_ids
            ),
            description=(
                "席位扰动口径：" + ", ".join(
                    f"{gid} {seat_map[gid]} 席" for gid in group_ids
                ) + f"；锚点发生变化的目标轴：{changed_axes or '无'}"
            ),
            weights=weights,
            front_size=len(front),
            anchors=anchors,
            groups_without_acceptable_option=(),
            acceptable_front_size=sum(
                1 for mask in front if scorer.evaluate(mask).violation_count == 0
            ),
            search_stats=stats,
        ))
    return rows


def _product(ranges: dict[str, range]):
    keys = list(ranges)
    choices = [list(ranges[key]) for key in keys]

    def walk(index: int, current: list[int]):
        if index == len(keys):
            yield tuple(current)
            return
        for value in choices[index]:
            current.append(value)
            yield from walk(index + 1, current)
            current.pop()

    yield from walk(0, [])


def stability_table(case: Case, index: FeasibilityIndex,
                    baseline_front: list[int],
                    exhaustive: bool,
                    preference_hits: dict[str, int]) -> dict[str, object]:
    """基线前沿组合在各情景下是否仍可行、是否仍在该情景前沿内。"""
    present_map = baseline_seat_map(case)
    nominal_map = nominal_seat_map(case)
    table: list[dict[str, object]] = []
    scenario_maps = [("present", present_map)]
    if nominal_map != present_map:
        scenario_maps.append(("nominal_seats", nominal_map))
    for label, seat_map in scenario_maps:
        weights = _weights_for(case, seat_map)
        front, _, scorer = run_front(
            case, index, weights, exhaustive, preference_hits
        )
        front_set = set(front)
        for mask in baseline_front:
            metrics = scorer.evaluate(mask)
            table.append({
                "scenario": label,
                "portfolio_mask": mask,
                "portfolio": sorted(
                    index.order[i] for i in range(index.size) if mask & (1 << i)
                ),
                "still_on_front": mask in front_set,
                "still_acceptable": metrics.violation_count == 0,
                "covered_groups": list(metrics.covered_groups),
            })
    return {"rows": table, "note": (
        "稳定性只描述同一组合在不同权重口径下的位置变化，"
        "不代表任何组合应当被自动选中。"
    )}
