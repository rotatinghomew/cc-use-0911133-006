"""组合分析主编排：非支配前沿、群体覆盖、循环证据与稳定性。"""

from __future__ import annotations

import time

from deliberation_lake.preferences.evidence import build_cycle_evidence
from deliberation_lake.preferences.feasibility import (
    FeasibilityIndex,
    branch_and_bound,
    build_index,
    enumerate_feasible,
    is_feasible,
    mask_to_ids,
    _mask_cost,
)
from deliberation_lake.preferences.model import Case, load_case
from deliberation_lake.preferences.scoring import Scorer
from deliberation_lake.preferences.sensitivity import (
    ScenarioResult,
    baseline_seat_map,
    scenario_suite,
    stability_table,
)

METHODOLOGY = {
    "hard_constraints": [
        "预算 budget 是硬边界：超预算组合直接不可行，不做软扣分",
        "depends_on 依赖是硬边界：缺少依赖的组合不可行",
        "exclusive_with 互斥是硬边界：互斥提案同时出现即不可行",
        "unacceptable 是硬边界：命中拒绝项的组合标记为违规，"
        "并从可接受前沿中剔除，但仍保留在完整前沿中供理事会看到代价",
    ],
    "partial_ranking": [
        "ranking 内层数组为并列项，并列之间不虚构先后",
        "未列出不代表反对；只有 unacceptable 表示拒绝",
        "表达损失按选票实际列出的层数归一化，未列提案被选中时中性记录、不加分减分",
        "不使用默认补全，不输出未经解释的综合分或唯一排名",
    ],
    "ballot_accounting": [
        "同一 representative 仅最高 revision 生效，旧票被取代且不增加席位",
        "选票通过 proposal_versions 声明版本；改版后仅当前版本或"
        "compatible_ballot_versions 白名单内的表达有效，其余逐条作废记录",
        "缺席代表不虚构偏好，只通过席位权重情景呈现其结构性影响",
    ],
    "objectives": [
        "cost：组合成本（越低越好）",
        "total_loss：全体出席代表表达损失加权平均（越低越好）",
        "worst_group_loss：最不满意群体的群体平均损失（越低越好，maximin 口径）",
        "三个目标独立呈现，仅做帕累托非支配判定，不归并为单一分数",
    ],
    "decision_boundary": (
        "本分析是协商材料，不写入承诺台账，不替代理事会决策；"
        "最终选择权在人。"
    ),
}


def preference_hit_masks(case: Case, index: FeasibilityIndex) -> dict[str, int]:
    """每张选票在各提案有效最高偏好层的命中位掩码（供贪心种子与报告）。"""
    hits: dict[str, int] = {}
    for ballot in case.ballots:
        valid = ballot.valid_ranked
        if not valid:
            continue
        top_tier = min(item.tier for item in valid)
        mask = 0
        for item in valid:
            if item.tier == top_tier and item.proposal_id in index.positions:
                mask |= 1 << index.positions[item.proposal_id]
        hits[ballot.representative] = mask
    merged: dict[str, int] = {}
    for mask in hits.values():
        for i in range(index.size):
            if mask & (1 << i):
                pid = index.order[i]
                merged[pid] = merged.get(pid, 0) | mask
    return merged


def portfolio_record(mask: int, scorer: Scorer,
                     index: FeasibilityIndex) -> dict[str, object]:
    metrics = scorer.evaluate(mask)
    return {
        "portfolio_id": "PF-" + format(mask, "0" + str(index.size) + "b")[::-1],
        "proposals": list(mask_to_ids(mask, index.order)),
        "cost": metrics.cost,
        "budget_remaining": scorer.case.budget - metrics.cost,
        "total_loss": metrics.total_loss,
        "worst_group_loss": metrics.worst_group_loss,
        "worst_groups": list(metrics.worst_groups),
        "group_losses": metrics.group_losses,
        "covered_groups": list(metrics.covered_groups),
        "uncovered_groups": list(metrics.uncovered_groups),
        "violation_count": metrics.violation_count,
        "violating_representatives": list(metrics.violating_reps),
        "acceptable": metrics.violation_count == 0,
        "per_ballot": [
            {
                "representative": outcome.representative,
                "group": outcome.group,
                "loss": outcome.loss,
                "ranked_hits": list(outcome.ranked_hits),
                "unexpressed_selected": list(outcome.unexpressed_selected),
                "rejected_hits": list(outcome.rejected_hits),
                "violated": outcome.violated,
            }
            for outcome in metrics.per_ballot
        ],
    }


def _verify_front_feasibility(front: list[int], index: FeasibilityIndex) -> None:
    for mask in front:
        if not is_feasible(mask, index):
            raise RuntimeError(f"内部校验失败：掩码 {mask} 不满足硬约束")


def _linear_pareto(feasible: list[int], index: FeasibilityIndex,
                   scorer: Scorer) -> list[int]:
    """用快速目标做线性帕累托过滤，作为穷举参考前沿（仅小案例）。"""
    front: list[int] = []
    front_obj: list[tuple[int, float, float]] = []
    for mask in feasible:
        cost = _mask_cost(mask, index)
        c_m, t_m, w_m = scorer.objective(mask, cost)
        dominated = False
        for c_o, t_o, w_o in front_obj:
            if (c_o <= c_m and t_o <= t_m + 1e-12 and w_o <= w_m + 1e-12
                    and (c_o < c_m or t_o < t_m - 1e-12 or w_o < w_m - 1e-12)):
                dominated = True
                break
        if dominated:
            continue
        kept = [(m, o) for m, o in zip(front, front_obj)
                if not (c_m <= o[0] and t_m <= o[1] + 1e-12
                        and w_m <= o[2] + 1e-12
                        and (c_m < o[0] or t_m < o[1] - 1e-12
                             or w_m < o[2] - 1e-12))]
        front = [item[0] for item in kept]
        front_obj = [item[1] for item in kept]
        front.append(mask)
        front_obj.append((c_m, t_m, w_m))
    return sorted(front)


def analyze_case(case: Case, exhaustive: bool,
                 display_cap: int = 50) -> dict[str, object]:
    started = time.monotonic()
    index = build_index(case)
    hits = preference_hit_masks(case, index)
    present_weights = {
        group_id: float(len(reps))
        for group_id, reps in case.present_reps().items()
    }
    scorer = Scorer(case, index, present_weights)

    exhaustive_check: dict[str, object] | None = None
    if exhaustive:
        feasible = enumerate_feasible(index)
        # 线性帕累托过滤得到穷举参考前沿（小案例规模小，可全量装入）。
        reference_front = _linear_pareto(feasible, index, scorer)
        front, stats = branch_and_bound(
            index, scorer.objective, hits
        )
        exhaustive_check = {
            "feasible_portfolio_count": len(feasible),
            "exhaustive_front_count": len(reference_front),
            "branch_and_bound_front_count": len(front),
            "fronts_match": set(front) == set(reference_front),
        }
        if set(front) != set(reference_front):
            raise RuntimeError("分支定界前沿与穷举前沿不一致")
    else:
        front, stats = branch_and_bound(
            index, scorer.objective, hits
        )
    _verify_front_feasibility(front, index)

    acceptable_front = [
        mask for mask in front if scorer.evaluate(mask).violation_count == 0
    ]
    records = [portfolio_record(mask, scorer, index) for mask in front]
    acceptable_records = [
        portfolio_record(mask, scorer, index) for mask in acceptable_front
    ]

    # 大案例的"群体可接受方案存在性"：独立的完整 DFS 遍历（不保存全部子集）。
    acceptable_existence = _acceptable_existence(case, index)

    scenarios = scenario_suite(case, index, exhaustive, hits)
    stability = stability_table(case, index, front, exhaustive, hits)
    elapsed = round(time.monotonic() - started, 4)

    return {
        "case": case.name,
        "budget": case.budget,
        "proposal_count": len(case.proposals),
        "search_space_size": 2 ** len(case.proposals),
        "exhaustive": exhaustive,
        "search": {
            **stats,
            "elapsed_seconds": elapsed,
            "memory_guarantee": (
                "分支定界仅保留当前 DFS 路径与现行非支配前沿，"
                "不枚举、不缓存全部子集"
            ),
        },
        "exhaustive_cross_check": exhaustive_check,
        "ballot_accounting": _ballot_accounting(case),
        "pareto_front": {
            "size": len(records),
            "displayed": min(display_cap, len(records)),
            "display_cap_note": (
                f"完整前沿含 {len(records)} 个组合，报告仅展示前 "
                f"{min(display_cap, len(records))} 个；全部记录见 "
                "front_records.json"
            ),
            "records": records[:display_cap],
        },
        "acceptable_front": {
            "size": len(acceptable_records),
            "records": acceptable_records[:display_cap],
        },
        "acceptable_option_existence": acceptable_existence,
        "group_coverage_summary": _coverage_summary(front, scorer),
        "cycle_evidence": build_cycle_evidence(case),
        "scenarios": [_scenario_to_dict(row, index) for row in scenarios],
        "stability": stability,
        "warnings": case.warnings,
        "methodology": METHODOLOGY,
    }


def _ballot_accounting(case: Case) -> dict[str, object]:
    present = case.present_reps()
    absent_by_group: dict[str, list[str]] = {gid: [] for gid in case.group_ids()}
    for row in case.absent_representatives:
        absent_by_group.setdefault(row["group"], []).append(row["representative"])
    return {
        "groups": [
            {
                "group_id": group["id"],
                "nominal_seats": group["seats"],
                "present_representatives": present[group["id"]],
                "absent_representatives": sorted(absent_by_group.get(group["id"], [])),
                "seat_inflation_check": (
                    "ok" if len(present[group["id"]]) <= group["seats"]
                    else "violation"
                ),
            }
            for group in case.groups
        ],
        "ballots": [
            {
                "representative": ballot.representative,
                "group": ballot.group,
                "applied_revision": ballot.revision,
                "superseded_revisions": list(ballot.superseded_revisions),
                "valid_ranked": [
                    {"proposal": item.proposal_id, "tier": item.tier,
                     "ballot_version": item.ballot_version, "basis": item.reason}
                    for item in ballot.valid_ranked
                ],
                "voided_ranked": [
                    {"proposal": item.proposal_id, "tier": item.tier,
                     "ballot_version": item.ballot_version, "reason": item.reason}
                    for item in ballot.ranked if not item.valid
                ],
                "valid_unacceptable": [
                    item.proposal_id for item in ballot.valid_unacceptable
                ],
                "voided_unacceptable": [
                    {"proposal": item.proposal_id,
                     "ballot_version": item.ballot_version, "reason": item.reason}
                    for item in ballot.unacceptable if not item.valid
                ],
            }
            for ballot in case.ballots
        ],
    }


def _coverage_summary(front: list[int], scorer: Scorer) -> dict[str, object]:
    coverage_count: dict[str, int] = {gid: 0 for gid in scorer.case.group_ids()}
    for mask in front:
        metrics = scorer.evaluate(mask)
        for gid in metrics.covered_groups:
            coverage_count[gid] += 1
    return {
        "front_size": len(front),
        "front_portfolios_covering_group": coverage_count,
        "groups_never_covered_on_front": sorted(
            gid for gid, count in coverage_count.items() if count == 0
        ),
    }


def _acceptable_existence(case: Case, index: FeasibilityIndex) -> dict[str, object]:
    """不保存全部子集的 DFS：逐群体判断是否存在可接受可行组合。"""
    result: dict[str, object] = {}
    present = case.present_reps()
    scorer = Scorer(case, index)
    for group_id in case.group_ids():
        reps = set(present[group_id])
        exists = False
        example: tuple[str, ...] | None = None
        if reps:
            for mask in _stream_feasible(index):
                metrics = scorer.evaluate(mask)
                outcomes = [o for o in metrics.per_ballot if o.group == group_id]
                if (all(not o.violated for o in outcomes)
                        and any(o.ranked_hits for o in outcomes)):
                    exists = True
                    example = tuple(mask_to_ids(mask, index.order))
                    break
        result[group_id] = {
            "has_acceptable_feasible_portfolio": exists,
            "example_portfolio": list(example) if example else None,
            "present_representatives": sorted(reps),
        }
    return result


def _stream_feasible(index: FeasibilityIndex):
    """以生成器逐个产出可行掩码，调用方不积累全集。

    与分支定界一致地只在相关提案（剔除中性提案后）上分支，位掩码仍基于
    完整拓扑序，保证与 Scorer/报告使用的掩码口径相同。
    """
    search_positions = [index.positions[pid] for pid in index.search_order]

    def dfs(step: int, chosen: int, cost: int) -> object:
        if step == len(search_positions):
            yield chosen
            return
        position = search_positions[step]
        bit = 1 << position
        # 不选当前提案。
        yield from dfs(step + 1, chosen, cost)
        # 满足硬约束时选入。
        new_cost = cost + index.costs[position]
        if (new_cost <= index.case.budget
                and not (index.dependency_mask[position] & ~chosen)
                and not (index.exclusive_mask[position] & chosen)):
            yield from dfs(step + 1, chosen | bit, new_cost)

    yield from dfs(0, 0, 0)


def _scenario_to_dict(row: ScenarioResult, index: FeasibilityIndex) -> dict[str, object]:
    return {
        "key": row.key,
        "description": row.description,
        "weights": row.weights,
        "front_size": row.front_size,
        "acceptable_front_size": row.acceptable_front_size,
        "groups_without_acceptable_option": list(
            row.groups_without_acceptable_option
        ),
        "anchors": {
            axis: [
                {
                    "portfolio_mask": mask,
                    "portfolio": list(mask_to_ids(mask, index.order)),
                }
                for mask in masks
            ]
            for axis, masks in row.anchors.items()
        },
        "search_stats": row.search_stats,
    }


def run_analysis(fixture_path: str, output_dir: str,
                 display_cap: int = 50) -> dict[str, object]:
    loaded = load_case(fixture_path)
    small = analyze_case(loaded["small"], exhaustive=True, display_cap=display_cap)
    large = None
    if loaded["large"] is not None:
        large = analyze_case(loaded["large"], exhaustive=False,
                             display_cap=display_cap)
    return {
        "seed": loaded["seed"],
        "small_case": small,
        "large_case": large,
        "disclaimer": METHODOLOGY["decision_boundary"],
    }
