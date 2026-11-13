"""部分排序下的表达损失与组合目标向量。

不做默认补全，不虚构并列项之间或未列提案之间的先后：

* 每张有效选票按其**实际列出的层数**归一化。第 ``tier`` 层（0 起）
  被组合命中时该层损失为 0，未命中时损失为 ``(tier + 1) / 层数``，
  落在 [1/层数, 1] 之间；选票的表达损失取所有有效层的平均。
* 未列入 ranking 的提案被选中：既不加分也不减分，只作为
  "未表达提案"中性统计。
* ``unacceptable`` 命中是硬违规，单独计数，不折算成损失分。
* 因提案改版而失效的表达不参与损失，但其失效本身被逐条记录。

三个相互独立、不归并为综合分的目标：

1. ``cost``：组合成本（硬预算之外的实际花费，越低越好）；
2. ``total_loss``：全体出席代表表达损失的加权平均；
3. ``worst_group_loss``：最不满意群体的群体平均损失（最大化最小）。
"""

from __future__ import annotations

from dataclasses import dataclass

from deliberation_lake.preferences.feasibility import (
    FeasibilityIndex,
    mask_to_ids,
    rejection_violations,
)
from deliberation_lake.preferences.model import Ballot, Case


@dataclass(frozen=True)
class BallotOutcome:
    representative: str
    group: str
    loss: float
    ranked_hits: tuple[str, ...]
    unexpressed_selected: tuple[str, ...]
    rejected_hits: tuple[str, ...]
    violated: bool


@dataclass(frozen=True)
class PortfolioMetrics:
    mask: int
    cost: int
    total_loss: float
    worst_group_loss: float
    worst_groups: tuple[str, ...]
    group_losses: dict[str, float]
    covered_groups: tuple[str, ...]
    uncovered_groups: tuple[str, ...]
    violation_count: int
    violating_reps: tuple[str, ...]
    per_ballot: tuple[BallotOutcome, ...]


def ballot_loss(ballot: Ballot, selected: set[str]) -> BallotOutcome:
    valid = ballot.valid_ranked
    if valid:
        width = ballot.ranking_width
        tier_losses = []
        hits: list[str] = []
        by_tier: dict[int, list[str]] = {}
        for item in valid:
            by_tier.setdefault(item.tier, []).append(item.proposal_id)
        for tier in sorted(by_tier):
            tier_items = by_tier[tier]
            if any(pid in selected for pid in tier_items):
                tier_losses.append(0.0)
                hits.extend(pid for pid in tier_items if pid in selected)
            else:
                tier_losses.append((tier + 1) / width)
        loss = sum(tier_losses) / len(tier_losses)
    else:
        loss = 0.0  # 没有任何有效表达时不臆造损失，另行通过失效记录呈现。

    ranked_ids = {item.proposal_id for item in valid}
    unexpressed = tuple(sorted(pid for pid in selected if pid not in ranked_ids))
    rejected = tuple(
        sorted(
            item.proposal_id for item in ballot.valid_unacceptable
            if item.proposal_id in selected
        )
    )
    return BallotOutcome(
        representative=ballot.representative,
        group=ballot.group,
        loss=round(loss, 6),
        ranked_hits=tuple(sorted(hits)),
        unexpressed_selected=unexpressed,
        rejected_hits=rejected,
        violated=bool(rejected),
    )


class Scorer:
    """按给定席位权重把可行掩码评估为三维目标向量。"""

    def __init__(self, case: Case, index: FeasibilityIndex,
                 weights: dict[str, float] | None = None,
                 loss_lens: str = "all_tiers"):
        if loss_lens not in ("all_tiers", "top_tier"):
            raise ValueError(f"未知损失口径: {loss_lens}")
        self.case = case
        self.index = index
        # 部分排序聚合口径（不补全排序，只改变对"已表达"偏好的聚合）：
        # all_tiers：默认，按选票实际列出的全部层计归一化损失；
        # top_tier：对照口径，只看最高偏好层是否被命中（命中损失 0，
        #   否则 1），用于检验推荐是否依赖层级加权方式。两者都不虚构
        #   并列项先后，也不把未列提案当作反对。
        self.loss_lens = loss_lens
        # 默认权重：每位出席代表 1 票（重复提交已在 model 层去重）。
        present = case.present_reps()
        self._present = present
        if weights is None:
            weights = {group_id: float(len(reps))
                       for group_id, reps in present.items()}
        self.weights = dict(weights)
        self._ballot_weight: dict[str, float] = {}
        for group_id, reps in present.items():
            if reps:
                per_head = self.weights[group_id] / len(reps)
                for rep in reps:
                    self._ballot_weight[rep] = per_head
        # 记忆化：分支定界中同一掩码会被多次评估/支配比较，前沿与叶子数量
        # 都有界，缓存完整评估结果可避免重复计算（不缓存全部子集）。
        self._eval_cache: dict[int, PortfolioMetrics] = {}
        # 预计算每张选票各有效偏好层的位掩码与层损失，使叶子损失可用
        # 按位与 O(层数) 计算，避免逐提案构造集合。
        self._tier_masks: dict[str, tuple[tuple[int, float], ...]] = {}
        self._top_mask: dict[str, int] = {}
        self._ballot_group: dict[str, str] = {}
        for ballot in case.ballots:
            self._ballot_group[ballot.representative] = ballot.group
            by_tier: dict[int, int] = {}
            for item in ballot.valid_ranked:
                if item.proposal_id in index.positions:
                    by_tier[item.tier] = (
                        by_tier.get(item.tier, 0)
                        | (1 << index.positions[item.proposal_id])
                    )
            width = ballot.ranking_width
            rows = tuple(
                (by_tier[tier], (tier + 1) / width)
                for tier in sorted(by_tier)
            )
            self._tier_masks[ballot.representative] = rows
            if by_tier:
                self._top_mask[ballot.representative] = by_tier[min(by_tier)]

    def fast_ballot_loss(self, representative: str, mask: int) -> float:
        """按位计算单票表达损失；无有效表达时为 0（不臆造）。"""
        rows = self._tier_masks.get(representative)
        if not rows:
            return 0.0
        if self.loss_lens == "top_tier":
            # 对照口径：仅看最高偏好层，命中 0、未命中 1，不引入层间权重。
            return 0.0 if (mask & self._top_mask[representative]) else 1.0
        total = 0.0
        for tier_mask, miss_loss in rows:
            # 该层任一并列提案被选中即损失 0，否则承担该层未命中损失。
            total += 0.0 if (mask & tier_mask) else miss_loss
        return total / len(rows)

    def fast_objective(self, mask: int, cost: int) -> tuple[int, float, float]:
        """叶子快速目标 (cost, total_loss, worst_group_loss)，仅按位运算。"""
        per_ballot: dict[str, float] = {}
        for representative in self._tier_masks:
            per_ballot[representative] = self.fast_ballot_loss(representative, mask)
        total_weight = sum(self.weights.values())
        weighted_sum = sum(
            self._ballot_weight.get(rep, 0.0) * value
            for rep, value in per_ballot.items()
        )
        total_loss = weighted_sum / total_weight if total_weight else 0.0
        group_values: dict[str, list[float]] = {
            gid: [] for gid in self.case.group_ids()
        }
        for representative, value in per_ballot.items():
            group_values[self._ballot_group[representative]].append(value)
        worst = max(
            (sum(values) / len(values) for values in group_values.values() if values),
            default=0.0,
        )
        return cost, round(total_loss, 6), round(worst, 6)

    def evaluate(self, mask: int) -> PortfolioMetrics:
        cached = self._eval_cache.get(mask)
        if cached is not None:
            return cached
        result = self._evaluate_uncached(mask)
        self._eval_cache[mask] = result
        return result

    def _evaluate_uncached(self, mask: int) -> PortfolioMetrics:
        selected = set(mask_to_ids(mask, self.index.order))
        outcomes = tuple(
            ballot_loss(ballot, selected) for ballot in self.case.ballots
        )
        cost = sum(
            self.index.costs[i] for i in range(self.index.size)
            if mask & (1 << i)
        )
        # 全体加权平均损失。
        weighted_sum = sum(
            self._ballot_weight.get(outcome.representative, 0.0) * outcome.loss
            for outcome in outcomes
        )
        total_weight = sum(self.weights.values())
        total_loss = weighted_sum / total_weight if total_weight else 0.0

        group_losses: dict[str, float] = {}
        covered: list[str] = []
        uncovered: list[str] = []
        for group_id in self.case.group_ids():
            group_outcomes = [o for o in outcomes if o.group == group_id]
            if group_outcomes:
                group_losses[group_id] = (
                    sum(o.loss for o in group_outcomes) / len(group_outcomes)
                )
                # 群体覆盖：群体内至少一位代表无拒绝违规且命中其有效偏好。
                if any(not o.violated and o.ranked_hits for o in group_outcomes):
                    covered.append(group_id)
                else:
                    uncovered.append(group_id)
            else:
                group_losses[group_id] = 0.0
                uncovered.append(group_id)
        worst_value = max(group_losses.values()) if group_losses else 0.0
        worst_groups = tuple(
            sorted(gid for gid, value in group_losses.items()
                   if abs(value - worst_value) < 1e-9)
        )
        violations = rejection_violations(mask, self.case, self.index)
        violating_reps = tuple(sorted(violations))
        return PortfolioMetrics(
            mask=mask,
            cost=cost,
            total_loss=round(total_loss, 6),
            worst_group_loss=round(worst_value, 6),
            worst_groups=worst_groups,
            group_losses={gid: round(value, 6) for gid, value in group_losses.items()},
            covered_groups=tuple(sorted(covered)),
            uncovered_groups=tuple(sorted(uncovered)),
            violation_count=sum(len(ids) for ids in violations.values()),
            violating_reps=violating_reps,
            per_ballot=outcomes,
        )

    def actual_loss(self, mask: int) -> tuple[float, float]:
        metrics = self.evaluate(mask)
        return metrics.total_loss, metrics.worst_group_loss

    def objective(self, mask: int, cost: int) -> tuple[int, float, float]:
        """分支定界用快速目标回调 (cost, total_loss, worst_group_loss)。"""
        return self.fast_objective(mask, cost)

    def dominates(self, candidate_mask: int, incumbent_mask: int) -> bool:
        """candidate 是否在三维目标上支配 incumbent（拒绝违规不参与支配，
        违规组合通过独立的可接受前沿呈现）。"""
        candidate = self.evaluate(candidate_mask)
        incumbent = self.evaluate(incumbent_mask)
        better_or_equal = (
            candidate.cost <= incumbent.cost
            and candidate.total_loss <= incumbent.total_loss + 1e-12
            and candidate.worst_group_loss <= incumbent.worst_group_loss + 1e-12
        )
        strictly_better = (
            candidate.cost < incumbent.cost
            or candidate.total_loss < incumbent.total_loss - 1e-12
            or candidate.worst_group_loss < incumbent.worst_group_loss - 1e-12
        )
        return better_or_equal and strictly_better
