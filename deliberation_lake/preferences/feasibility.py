"""可行性硬边界与组合搜索。

预算、提案依赖、互斥、选票 ``unacceptable`` 全部是**硬边界**：
违反任意一条的组合不进入非支配前沿（拒绝违规会单独统计，绝不折算成
加减分的软指标）。

搜索方式：

* 小案例：完整枚举全部 2^n 子集，作为分支定界结果的核对基准；
* 大案例：按拓扑序分支定界，任何时刻只保留"当前路径 + 现行非支配前沿"，
  不把全部子集装入内存。
"""

from __future__ import annotations

from dataclasses import dataclass

from deliberation_lake.preferences.model import Case


@dataclass(frozen=True)
class FeasibilityIndex:
    case: Case
    order: tuple[str, ...]                      # 全部提案的拓扑序（依赖在前）
    positions: dict[str, int]
    costs: tuple[int, ...]                      # 按 order 排列
    dependency_mask: tuple[int, ...]           # 每个提案所需依赖的位掩码
    exclusive_mask: tuple[int, ...]            # 每个提案互斥对象的位掩码
    neutral_mask: int                           # 可证剔除的中性提案位掩码
    search_order: tuple[str, ...]               # 实际参与搜索的相关提案拓扑序

    @property
    def size(self) -> int:
        return len(self.order)

    @property
    def search_size(self) -> int:
        return len(self.search_order)


def build_index(case: Case) -> FeasibilityIndex:
    proposal_map = case.proposal_map
    remaining = set(proposal_map)
    order: list[str] = []
    # Kahn 拓扑：依赖必须排在前面。
    while remaining:
        ready = sorted(
            pid for pid in remaining
            if all(dep not in remaining for dep in proposal_map[pid].depends_on)
        )
        if not ready:
            raise ValueError("提案依赖存在环，无法建立拓扑序")
        order.extend(ready)
        remaining.difference_update(ready)
    positions = {pid: index for index, pid in enumerate(order)}
    costs = tuple(proposal_map[pid].cost for pid in order)
    dependency_mask = tuple(
        _mask_of(proposal_map[pid].depends_on, positions) for pid in order
    )
    exclusive_mask = tuple(
        _mask_of(proposal_map[pid].exclusive_with, positions) for pid in order
    )
    neutral_mask, search_order = _structural_reduction(case, order, positions)
    return FeasibilityIndex(
        case=case,
        order=tuple(order),
        positions=positions,
        costs=costs,
        dependency_mask=dependency_mask,
        exclusive_mask=exclusive_mask,
        neutral_mask=neutral_mask,
        search_order=search_order,
    )


def _structural_reduction(case: Case, order: tuple[str, ...],
                          positions: dict[str, int]) -> tuple[int, tuple[str, ...]]:
    """识别并剔除可证不进入帕累托前沿的中性提案。

    中性提案 = 不被任何选票表达（既不在 ranking 也不在 unacceptable），
    且不在任何被表达提案的传递依赖闭包中。对任意含中性提案 N 的可行组合
    S，S\\{N}（连带剔除仅被中性提案依赖的部分）仍然可行、成本严格更低、
    表达损失不变、拒绝违规不增，因此 S 被支配，前沿不含任何中性提案。
    这是基于偏好结构的可证缩减，不是软打分。
    """
    expressed: set[str] = set()
    for ballot in case.ballots:
        for item in (*ballot.ranked, *ballot.unacceptable):
            if item.valid:
                expressed.add(item.proposal_id)
    proposal_map = case.proposal_map
    # 被表达提案的传递依赖闭包：这些提案即使未被表达也必须保留。
    required: set[str] = set()
    stack = [pid for pid in expressed if pid in proposal_map]
    while stack:
        current = stack.pop()
        for dep in proposal_map[current].depends_on:
            if dep not in required:
                required.add(dep)
                stack.append(dep)
    relevant = expressed | required
    neutral = [pid for pid in order if pid not in relevant]
    neutral_mask = _mask_of(tuple(neutral), positions)
    search_order = tuple(pid for pid in order if pid in relevant)
    return neutral_mask, search_order


def _mask_of(ids: tuple[str, ...], positions: dict[str, int]) -> int:
    mask = 0
    for pid in ids:
        mask |= 1 << positions[pid]
    return mask


def mask_to_ids(mask: int, order: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(order[i] for i in range(len(order)) if mask & (1 << i))


def is_feasible(mask: int, index: FeasibilityIndex) -> bool:
    """预算、依赖、互斥三类硬约束。"""
    total = 0
    for i in range(index.size):
        if mask & (1 << i):
            total += index.costs[i]
            if total > index.case.budget:
                return False
            if index.dependency_mask[i] & ~mask:
                return False
            if index.exclusive_mask[i] & mask:
                return False
    return True


def rejection_violations(mask: int, case: Case,
                         index: FeasibilityIndex) -> dict[str, list[str]]:
    """返回每个代表被组合触及的、仍有效的 unacceptable 提案。"""
    selected = set(mask_to_ids(mask, index.order))
    result: dict[str, list[str]] = {}
    for ballot in case.ballots:
        hit = [
            item.proposal_id
            for item in ballot.valid_unacceptable
            if item.proposal_id in selected
        ]
        if hit:
            result[ballot.representative] = sorted(hit)
    return result


def enumerate_feasible(index: FeasibilityIndex) -> list[int]:
    """完整枚举（仅小案例使用），返回全部满足硬约束的位掩码。"""
    feasible: list[int] = []
    for mask in range(1 << index.size):
        if is_feasible(mask, index):
            feasible.append(mask)
    return feasible


def greedy_seed(index: FeasibilityIndex,
                preference_hits: dict[str, int]) -> int:
    """贪心构造一个可行初始解，让分支定界尽早获得可剪枝的现行前沿。

    选择顺序按"被有效最高偏好层提及次数 / 成本"，仅用于搜索顺序，
    不构成任何推荐口径；受预算、依赖、互斥约束，任何提案至多尝试一次。
    """
    order = index.order
    mentions = {pid: 0 for pid in order}
    for pid, mask in preference_hits.items():
        mentions[pid] = bin(mask).count("1")
    ranked = sorted(
        order,
        key=lambda pid: (-(mentions[pid] / index.costs[index.positions[pid]]),
                         index.positions[pid]),
    )
    selected = 0
    total_cost = 0
    for pid in ranked:
        i = index.positions[pid]
        bit = 1 << i
        if selected & bit:
            continue
        if total_cost + index.costs[i] > index.case.budget:
            continue
        if index.dependency_mask[i] & ~selected:
            continue
        if index.exclusive_mask[i] & selected:
            continue
        selected |= bit
        total_cost += index.costs[i]
    return selected


def branch_and_bound(index: FeasibilityIndex,
                     objective,
                     preference_hits: dict[str, int] | None = None,
                     front_cap: int = 5000) -> tuple[list[int], dict[str, int]]:
    """分支定界求非支配组合（正确性优先）。

    关于损失目标的关键事实：部分路径的表达损失在加入提案后**只会下降**，
    因此部分路径损失是其超集的**上界**，而超集的非平凡损失下界不存在
    （最乐观情况下剩余偏好层全部命中，损失为 0）。所以不能用"现行前沿
    损失更优"去裁剪未完成路径——超集完全可能靠继续选入提案把损失反超。

    ``objective(mask, cost)`` 返回轻量目标元组
    ``(cost, total_loss, worst_group_loss)``，应尽量用按位预算实现。

    本实现只采用可证安全的缩减：

    0. 结构缩减：中性提案（无人表达、且非被表达提案的传递依赖）已在
       ``build_index`` 中识别，DFS 不展开这些位置——含它们的组合必被
       剔除后的同损低价组合支配；
    1. 硬约束剪枝：超预算、缺依赖、触互斥的选入分支不展开；跳过某提案时
       把依赖它的传递后代封禁（这些都是可行性硬边界）；
    2. 零损失支配剪枝：仅当现行前沿中已存在两个损失维度都达到理论最小值
       （0）且成本不高于当前路径的完整组合时，该路径的任何超集都被它
       三维支配，可安全剪去；
    3. 帕累托支配判定只在**完整叶子**上进行（``consider``）。

    位掩码始终基于完整拓扑序；任何时刻内存里只有当前 DFS 路径与现行
    非支配前沿，绝不保存全部子集。
    """
    stats = {"nodes_visited": 0, "pruned_bounds": 0, "pruned_dominance": 0,
             "front_capped": 0, "hard_constraint_skips": 0}
    dependents = transitive_dependents(index)
    # 仅在相关提案位置上分支；中性提案位置从不进入搜索。
    search_positions = [index.positions[pid] for pid in index.search_order]
    seed_mask = greedy_seed(index, preference_hits or {})
    # 前沿同时保存掩码与轻量目标元组 (cost, total_loss, worst_loss)，
    # 使支配过滤只需数值比较，不再对同一掩码反复做完整评估。
    front: list[int] = [seed_mask]
    front_obj: list[tuple[int, float, float]] = [
        objective(seed_mask, _mask_cost(seed_mask, index))
    ]

    def consider(mask: int, cost: int) -> None:
        nonlocal front, front_obj
        cost_m, total_m, worst_m = objective(mask, cost)
        for cost_o, total_o, worst_o in front_obj:
            # 现行成员支配新叶子：三维不劣且至少一维严格更优。
            if (cost_o <= cost_m and total_o <= total_m + 1e-12
                    and worst_o <= worst_m + 1e-12
                    and (cost_o < cost_m or total_o < total_m - 1e-12
                         or worst_o < worst_m - 1e-12)):
                stats["pruned_dominance"] += 1
                return
        # 新叶子支配的旧成员被剔除。
        kept: list[int] = []
        kept_obj: list[tuple[int, float, float]] = []
        for incumbent, obj in zip(front, front_obj):
            cost_o, total_o, worst_o = obj
            if (cost_m <= cost_o and total_m <= total_o + 1e-12
                    and worst_m <= worst_o + 1e-12
                    and (cost_m < cost_o or total_m < total_o - 1e-12
                         or worst_m < worst_o - 1e-12)):
                continue
            kept.append(incumbent)
            kept_obj.append(obj)
        kept.append(mask)
        kept_obj.append((cost_m, total_m, worst_m))
        if len(kept) > front_cap:
            # 极端情况下保留确定性的子集，并显式记录截断。
            order_idx = sorted(range(len(kept)), key=lambda k: kept[k])[:front_cap]
            front = [kept[k] for k in order_idx]
            front_obj = [kept_obj[k] for k in order_idx]
            stats["front_capped"] += 1
        else:
            front, front_obj = kept, kept_obj

    def _any_selectable(step: int, chosen: int, cost: int,
                        blocked: int) -> bool:
        """剩余相关位置中是否还存在可选入的提案。"""
        for j in search_positions[step:]:
            bit_j = 1 << j
            if blocked & bit_j:
                continue
            if cost + index.costs[j] > index.case.budget:
                continue
            if index.dependency_mask[j] & ~chosen:
                continue
            if index.exclusive_mask[j] & chosen:
                continue
            return True
        return False

    def dfs(step: int, chosen: int, cost: int, blocked: int) -> None:
        if step == len(search_positions):
            stats["nodes_visited"] += 1
            consider(chosen, cost)
            return

        # 安全剪枝：仅针对"零损失" incumbent。超集损失下界为 0，若某现行
        # 完整组合 I 的两个损失都为 0、且成本不高于当前路径，则任意超集 S
        # 有 loss(S) >= 0 = loss(I)、cost(S) >= cost(C) >= cost(I)，
        # 即 I 支配 S。这是唯一不依赖松散下界的安全损失剪枝。
        for cost_o, total_o, worst_o in front_obj:
            if (total_o <= 1e-12 and worst_o <= 1e-12 and cost_o <= cost):
                stats["pruned_bounds"] += 1
                return

        # 提前终止：剩余相关位置已无可选提案，当前掩码是唯一可行完成态，
        # 直接作为叶子处理，避免沿一长串强制"不选"分支空转。
        if not _any_selectable(step, chosen, cost, blocked):
            stats["nodes_visited"] += 1
            consider(chosen, cost)
            return

        position = search_positions[step]
        bit = 1 << position
        # 分支 A：不选当前提案，依赖它的传递后代一并封禁。
        dfs(step + 1, chosen, cost, blocked | dependents[position])

        # 分支 B：硬约束全部满足时才选入，否则该选入分支安全跳过
        # （分支 A 已在上方执行，此处 return 不影响"不选"路径）。
        if blocked & bit:
            stats["hard_constraint_skips"] += 1
            return
        new_cost = cost + index.costs[position]
        if new_cost > index.case.budget:
            stats["hard_constraint_skips"] += 1
            return
        if index.dependency_mask[position] & ~chosen:
            stats["hard_constraint_skips"] += 1
            return
        if index.exclusive_mask[position] & chosen:
            stats["hard_constraint_skips"] += 1
            return
        dfs(step + 1, chosen | bit, new_cost, blocked)

    dfs(0, 0, 0, 0)
    return sorted(set(front)), stats


def _mask_cost(mask: int, index: FeasibilityIndex) -> int:
    total = 0
    for i in range(index.size):
        if mask & (1 << i):
            total += index.costs[i]
    return total


def transitive_dependents(index: FeasibilityIndex) -> tuple[int, ...]:
    """每个提案被跳过时需要连带封禁的传递依赖集（位掩码）。"""
    result = [0] * index.size
    for position, pid in enumerate(index.order):
        stack = [position]
        seen = 0
        while stack:
            current = stack.pop()
            for other in range(index.size):
                if other == current:
                    continue
                if index.dependency_mask[other] & (1 << current) and not (seen & (1 << other)):
                    seen |= 1 << other
                    stack.append(other)
        result[position] = seen
    return tuple(result)
