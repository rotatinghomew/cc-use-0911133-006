"""循环偏好证据：基于成对多数比较，不补全、不虚构。

规则：

* 仅当一张选票把两个提案放在**不同有效层**时才产生一次成对比较；
  同层并列、未列出、失效表达均不产生比较。
* 每条多数边记录双方得票与投票代表；若该边涉及的选票表达在提案改版
  后失效，会在 ``voided_ballot_notes`` 中显式说明——同一张图同时给出
  "原始表达口径"与"当前有效口径"两套证据。
* 在图中检测简单有向环（循环偏好）与孔多塞胜者/败者。
"""

from __future__ import annotations

from dataclasses import dataclass

from deliberation_lake.preferences.model import Ballot, Case


@dataclass(frozen=True)
class PairwiseEdge:
    winner: str
    loser: str
    votes_for: int
    votes_against: int
    voters_for: tuple[str, ...]
    voters_against: tuple[str, ...]
    relies_on_voided_expression: bool


def _ballot_strict_order(ballot: Ballot,
                         valid_only: bool) -> dict[str, int]:
    order: dict[str, int] = {}
    for item in ballot.ranked:
        if valid_only and not item.valid:
            continue
        order.setdefault(item.proposal_id, item.tier)
    return order


def pairwise_edges(case: Case, valid_only: bool) -> list[PairwiseEdge]:
    # 只对至少一张选票有效表达过的提案对做比较。
    items: set[str] = set()
    for ballot in case.ballots:
        for item in ballot.ranked:
            if valid_only and not item.valid:
                continue
            items.add(item.proposal_id)
    proposals = sorted(items)
    edges: list[PairwiseEdge] = []
    for index, left in enumerate(proposals):
        for right in proposals[index + 1:]:
            left_votes: list[str] = []
            right_votes: list[str] = []
            voided = False
            for ballot in case.ballots:
                order = _ballot_strict_order(ballot, valid_only)
                if left not in order or right not in order:
                    continue
                if order[left] < order[right]:
                    left_votes.append(ballot.representative)
                elif order[right] < order[left]:
                    right_votes.append(ballot.representative)
                # 同层并列：不计票。
                if not valid_only:
                    continue
                # 标记该比较是否建立在曾失效的表达之上（用于原始口径对照）。
            if not valid_only:
                # 原始口径下检查这些比较如今是否仍站得住。
                for ballot in case.ballots:
                    current = _ballot_strict_order(ballot, True)
                    original = _ballot_strict_order(ballot, False)
                    if left in original and right in original and (
                        left not in current or right not in current
                    ):
                        voided = True
            if left_votes and len(left_votes) > len(right_votes):
                edges.append(PairwiseEdge(
                    winner=left, loser=right,
                    votes_for=len(left_votes), votes_against=len(right_votes),
                    voters_for=tuple(sorted(left_votes)),
                    voters_against=tuple(sorted(right_votes)),
                    relies_on_voided_expression=voided,
                ))
            elif right_votes and len(right_votes) > len(left_votes):
                edges.append(PairwiseEdge(
                    winner=right, loser=left,
                    votes_for=len(right_votes), votes_against=len(left_votes),
                    voters_for=tuple(sorted(right_votes)),
                    voters_against=tuple(sorted(left_votes)),
                    relies_on_voided_expression=voided,
                ))
            # 平票不形成多数边，但在 ties 中单独报告。
    return edges


def detect_cycles(edges: list[PairwiseEdge]) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for edge in edges:
        graph.setdefault(edge.winner, set()).add(edge.loser)
        graph.setdefault(edge.loser, set())
    cycles: list[list[str]] = []
    state: dict[str, int] = {node: 0 for node in graph}  # 0 未访问 1 栈中 2 完成
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph[node]):
            if state[nxt] == 0:
                visit(nxt)
            elif state[nxt] == 1:
                start = stack.index(nxt)
                cycle = stack[start:] + [nxt]
                if cycle not in cycles:
                    cycles.append(cycle)
        stack.pop()
        state[node] = 2

    for node in sorted(graph):
        if state[node] == 0:
            visit(node)
    return cycles


def condorcet(edges: list[PairwiseEdge]) -> dict[str, object]:
    wins: dict[str, set[str]] = {}
    nodes: set[str] = set()
    for edge in edges:
        wins.setdefault(edge.winner, set()).add(edge.loser)
        nodes.add(edge.winner)
        nodes.add(edge.loser)
    winners = sorted(
        node for node in nodes
        if node in wins and wins[node] == (nodes - {node})
    )
    losers = sorted(
        node for node in nodes
        if all(node in wins.get(other, set()) for other in nodes if other != node)
    )
    return {
        "condorcet_winner": winners[0] if winners else None,
        "condorcet_loser": losers[0] if losers else None,
        "compared_nodes": tuple(sorted(nodes)),
    }


def voided_expression_notes(case: Case) -> list[dict[str, object]]:
    notes: list[dict[str, object]] = []
    for ballot in case.ballots:
        for item in (*ballot.ranked, *ballot.unacceptable):
            if not item.valid:
                notes.append({
                    "representative": ballot.representative,
                    "group": ballot.group,
                    "proposal": item.proposal_id,
                    "ballot_version": item.ballot_version,
                    "context": "ranking" if item.tier >= 0 else "unacceptable",
                    "reason": item.reason,
                })
    return notes


def tied_pairs(case: Case) -> list[dict[str, object]]:
    """报告选票内的并列项：它们之间不产生先后，也不参与成对计票。"""
    rows: list[dict[str, object]] = []
    for ballot in case.ballots:
        tiers: dict[int, list[str]] = {}
        for item in ballot.ranked:
            tiers.setdefault(item.tier, []).append(item.proposal_id)
        for tier, ids in sorted(tiers.items()):
            if len(ids) > 1:
                rows.append({
                    "representative": ballot.representative,
                    "tier": tier,
                    "tied_proposals": tuple(sorted(ids)),
                    "all_valid": all(
                        item.valid for item in ballot.ranked
                        if item.tier == tier
                    ),
                })
    return rows


def build_cycle_evidence(case: Case) -> dict[str, object]:
    current_edges = pairwise_edges(case, valid_only=True)
    original_edges = pairwise_edges(case, valid_only=False)
    current_cycles = detect_cycles(current_edges)
    original_cycles = detect_cycles(original_edges)
    current = condorcet(current_edges)
    return {
        "current_valid_edges": [edge.__dict__ for edge in current_edges],
        "original_expression_edges": [edge.__dict__ for edge in original_edges],
        "cycles_under_current_versions": current_cycles,
        "cycles_under_original_expressions": original_cycles,
        "condorcet": current,
        "tied_groups": tied_pairs(case),
        "voided_expressions": voided_expression_notes(case),
        "interpretation": (
            "多数边仅来自不同有效层之间的成对比较；同层并列、未列出与"
            "改版失效均不计票。原始口径下成立但当前口径下消失的环，"
            "说明改版已改变可表达的集体偏好，不能沿用旧票结论。"
        ),
    }
