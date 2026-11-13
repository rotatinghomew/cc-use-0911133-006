"""偏好组合分析的数据模型与资料核算。

口径（与 ``docs/data.md`` 一致并在报告中再次声明）：

* ranking 的每个内层数组是一组**并列**项目，数组之间偏好从高到低；
  并列项之间不虚构先后。
* 未出现在 ranking 中**不代表反对**；只有 ``unacceptable`` 是拒绝。
* 同一 representative 的多张选票只有最高 revision 生效，旧票被取代，
  不会因重复提交而增加群体席位。
* 选票通过 ``proposal_versions`` 声明它针对的提案版本。提案改版后，
  只有"选票版本 == 当前版本"或"选票版本在该提案
  ``compatible_ballot_versions`` 白名单内"时，该选票对该提案的表达
  （包括 ranking 与 unacceptable）才仍然有效，否则该条表达作废并被
  逐条记录，绝不默认沿用旧意愿。
* 缺席代表不虚构偏好；其影响只通过席位加权情景与脆弱性标记呈现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PreferenceDataError(ValueError):
    """资料结构不满足可分析的基本条件。"""


@dataclass(frozen=True)
class Proposal:
    id: str
    version: int
    cost: int
    depends_on: tuple[str, ...]
    exclusive_with: tuple[str, ...]
    compatible_ballot_versions: tuple[int, ...]

    def accepts_ballot_version(self, ballot_version: int) -> bool:
        if ballot_version == self.version:
            return True
        return ballot_version in self.compatible_ballot_versions


@dataclass(frozen=True)
class RankedItem:
    proposal_id: str
    tier: int  # 0 为最高偏好层
    ballot_version: int
    valid: bool
    reason: str  # valid 时为 "current" 或 "compatible"，否则说明作废原因


@dataclass(frozen=True)
class Ballot:
    representative: str
    group: str
    revision: int
    ranked: tuple[RankedItem, ...]
    unacceptable: tuple[RankedItem, ...]
    superseded_revisions: tuple[int, ...]
    ranking_width: int  # 选票实际列出的层数（用于损失归一化）

    @property
    def valid_ranked(self) -> tuple[RankedItem, ...]:
        return tuple(item for item in self.ranked if item.valid)

    @property
    def valid_unacceptable(self) -> tuple[RankedItem, ...]:
        return tuple(item for item in self.unacceptable if item.valid)

    def has_any_valid_expression(self) -> bool:
        return bool(self.valid_ranked or self.valid_unacceptable)


@dataclass
class Case:
    name: str
    budget: int
    groups: tuple[dict[str, Any], ...]
    proposals: tuple[Proposal, ...]
    ballots: tuple[Ballot, ...]
    absent_representatives: tuple[dict[str, str], ...]
    warnings: list[str] = field(default_factory=list)

    @property
    def proposal_map(self) -> dict[str, Proposal]:
        return {proposal.id: proposal for proposal in self.proposals}

    def group_ids(self) -> tuple[str, ...]:
        return tuple(group["id"] for group in self.groups)

    def nominal_seats(self, group_id: str) -> int:
        for group in self.groups:
            if group["id"] == group_id:
                return int(group["seats"])
        raise PreferenceDataError(f"未知群体: {group_id}")

    def present_reps(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {group["id"]: [] for group in self.groups}
        for ballot in self.ballots:
            if ballot.representative not in result[ballot.group]:
                result[ballot.group].append(ballot.representative)
        return result


def _build_proposals(raw_proposals: list[dict[str, Any]],
                     changes: list[dict[str, Any]] | None
                     ) -> tuple[tuple[Proposal, ...], list[str]]:
    by_id: dict[str, Proposal] = {}
    warnings: list[str] = []
    for row in raw_proposals:
        pid = str(row["id"])
        by_id[pid] = Proposal(
            id=pid,
            version=int(row.get("version", 1)),
            cost=int(row["cost"]),
            depends_on=tuple(row.get("depends_on", ())),
            exclusive_with=tuple(row.get("exclusive_with", ())),
            compatible_ballot_versions=tuple(
                int(version) for version in row.get("compatible_ballot_versions", ())
            ),
        )
    for change in changes or ():
        pid = str(change["id"])
        if pid not in by_id:
            raise PreferenceDataError(f"proposal_changes 指向未知提案 {pid}")
        old = by_id[pid]
        by_id[pid] = Proposal(
            id=pid,
            version=int(change.get("version", old.version + 1)),
            cost=int(change.get("cost", old.cost)),
            depends_on=old.depends_on,
            exclusive_with=old.exclusive_with,
            compatible_ballot_versions=tuple(
                int(version) for version in change.get("compatible_ballot_versions", ())
            ),
        )
    # 互斥是对称关系：单边声明做对称闭包，硬约束对双方同时生效；
    # 指向未知提案的悬空引用不臆造对象，降级为资料警告。
    for pid, proposal in list(by_id.items()):
        valid_exclusive = []
        for other in proposal.exclusive_with:
            if other not in by_id:
                warnings.append(
                    f"提案 {pid} 的互斥对象 {other} 在资料中不存在，"
                    "该悬空引用已忽略，其余约束照常生效"
                )
                continue
            valid_exclusive.append(other)
            if pid not in by_id[other].exclusive_with:
                mirrored = tuple(sorted(set(by_id[other].exclusive_with) | {pid}))
                by_id[other] = Proposal(
                    id=other,
                    version=by_id[other].version,
                    cost=by_id[other].cost,
                    depends_on=by_id[other].depends_on,
                    exclusive_with=mirrored,
                    compatible_ballot_versions=by_id[other].compatible_ballot_versions,
                )
        if len(valid_exclusive) != len(proposal.exclusive_with):
            by_id[pid] = Proposal(
                id=pid,
                version=proposal.version,
                cost=proposal.cost,
                depends_on=proposal.depends_on,
                exclusive_with=tuple(valid_exclusive),
                compatible_ballot_versions=proposal.compatible_ballot_versions,
            )
    return tuple(by_id[pid] for pid in sorted(by_id)), warnings


def _check_relations(proposals: tuple[Proposal, ...]) -> None:
    known = {proposal.id for proposal in proposals}
    for proposal in proposals:
        for dependency in proposal.depends_on:
            if dependency not in known:
                raise PreferenceDataError(
                    f"提案 {proposal.id} 依赖未知提案 {dependency}"
                )
        for other in proposal.exclusive_with:
            if other not in known:
                raise PreferenceDataError(
                    f"提案 {proposal.id} 与未知提案 {other} 互斥"
                )


def _resolve_item(pid: str, ballot_version: int,
                  proposal_map: dict[str, Proposal], tier: int | None) -> RankedItem:
    if pid not in proposal_map:
        return RankedItem(pid, -1 if tier is None else tier, ballot_version,
                          False, "未知提案，资料中不存在")
    proposal = proposal_map[pid]
    if proposal.accepts_ballot_version(ballot_version):
        reason = "current" if ballot_version == proposal.version else "compatible"
        return RankedItem(pid, -1 if tier is None else tier, ballot_version, True, reason)
    return RankedItem(
        pid,
        -1 if tier is None else tier,
        ballot_version,
        False,
        f"选票基于 v{ballot_version}，当前为 v{proposal.version}，"
        f"兼容白名单 {list(proposal.compatible_ballot_versions)} 不含该版本",
    )


def _build_ballots(raw_ballots: list[dict[str, Any]],
                   proposal_map: dict[str, Proposal]) -> tuple[Ballot, ...]:
    # 第一步：按代表归集，revision 最高者取代旧票。
    latest: dict[str, dict[str, Any]] = {}
    superseded: dict[str, list[int]] = {}
    for row in raw_ballots:
        rep = str(row["representative"])
        revision = int(row["revision"])
        if rep not in latest:
            latest[rep] = row
            superseded[rep] = []
            continue
        if revision == int(latest[rep]["revision"]):
            raise PreferenceDataError(f"代表 {rep} 存在相同 revision={revision} 的重复选票")
        if revision > int(latest[rep]["revision"]):
            superseded[rep].append(int(latest[rep]["revision"]))
            latest[rep] = row
        else:
            superseded[rep].append(revision)

    ballots: list[Ballot] = []
    for rep in sorted(latest):
        row = latest[rep]
        versions = row.get("proposal_versions", {})
        ranked_items: list[RankedItem] = []
        for tier, tier_group in enumerate(row.get("ranking", ())):
            for pid in tier_group:
                if pid not in versions:
                    raise PreferenceDataError(
                        f"选票 {rep} 的 ranking 含 {pid}，但 proposal_versions 未声明其版本"
                    )
                ranked_items.append(
                    _resolve_item(str(pid), int(versions[str(pid)]), proposal_map, tier)
                )
        unacceptable_items: list[RankedItem] = []
        for pid in row.get("unacceptable", ()):
            ballot_version = int(versions.get(str(pid), 1))
            if str(pid) not in versions:
                # 拒绝声明没有版本依据时不臆测，按缺失版本依据作废并记录。
                unacceptable_items.append(
                    RankedItem(str(pid), -1, ballot_version, False,
                               "unacceptable 未在 proposal_versions 中声明版本依据")
                )
            else:
                unacceptable_items.append(
                    _resolve_item(str(pid), ballot_version, proposal_map, None)
                )
        width = len(row.get("ranking", ()))
        ballots.append(
            Ballot(
                representative=rep,
                group=str(row["group"]),
                revision=int(row["revision"]),
                ranked=tuple(ranked_items),
                unacceptable=tuple(unacceptable_items),
                superseded_revisions=tuple(sorted(superseded[rep])),
                ranking_width=width,
            )
        )
    return tuple(ballots)


def _build_case(name: str, budget: int, groups: list[dict[str, Any]],
                raw_proposals: list[dict[str, Any]],
                raw_ballots: list[dict[str, Any]],
                absent: list[dict[str, Any]],
                changes: list[dict[str, Any]] | None) -> Case:
    proposals, proposal_warnings = _build_proposals(raw_proposals, changes)
    _check_relations(proposals)
    proposal_map = {proposal.id: proposal for proposal in proposals}
    ballots = _build_ballots(raw_ballots, proposal_map)

    known_groups = {str(group["id"]) for group in groups}
    warnings: list[str] = list(proposal_warnings)
    for ballot in ballots:
        if ballot.group not in known_groups:
            raise PreferenceDataError(
                f"选票 {ballot.representative} 属于未知群体 {ballot.group}"
            )
        if not ballot.has_any_valid_expression():
            warnings.append(
                f"代表 {ballot.representative}（{ballot.group}）的全部表达均已失效，"
                "其意愿在当前提案版本下无任何可计依据"
            )
    present = {ballot.representative for ballot in ballots}
    for row in absent:
        rep = str(row["representative"])
        group = str(row["group"])
        if group not in known_groups:
            raise PreferenceDataError(f"缺席代表 {rep} 属于未知群体 {group}")
        if rep in present:
            warnings.append(f"缺席名单中的 {rep} 实际已投票，缺席记录不生效")
    # 席位核算：同一群体的出席代表数不得超过名义席位数。
    case = Case(
        name=name,
        budget=int(budget),
        groups=tuple(dict(group) for group in groups),
        proposals=proposals,
        ballots=ballots,
        absent_representatives=tuple(dict(row) for row in absent),
        warnings=warnings,
    )
    for group_id, reps in case.present_reps().items():
        seats = case.nominal_seats(group_id)
        if len(reps) > seats:
            raise PreferenceDataError(
                f"群体 {group_id} 出席代表 {len(reps)} 人超过席位 {seats}，"
                "重复提交不应增加席位，请核对 revision 核算"
            )
    return case


def load_case(path: str | Path) -> dict[str, Any]:
    """加载资料文件，返回小案例与大案例两个 :class:`Case`。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("schema_version", 1)) != 1:
        raise PreferenceDataError(
            f"仅支持 schema_version=1，收到 {data.get('schema_version')}"
        )
    small = _build_case(
        name="small",
        budget=data["budget"],
        groups=data["groups"],
        raw_proposals=data["proposals"],
        raw_ballots=data["ballots"],
        absent=data.get("absent_representatives", ()),
        changes=data.get("proposal_changes", ()),
    )
    larger = data.get("larger_case")
    large = None
    if larger:
        large = _build_case(
            name="large",
            budget=larger["budget"],
            groups=data["groups"],
            raw_proposals=larger["proposals"],
            raw_ballots=larger["ballots"],
            absent=larger.get("absent_representatives", ()),
            changes=larger.get("proposal_changes", ()),
        )
    return {"small": small, "large": large, "seed": int(data.get("seed", 0))}
