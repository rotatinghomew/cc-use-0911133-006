"""把分析结果落盘为 JSON 与一份理事会可读的 Markdown 协商材料。"""

from __future__ import annotations

import json
from pathlib import Path


def write_outputs(result: dict[str, object], output_dir: str) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    summary_path = directory / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["summary"] = str(summary_path)

    for case_key in ("small_case", "large_case"):
        case_result = result.get(case_key)
        if not case_result:
            continue
        prefix = "small" if case_key == "small_case" else "large"
        front_path = directory / f"{prefix}_front_records.json"
        front_path.write_text(
            json.dumps(
                {
                    "case": case_result["case"],
                    "pareto_front": case_result["pareto_front"],
                    "acceptable_front": case_result["acceptable_front"],
                },
                ensure_ascii=False, indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
        paths[f"{prefix}_front"] = str(front_path)

    report_path = directory / "council_briefing.md"
    report_path.write_text(render_markdown(result), encoding="utf-8")
    paths["briefing"] = str(report_path)
    return paths


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(result: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# 园区协商组合分析（协商材料，非决定）\n")
    lines.append("> " + str(result["disclaimer"]) + "\n")
    lines.append("本材料不提供单一排名或综合分，只呈现成本、整体偏好损失、"
                 "最不满意群体损失三个独立目标之间的取舍，最终选择权在理事会。\n")

    for case_key, title in (("small_case", "小案例（可穷举核对）"),
                            ("large_case", "大案例（分支定界，不全量装入内存）")):
        case = result.get(case_key)
        if not case:
            continue
        lines.append(f"\n## {title}\n")
        lines.append(f"- 预算：{case['budget']}；提案数：{case['proposal_count']}；"
                     f"理论子集空间：2^{case['proposal_count']} = {case['search_space_size']}")
        search = case["search"]
        lines.append(
            f"- 搜索：访问节点 {search['nodes_visited']}，"
            f"边界剪枝 {search['pruned_bounds']}，支配剪枝 {search['pruned_dominance']}，"
            f"耗时 {search['elapsed_seconds']} 秒；{search['memory_guarantee']}"
        )
        if case.get("exhaustive_cross_check"):
            check = case["exhaustive_cross_check"]
            lines.append(
                f"- 穷举核对：可行组合 {check['feasible_portfolio_count']} 个，"
                f"穷举前沿 {check['exhaustive_front_count']} 个，"
                f"分支定界前沿 {check['branch_and_bound_front_count']} 个，"
                f"一致：**{check['fronts_match']}**"
            )

        lines.append("\n### 选票核算（修订取代 / 改版失效 / 缺席）\n")
        for group in case["ballot_accounting"]["groups"]:
            lines.append(
                f"- 群体 {group['group_id']}：名义席位 {group['nominal_seats']}，"
                f"出席 {group['present_representatives']}，"
                f"缺席 {group['absent_representatives']}，席位校验 {group['seat_inflation_check']}"
            )
        for ballot in case["ballot_accounting"]["ballots"]:
            if ballot["superseded_revisions"]:
                lines.append(
                    f"- {ballot['representative']}：revision {ballot['applied_revision']} "
                    f"取代旧 revision {ballot['superseded_revisions']}，不增加席位"
                )
            for item in ballot["voided_ranked"]:
                lines.append(
                    f"- ⚠️ {ballot['representative']} 对 {item['proposal']} 的排序表达作废："
                    f"{item['reason']}"
                )
            for item in ballot["voided_unacceptable"]:
                lines.append(
                    f"- ⚠️ {ballot['representative']} 对 {item['proposal']} 的拒绝表达作废："
                    f"{item['reason']}"
                )

        lines.append("\n### 非支配组合（帕累托前沿，按成本排序展示）\n")
        lines.append("| 组合 | 提案 | 成本 | 整体损失 | 最不满意群体损失 | 覆盖群体 | 拒绝违规 |")
        lines.append("|---|---|---|---|---|---|---|")
        records = sorted(case["pareto_front"]["records"],
                         key=lambda row: (row["cost"], row["total_loss"],
                                          row["worst_group_loss"]))
        for record in records:
            lines.append(
                "| {pid} | {proposals} | {cost} | {total} | {worst} | {covered} | {violations} |".format(
                    pid=record["portfolio_id"],
                    proposals=", ".join(record["proposals"]) or "（空组合）",
                    cost=record["cost"],
                    total=_fmt(record["total_loss"]),
                    worst=_fmt(record["worst_group_loss"]),
                    covered=", ".join(record["covered_groups"]) or "无",
                    violations=(
                        "无" if record["acceptable"]
                        else f"{record['violation_count']} 项（{', '.join(record['violating_representatives'])}）"
                    ),
                )
            )
        lines.append(f"\n_{case['pareto_front']['display_cap_note']}_\n")

        acceptable = case["acceptable_front"]["records"]
        lines.append(f"### 可接受前沿（无任何拒绝违规，{case['acceptable_front']['size']} 个）\n")
        if not acceptable:
            lines.append("**当前硬约束下不存在让所有代表都不触发 unacceptable 的前沿组合。**\n")
        else:
            lines.append("| 组合 | 提案 | 成本 | 整体损失 | 最不满意群体损失 | 覆盖群体 |")
            lines.append("|---|---|---|---|---|---|")
            for record in sorted(acceptable, key=lambda row: row["cost"]):
                lines.append(
                    "| {pid} | {proposals} | {cost} | {total} | {worst} | {covered} |".format(
                        pid=record["portfolio_id"],
                        proposals=", ".join(record["proposals"]) or "（空组合）",
                        cost=record["cost"],
                        total=_fmt(record["total_loss"]),
                        worst=_fmt(record["worst_group_loss"]),
                        covered=", ".join(record["covered_groups"]) or "无",
                    )
                )

        lines.append("\n### 群体可接受方案存在性（硬约束下）\n")
        for group_id, info in case["acceptable_option_existence"].items():
            if info["has_acceptable_feasible_portfolio"]:
                lines.append(
                    f"- {group_id}：存在可接受组合，例如 {info['example_portfolio']}"
                )
            else:
                lines.append(
                    f"- 🚫 **{group_id}：在全部可行组合中都没有可接受方案"
                    "（无人出席，或任何满足预算/依赖/互斥的组合都无法同时避免拒绝并命中其偏好）**"
                )

        lines.append("\n### 循环偏好证据\n")
        cycle = case["cycle_evidence"]
        current_cycles = cycle["cycles_under_current_versions"]
        original_cycles = cycle["cycles_under_original_expressions"]
        lines.append(f"- 当前有效版本口径下的循环：{current_cycles or '无'}")
        lines.append(f"- 原始表达口径下的循环：{original_cycles or '无'}")
        condorcet = cycle["condorcet"]
        lines.append(
            f"- 孔多塞胜者：{condorcet['condorcet_winner'] or '无'}；"
            f"孔多塞败者：{condorcet['condorcet_loser'] or '无'}"
        )
        for edge in cycle["current_valid_edges"]:
            flag = "（含改版失效依据，需谨慎）" if edge["relies_on_voided_expression"] else ""
            lines.append(
                f"- 多数边 {edge['winner']} ≻ {edge['loser']}："
                f"{edge['votes_for']} 票 {edge['voters_for']} 对 "
                f"{edge['votes_against']} 票 {edge['voters_against']}{flag}"
            )
        for row in cycle["tied_groups"]:
            lines.append(
                f"- 并列（不计票）：{row['representative']} 第 {row['tier']} 层 "
                f"{row['tied_proposals']}，全部有效：{row['all_valid']}"
            )
        lines.append(f"\n_{cycle['interpretation']}_\n")

        lines.append("### 敏感性：缺席、席位扰动与口径变化\n")
        lines.append("| 情景 | 权重 | 前沿数 | 可接受前沿数 | 失去可接受方案的群体 |")
        lines.append("|---|---|---|---|---|")
        for scenario in case["scenarios"]:
            lines.append(
                "| {key} | {weights} | {front} | {acceptable} | {lost} |".format(
                    key=scenario["key"],
                    weights=scenario["weights"],
                    front=scenario["front_size"],
                    acceptable=scenario["acceptable_front_size"],
                    lost=", ".join(scenario["groups_without_acceptable_option"]) or "无",
                )
            )
        for scenario in case["scenarios"]:
            lines.append(f"\n**{scenario['key']}** — {scenario['description']}")
            for axis, anchors in scenario["anchors"].items():
                text = "; ".join(
                    ",".join(item["portfolio"]) or "空组合" for item in anchors
                )
                lines.append(f"- {axis} 锚点：{text}")

        lines.append("\n### 稳定性：基线组合在不同口径下的位置\n")
        stable_rows = case["stability"]["rows"]
        lines.append("| 情景 | 组合 | 仍在前沿 | 仍可接受 | 覆盖群体 |")
        lines.append("|---|---|---|---|---|")
        for row in stable_rows:
            lines.append(
                f"| {row['scenario']} | {', '.join(row['portfolio']) or '空组合'} | "
                f"{row['still_on_front']} | {row['still_acceptable']} | "
                f"{', '.join(row['covered_groups']) or '无'} |"
            )

        if case.get("warnings"):
            lines.append("\n### 资料警告\n")
            for warning in case["warnings"]:
                lines.append(f"- {warning}")

    lines.append("\n## 方法学口径\n")
    methodology = result["small_case"]["methodology"]
    for section, content in methodology.items():
        if isinstance(content, list):
            lines.append(f"### {section}")
            for item in content:
                lines.append(f"- {item}")
        else:
            lines.append(f"### {section}\n{content}")
    return "\n".join(lines) + "\n"
