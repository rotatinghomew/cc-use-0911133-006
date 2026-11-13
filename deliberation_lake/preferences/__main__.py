"""命令行入口：

    python3 -m deliberation_lake.preferences fixtures/preference_portfolios.json \
        --output /tmp/preferences
"""

from __future__ import annotations

import argparse
import json

from deliberation_lake.preferences.analysis import run_analysis
from deliberation_lake.preferences.report import write_outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="园区协商偏好组合分析：非支配组合、群体覆盖、循环证据与稳定性"
    )
    parser.add_argument("fixture", help="偏好资料 JSON 路径")
    parser.add_argument("--output", required=True, help="分析结果输出目录")
    parser.add_argument("--display-cap", type=int, default=50,
                        help="Markdown/摘要中每个前沿最多展示的组合数（默认 50）")
    args = parser.parse_args()

    result = run_analysis(args.fixture, args.output, display_cap=args.display_cap)
    paths = write_outputs(result, args.output)

    small = result["small_case"]
    large = result["large_case"]
    console = {
        "输出文件": paths,
        "小案例": {
            "可行组合数": small["exhaustive_cross_check"]["feasible_portfolio_count"],
            "穷举与分支定界前沿一致": small["exhaustive_cross_check"]["fronts_match"],
            "非支配前沿大小": small["pareto_front"]["size"],
            "可接受前沿大小": small["acceptable_front"]["size"],
            "当前口径循环": small["cycle_evidence"]["cycles_under_current_versions"],
            "原始口径循环": small["cycle_evidence"]["cycles_under_original_expressions"],
            "失去可接受方案的群体": _lost_groups(small),
        },
        "声明": result["disclaimer"],
    }
    if large is not None:
        console["大案例"] = {
            "搜索空间": large["search_space_size"],
            "访问节点数": large["search"]["nodes_visited"],
            "非支配前沿大小": large["pareto_front"]["size"],
            "可接受前沿大小": large["acceptable_front"]["size"],
            "耗时秒": large["search"]["elapsed_seconds"],
        }
    print(json.dumps(console, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _lost_groups(case_result: dict[str, object]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for scenario in case_result["scenarios"]:
        if scenario["groups_without_acceptable_option"]:
            output[scenario["key"]] = scenario["groups_without_acceptable_option"]
    existence = {
        gid: info["has_acceptable_feasible_portfolio"]
        for gid, info in case_result["acceptable_option_existence"].items()
    }
    output["硬约束存在性检查"] = sorted(
        gid for gid, ok in existence.items() if not ok
    )
    return output


if __name__ == "__main__":
    raise SystemExit(main())
