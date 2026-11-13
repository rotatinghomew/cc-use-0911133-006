"""组合式偏好分析（协商材料，不产生承诺）。

本包只做分析，不写入 ``deliberation_lake.commitments`` 的承诺台账。
所有输出都带有明确的方法学说明与口径，不输出单一"客观排名"。
"""

from deliberation_lake.preferences.model import (
    Case,
    Proposal,
    Ballot,
    load_case,
)
from deliberation_lake.preferences.analysis import analyze_case, run_analysis

__all__ = [
    "Case",
    "Proposal",
    "Ballot",
    "load_case",
    "analyze_case",
    "run_analysis",
]
