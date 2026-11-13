"""隔离模式启动器（python3 -I）：显式加入工作区包路径后转交 CLI。

用途：处理不可信资料时，用 ``python3 -I scripts/run_preferences.py ...``
运行，避免当前目录中的同名模块（如 planted json.py）被加载。
正常环境下等价于 ``python3 -m deliberation_lake.preferences``。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deliberation_lake.preferences.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
