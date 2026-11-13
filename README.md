# deliberation_lake

园区协商资料处理：议题、利益方、提案关系、证据、承诺和后续核验。当前入口是 `deliberation_lake.console.run`，它使用本地 JSON 资料和 SQLite 状态；目录 `deliberation_lake` 存放业务模块，`tests/deliberation_cycles` 验证既有业务行为。

## 环境

使用 Python 3.11，全部依赖来自标准库。`requirements.txt` 为空，不需要下载数据、模型或第三方包。数据库在运行时创建，可以删除临时数据库重新开始。

## 检查

```bash
python3 -m unittest discover -s tests/deliberation_cycles -v
python3 -m unittest discover -s tests/analysis_context -v
python3 -m compileall -q deliberation_lake tests scripts
```

## 运行

```bash
python3 -m deliberation_lake.console.run  fixtures/demo.json --database /tmp/deliberation_lake-demo.sqlite3
python3 scripts/check_project.py
```

`check_project.py` 串行执行测试、编译和真实业务入口，将本次业务输出完整打印到终端；状态文件使用独立临时目录，重复运行不沿用上次状态。

批量分析资料位于 `fixtures/preference_portfolios.json`，字段含义见 `docs/data.md`。它与当前单笔流程的输入格式不同，不能直接交给原来的入口处理。

## 偏好组合分析

组合式协商材料分析（不产生承诺、不替代理事会决策）：

```bash
python3 -m deliberation_lake.preferences fixtures/preference_portfolios.json --output /tmp/preferences
# 处理不可信资料时可用隔离模式启动器（不依赖当前目录的同名模块）：
python3 -I scripts/run_preferences.py fixtures/preference_portfolios.json --output /tmp/preferences
```

输出目录包含 `analysis_summary.json`、`small_front_records.json`、
`large_front_records.json` 与一份理事会可读的 `council_briefing.md`。
口径要点：

- 预算、依赖、互斥、`unacceptable` 都是**硬边界**，不做软扣分；
- 部分排序与并列不虚构先后，未列出不等于反对，不做默认补全、不输出综合分；
- 同代表仅最高 `revision` 生效，重复提交不增加席位；提案改版后仅当前版本或
  `compatible_ballot_versions` 白名单内的表达有效，其余逐条作废记录；
- 小案例完整穷举并与分支定界前沿核对；大案例用结构缩减 + 分支定界，
  不把全部子集装入内存；
- 三个独立目标（成本、整体偏好损失、最不满意群体损失）只做帕累托非支配判定，
  并给出缺席/席位扰动情景、循环偏好证据与稳定性，最终选择权在人。
