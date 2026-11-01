# RD-Agent Research Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 可运行的真实数据因子研究工作台和同步论文材料。

**Architecture:** React 独立展示，FastAPI 查询/任务 API，SQLite 运行索引，Parquet 数据和评价产物；真正 RDLoop 编排经检查的确定性因子计算，最终测试独立执行。

**Tech Stack:** Python 3.10、FastAPI、pandas、scipy、RD-Agent 固定源码、React、TypeScript、Vite、ECharts。

**Spec:** ../specs/2026-11-01-workbench-design.md

## Global Constraints

- 原 qlib_trading 工作区不修改；新目录为独立开发分支。
- ≥10000 条真实有效多股票记录；离线、live、archive 明确区分。
- 训练/验证/测试隔离，标签按退出日 purge；缺失指标为 null。
- 不执行任意模型 Python，不以模拟结果代替真实 LLM。
- 所有前端页面均需要空/错误/加载/成功状态，前端计算不产生研究指标。

## Review Focus

- 索引重新排序不得串股，未来变化不得改变过去因子；Task 1 数值测试。
- 模型缺凭据/输出非法不得默默降级；Task 3 错误测试。
- 浏览器刷新/重复提交不得重复运行；Task 4 持久化与幂等测试。
- 测试集不得进入优化，比较不同协议不得混图；Task 2/4 隔离与契约测试。
- 页面无数据/断网仍能解释状态；Task 5 浏览器检查。

---

### Task 1: 数据契约与确定性因子评价

Files: thesis/data.py, thesis/factors.py, thesis/evaluation.py, tests/test_research.py.
Interfaces: clean_panel(frame) -> (DataFrame, dict); calculate(expression, panel) -> Series; evaluate(factor, panel, segment) -> dict.
- [ ] 先写失败测试：乱序标签、前缀因果、非法语法、日IC与分层、常数/null、purge 边界。
- [ ] 运行 pytest，确认因模块未实现失败。
- [ ] 实现数据清洗、AST 算子、逐日评价与时点协议。
- [ ] pytest 通过并记录证据，提交代码。

### Task 2: 公开行情与经典因子归档

Files: thesis/acquisition.py, thesis/cli.py, tests/test_acquisition.py.
Interfaces: fetch_market(codes, start, end, output) -> dict; build_baselines(root) -> list[dict].
- [ ] 用真实接口响应格式写测试并观察失败。
- [ ] 实现超时、原始快照、来源与去重/单位/复权质量报告。
- [ ] 获取真实多股票数据，确认清洗后≥10000条；运行经典基线并保存真实指标。
- [ ] 接口失败明确存档，不能用随机数据填补；测试通过后提交。

### Task 3: 上游研究循环与生成适配

Files: vendor/RD-Agent, thesis/rd_adapter.py, thesis/llm.py, tests/test_rd_adapter.py.
Interfaces: run_research(request, root, emit) -> dict; generate_candidate(request, feedback) -> Candidate.
- [ ] 测试真实 RDLoop/FactorTask 调用、解析失败和无凭据行为；先观察失败。
- [ ] 固定上游源码及来源，建立独立 Python 环境，锁定实际通过依赖。
- [ ] 实现真实上游循环的课题适配，不用自写替身冒充集成；经典运行使用显式 baseline 模式。
- [ ] 真实 LLM 在可用凭据下完成一轮，否则保留明确未验证状态；测试并提交。

### Task 4: API、任务与实验追溯

Files: thesis/store.py, thesis/api.py, thesis/worker.py, tests/test_api.py.
Interfaces: /api/overview, /api/datasets, /api/runs, /api/runs/{id}, /api/factors, /api/reports.
- [ ] 写持久化、重复提交、失败状态、只读归档和测试集隔离的测试并观察失败。
- [ ] 独立 worker 执行，SQLite 记录事件和状态；API 永不在请求进程运行任意代码。
- [ ] 每项结果附数据/协议/版本及父子关系；模型配置只暴露可用状态。
- [ ] 通过 API 和集成测试，提交。

### Task 5: 中文研究工作台

Files: frontend/src/App.tsx, frontend/src/styles.css, frontend/src/api.ts, frontend/src/charts.tsx.
Consumes: Task 4 API；图表只绘制后端真实数据。
- [ ] 建立前端、6个视图和状态组件；先接真实经典归档形成可见成果。
- [ ] 实现自然语言输入、任务进度、因子详情、日IC/分层、父子对照和论文材料导出。
- [ ] 类型检查和生产构建通过；浏览器检查 1440/1024 宽度及错误/空状态。
- [ ] 校验按钮可用、指标真实、刷新恢复，保存截图并提交。

### Task 6: 系统验收与论文材料

Files: README.md, docs/verification.md, paper/论文初稿.md, paper/实验索引.md.
- [ ] 汇总自动化测试、数据质量、上游调用和浏览器证据。
- [ ] 写可从存档复算的运行说明和当前风险表，标明尚未实测项。
- [ ] 同步论文需求、架构、方法、测试与真实结果，未完成实验不能写结论。
- [ ] 独立审阅关键链路，修正影响数据/结果/界面的缺陷；完成后续工作而非仅交计划。
