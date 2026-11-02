# AlphaLab v1.2.0

AlphaLab 是一个面向毕业论文展示的智能量化因子研究工作台：用 Vue 3/SoybeanAdmin 做数据总控大屏，用 FastAPI、SQLite 和独立 worker 保存可追溯研究过程。

## v1.2 的真实边界

本版本已经具备：

- 30 只 A 股、19,190 条真实日频行情记录及原始响应、清洗报告和 SHA-256 指纹；
- 6 个预先定义的经典因子，仅在验证集完成逐日 IC、Rank IC、ICIR 和五分层统计评价；
- 训练/验证/最终测试时间隔离，最终测试明确保留、未评价；
- 研究总控、数据资产、因子实验室、研究进化、实验档案、论文材料六个展示页面；
- SQLite 持久任务、幂等提交、独立 worker、刷新恢复、只读材料导出；
- 本地离线归档展示，不依赖启动时继续访问行情接口。
- 数据文件完整性与归档身份检查、持续研究笔记索引、证据缺口展示和审计导出。

RD-Agent 真实源码接线、真实 LLM 反馈、候选冻结和最终测试尚未在本版本中完成。页面与 API 会明确显示“待验证/未配置”，经典因子复算不会伪装成模型生成结果。

## 本地运行

需要 Python 3.10+、Node.js 20.19+ 和 pnpm 10+。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd frontend
pnpm install --frozen-lockfile
CI=1 pnpm typecheck
CI=1 pnpm build
cd ..
PORT=18000 ./scripts/start.sh
```

然后打开 `http://127.0.0.1:18000/`。`scripts/start.sh` 会同时启动 API 和独立 worker，只监听本机回环地址。也可以分别运行 `scripts/run-api.sh` 与 `scripts/run-worker.sh`。

## 测试与复现

```sh
./scripts/verify.sh

# 基于当前真实归档复算经典基线
.venv/bin/python -m thesis.baselines --root .
```

当前归档数据已随本地 v1.2 发布包保留；数据源、请求批次、局限和评价公式见 `docs/data-evaluation.md`，API 见 `docs/api.md`。

2026-11-02 的最新数据层方向为：零采购预算，允许 AKShare，参考 AKQuant 的数据处理方式，自建调度、缓存、质量检查与数据总控。单次日常增量以固定 30 只研究池、已有历史缓存时 60 秒为待验收目标；首次多年历史补库单独计时。旧版 `thesis.acquisition` 是腾讯采集的历史实现，不再作为新接入流程的默认入口；现有真实归档继续可离线展示。

具体取舍、实测结果和后续任务见[稳定采集设计](docs/superpowers/specs/2026-11-02-stable-data-ingestion-design.md)、[执行计划](docs/superpowers/plans/2026-11-02-stable-data-ingestion.md)和[A 股全市场采集测试](docs/a-share-full-market-test-2026-11-02.md)。本轮已完成独立的 5,576 只 A 股新浪原始日线采集验证，但尚未实现新采集 worker、研究快照发布或验收一分钟增量批次，不能将这些目标描述为已完成。

持续调研的已落地更新和下一步见 [研究语料吸收报告](docs/research-integration-2026-11-02.md)。数据页将文件校验与 PIT 验证分开，论文页展示笔记和临时引用状态。公开的 `frontend/.env` 用于构建，不包含模型凭据。

## 论文使用注意

可以把本版本作为“真实数据接入、经典基线正确评价、可追溯展示系统”的实验基线。不要把当前的验证集结果写成 LLM 反馈增益或最终测试成绩，也不要把分层统计收益解释为含费用、滑点和可交易性约束的策略回测。固定股票池存在幸存者偏差；腾讯前复权是下载时点快照；这些限制已在数据页和论文材料页展示。

## 版本与数据回退

2026-11-02 数据层改造前检查点为 `alphalab-v1.2.0-data-checkpoint-20261102`，备份分支为 `backup/alphalab-20261102`，目标仓库为 `guoer9/money-agent`。代码、SQLite 一致性备份、现有基线、5,576 只全市场采集资产和前端构建分别保存并关联校验和。恢复步骤、附件范围和后续数据版本规则见[回退说明](docs/rollback.md)。发布和恢复核验状态以 Release 附件 `checkpoint-manifest.json` 为准。

## 许可证

前端底座保留 SoybeanAdmin 的 MIT 许可证与上游信息，见 `frontend/LICENSE` 和 `frontend/UPSTREAM.json`。
