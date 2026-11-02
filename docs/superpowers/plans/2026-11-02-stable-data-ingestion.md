# 稳定日频采集与数据总控 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立零采购预算、允许 AKShare、以本地真实快照保障展示的采集系统，并验收固定 30 只研究池日常增量批次的 60 秒目标。

**Architecture:** 来源适配器与网络调度分离，原文进入内容寻址归档，再通过标准化和质量门槛形成不可变数据集。数据采集使用独立 SQLite 和 worker，前端 GET 只查询本地资产；AKQuant 作为后续回测适配方向。

**Tech Stack:** 已有 Python 3.10、FastAPI、SQLite、pandas、pyarrow、httpx、Vue 3/SoybeanAdmin/ECharts；AKShare 版本须依据已测试版本固定并记录，初始参考为 1.18.88。

**Spec:** `docs/superpowers/specs/2026-11-02-stable-data-ingestion-design.md`

## Global Constraints

- 数据采购预算为 0；允许 AKShare 和自建适配器，不引入付费 Key 或把 BaoStock 串行接口设为必要前置条件。
- 总批次截止 60 秒，网络预算最多 45 秒，单采集子进程硬上限 12 秒；全局网络最多 4 个任务，同供应方初始最多 1 个，最小发起间隔 0.5 秒。
- 临时错误在剩余预算内最多重试 1 次；403 停止该来源，429 遵守 Retry-After；同一供应方的子域和库/自建实现共用预算。
- 最近合格快照可持续展示，但旧缓存不计作新数据更新成功；部分批次不发布成完整研究快照。
- 不用随机行情兜底，不把空值统一填 0，不跨供应方静默拼接复权/成交量，不覆盖旧实验的数据身份。
- 保留最终测试区间 2026-04-22 至 2026-10-30；采集验收不计算该区间因子指标。
- 本轮文档与探测不表示任务已经实施；以下复选框保持未完成，直到实际执行并验证。

## Review Focus

- 库函数卡在 DNS、TLS 或没有 timeout 的附加请求：硬截止能终止，任务和旧快照仍可查看，对应任务 2、3。
- 同供应方有多个域名和适配器：共享预算，不以域名数放大并发，对应任务 1、2。
- 基础日线成功而股本、新闻或复权失败：按资产和研究依赖处理，展示可用性与研究就绪状态分开，对应任务 3、4。
- 进程在写文件或发布途中退出：旧 manifest 完整，不出现半份新数据，对应任务 1、4。
- 最新任务失败、旧缓存有效、页面刷新：显示实际快照日期与失败状态，GET 无外网调用，对应任务 5。

## Task 1: 来源契约、原文归档与独立任务存储

**Files:** Create `thesis/ingestion/__init__.py`, `thesis/ingestion/models.py`, `thesis/ingestion/store.py`; Test `tests/test_ingestion_store.py`.

**Interfaces:**

- `SourceSpec(id: str, upstream_group: str, kinds: tuple[str, ...], units: dict, max_concurrency: int = 1, min_interval: float = 0.5)`。
- `FetchRequest(source_id: str, kind: str, instrument: str | None, start: str | None, end: str | None)`。
- `FetchArtifact(source_id: str, kind: str, path: str, sha256: str, fetched_at: str, metadata: dict)`。
- `IngestionStore(root: Path)` 提供 `enqueue(request: dict, key: str) -> dict`, `claim(worker_id: str) -> dict | None`, `record_artifact(payload: bytes, metadata: dict) -> FetchArtifact`, `finish(job_id: str, status: str, result: dict) -> dict`, `get_job(job_id: str) -> dict | None`。

- [ ] 写失败测试：相同键相同请求幂等，不同请求冲突；中断的原文写入不产生可读半文件；同供应方别名归到同一 `upstream_group`。
- [ ] 运行 `.venv/bin/python -m pytest tests/test_ingestion_store.py -q`，确认新模块缺失导致失败。
- [ ] 在独立 `runtime/ingestion.sqlite3` 实现任务、子任务、事件和原文元数据；原文以 SHA-256 落盘并原子重命名，不修改研究库的单任务锁。
- [ ] 重跑该测试，检查通过并仅审阅本任务差异。

## Task 2: 受控调度、硬截止与来源状态

**Files:** Create `thesis/ingestion/worker.py`, `thesis/ingestion/policy.py`; Test `tests/test_ingestion_worker.py`.

**Interfaces:** 消费任务 1 的 `SourceSpec`、`FetchRequest` 与 `IngestionStore`；提供 `run_bounded(request: FetchRequest, root: Path, timeout_seconds: float) -> dict`，`run_batch(job_id: str, root: Path) -> dict`。子进程执行适配器，父进程负责来源预算和截止。

- [ ] 写失败测试：冻结的子进程在硬截止后退出；并行上限和同供应方间隔有效；429 不提前重试；60 秒批次中未完成项为 deferred，旧数据仍可查询。
- [ ] 运行 `.venv/bin/python -m pytest tests/test_ingestion_worker.py -q`，确认失败。
- [ ] 实现独立 worker、进程截止和来源熔断，按全局约束限制网络；测试用可控时钟和本地假服务，不向真实供应方压测。
- [ ] 重跑任务 1、2 的测试；核对没有改变 `thesis/worker.py` 的研究执行锁与 LLM 中断语义。

## Task 3: 最小日线适配与辅助资产隔离

**Files:** Create `thesis/ingestion/providers.py`; Modify `requirements.txt` after adapter acceptance; Test `tests/test_ingestion_providers.py`.

**Interfaces:** 提供 `fetch_asset(request: FetchRequest, root: Path) -> FetchArtifact`, `parse_asset(artifact: FetchArtifact, root: Path) -> pd.DataFrame`；来源注册表返回 `SourceSpec`。网络只发生在 fetch，parse 可离线重放。

- [ ] 写失败测试：新浪基础日线只请求 K 线资产，不因股本接口失败而丢失；空表、HTML 错误页、缺列明确失败；不同日期相同 OHLCV 不被数值去重；零量原文保留；重复重放结果一致。
- [ ] 运行 `.venv/bin/python -m pytest tests/test_ingestion_providers.py -q`，确认失败。
- [ ] 参考固定源码快照实现新浪最小日线解析，并将日历、复权、证券资料、新闻拆为独立资产；AKShare 可用方法通过有界子进程调用。不得使用教材的随机数据兜底，也不复制远端字符串 `eval` 作为本项目解析入口。
- [ ] 对复权字段暂未验收的来源标记 research-ready 为 false；东财、腾讯适配器只有在样本通过后才启用。核对金额、成交量单位和日期循环边界。
- [ ] 运行前三项测试；将实际复用代码与许可写入来源清单，固定产品依赖后在隔离环境验证依赖可安装，禁止顺带升级产品 pandas/pyarrow。

## Task 4: 不可变快照、质量门槛与研究输入

**Files:** Create `thesis/ingestion/snapshot.py`; Test `tests/test_ingestion_snapshot.py`.

**Interfaces:** `build_snapshot(artifacts: list[FetchArtifact], root: Path, requirements: dict) -> dict`；`publish_snapshot(dataset_id: str, root: Path) -> dict`；`export_research_input(dataset_id: str, root: Path, target_root: Path) -> dict`。快照字典至少包含 id、来源、原文引用、交易日期范围、单位、覆盖、quality、research_ready 和资产路径。

- [ ] 写失败测试：重复键冲突、无复权证据、跨源因子、缺失行情、零量行、写入中断；部分资产不可满足研究依赖时不发布研究就绪快照；旧快照哈希不改变。
- [ ] 运行 `.venv/bin/python -m pytest tests/test_ingestion_snapshot.py -q`，确认失败。
- [ ] 标准化为分来源的不复权资产与明确复权视图，将原始零量行和未知交易状态留在资产层；只在研究视图应用评价所需过滤。
- [ ] 在新 `runtime/datasets/<id>` 写不可变 Parquet/manifest，通过校验后原子切换采集层 `current.json`；研究输入导出到独立目标根目录，符合既有 `build_baselines(root)` 目录契约，不覆盖当前归档。
- [ ] 运行 `.venv/bin/python -m pytest tests/test_ingestion_snapshot.py tests/test_research.py -q`；后者覆盖既有清洗、缺失交易日、标签对齐、切分清除与最终测试隔离，不得绕过。

## Task 5: 本地 API 与 Vue 3 数据总控

**Files:** Modify `thesis/api.py`, `frontend/src/service/api/research.ts`, `frontend/src/views/home/modules/detail-panels.vue`, `frontend/src/views/home/research.css`; Create `frontend/src/views/home/modules/data-control-panel.vue`; Test `tests/test_ingestion_api.py`.

**Interfaces:** `GET /api/data/sources`, `GET /api/data/summary`, `POST /api/data/jobs`, `GET /api/data/jobs/{id}`, `GET /api/data/coverage`；POST 使用 Idempotency-Key，返回 202 与任务 ID；GET 只读取本地。摘要分开返回当前快照与最近采集任务，不将失败藏在 ready 中。

- [ ] 写失败测试：所有数据 GET 在阻止外网的测试条件下仍工作；最新任务失败但旧快照有效时两个状态都返回；worker 不在线与无快照都有明确响应。
- [ ] 运行 `.venv/bin/python -m pytest tests/test_ingestion_api.py -q`，确认失败。
- [ ] 增加本地 API 与类型，沿用现有异常处理；数据总控组件展示截至日、来源健康、真实覆盖热力图、字段质量、任务进度和溯源抽屉。
- [ ] 手动更新仅提交后台任务；刷新后恢复任务，网络错误不清空原图表。样本不足时隐藏 P95 数字，缺失时显示空状态，不生成展示用行情。
- [ ] 运行后端相关测试，以及前端 `CI=1 pnpm typecheck`、`CI=1 pnpm build`；浏览器核对正常快照、采集中、失败缓存和无数据四种状态，截图保存为证据。

## Task 6: 30 只试点与论文证据

**Files:** Create `scripts/verify_ingestion.py`; update `docs/verification.md` with actual results only; evidence under a new run-specific `runtime/verification/` directory and `docs/evidence/`.

**Interfaces:** 验收脚本调用前述本地任务 API 或 `run_batch`，记录研究池、起止交易日、请求/来源数量、导入/HTTP/解析/校验/发布耗时、成功/失败/延后数、缓存日期和数据集 ID；不得调用最终测试评价。

- [ ] 先重放已取得原文，并注入断网、429、缺列、子进程超时、发布中断，验证旧真实快照持续可读；模拟结果不计入真实上游成功率。
- [ ] 对固定 30 只池在独立数据根目录采集有限窗口，验收复权和公司行为样本；未通过则保持 research-ready=false，禁止扩池后掩盖问题。
- [ ] 对已有历史缓存的一次真实增量批次计时，分别报告是否全量成功、是否在 60 秒内完成、是否保留旧缓存；三者不得合并为一个“成功”。
- [ ] 持续积累正常更新批次；至少 20 个真实观测后再报告 P95，并注明样本数和时间跨度，不为刷指标重复请求同一份数据。初次历史补库另报耗时。
- [ ] 30 只试点通过后再评估 300 只；AKQuant 后续仅通过已验收快照做引擎输入适配和交易语义对照，不为追求 Rust 性能更换数据来源。

## 本轮实施状态

本轮新增完成一项独立验证：使用固定版本 AKShare 隔离环境，针对当前 5,576 只 A 股完成新浪原始日线全市场采集，并完成离线哈希、覆盖、价格质量和交易日缺口审计；同时完成东财 30 只分层原始/前复权 60/60 试验和新浪复权因子 30/30 试点。证据见 `docs/a-share-full-market-test-2026-11-02.md`。

这项 probe 不等同于产品采集 worker、不可变研究快照、本地数据 API、Vue 数据总控、60 秒日常增量批次或长期并发稳定性验收。计划任务 1–6 的复选框保持未完成，直到产品模块实际实现并运行对应测试；全市场原始日线也没有替换当前腾讯归档或进入最终测试评价。
