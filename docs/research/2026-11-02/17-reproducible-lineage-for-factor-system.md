# 2026-11-02 第 17 轮：量化因子系统的数据与运行证据链

## 搜索问题

怎样把原始行情响应、清洗后的面板、因子/评估运行和最终归档连成可复现、可审计的证据链，使系统能够区分“本地文件完整”“实验可复现”“点时数据已验证”和“真实 LLM 生成已发生”？本轮只研究能直接转化为 `manifest`、运行记录和 API 检查项的设计要求，不修改交易系统实现，不运行最终测试。

检索范围集中在 provenance、数据集指纹、Git/数据版本、运行与输入输出 lineage。访问日期统一为 2026-11-02（Asia/Shanghai）。

## 本机状态与核对命令

工作根目录：`/Volumes/2tb/mydata/code/Quantitative_trading/rdagent_thesis`。

本轮先执行：

```text
git status --short
sed -n '1,260p' docs/research/00-progress-index.md
sed -n '1,320p' docs/research/01-current-status.md
sed -n '1,280p' thesis/evidence.py
sed -n '228,268p' thesis/acquisition.py
sed -n '1,135p' runtime/artifacts/baselines.json
```

观察到的当前实现事实：

- `runtime/data/manifest.json` 记录数据集 ID、供应方/端点、下载时间、请求范围、股票池、质量字段、切分、面板 SHA-256、原始响应路径与 SHA-256，以及规范化价格指纹。
- `thesis/evidence.py` 会校验面板和原始响应的本地字节指纹，并把当前归档中的 `dataset.id` 与面板 SHA-256 做关联核对。
- `runtime/artifacts/baselines.json` 记录了数据集、协议、经典 baseline 结果和生成时间；当前归档没有形成完整的“代码版本—运行参数—输入对象—输出对象”运行图。
- 研究代码明确把完整性判断与 PIT、实验有效性分开；哈希一致不能证明供应方没有历史回写，也不能证明真实 LLM 生成已经发生。

这些是仓库观察，不是外部来源对本项目行为的背书。

## 新增来源

### S1：W3C PROV-Overview

- 标题：PROV-Overview
- URL：<https://www.w3.org/TR/prov-overview/>
- 发布时间：2013-04-30（W3C Working Group Note）
- 访问日期：2026-11-02
- 来源类型：W3C 官方 provenance 标准族概览
- 原文依据：文档把 provenance 定义为涉及数据/事物生产过程中的 entities、activities 和 people/agents 的信息，并明确其用途包括质量、可靠性和可信度评估；概览列出可支持的 reproducibility、versioning、procedures 和 derivation 等要求。
- 对本项目的适用性：可以把原始供应方响应、清洗面板、因子结果和归档视为实体；把采集、清洗、baseline 计算和 LLM 运行视为活动；把代码版本、模型/供应方和操作者视为责任主体，并要求保存实体之间的派生关系。
- 限制：这是通用 provenance 模型，不规定量化数据的 PIT 判定、复权公式、IC 计算或 LLM provider 的真实响应格式；采用该模型只能改善可追溯性，不能自动证明回测无泄漏。

### S2：MLflow Dataset Tracking

- 标题：ML Dataset Tracking
- URL：<https://mlflow.org/docs/latest/dataset/>
- 发布时间：页面未标注；访问日期 2026-11-02
- 来源类型：MLflow 官方文档
- 原文依据：官方文档把 Dataset 建模为带有 name、digest、source、schema 和 profile 的对象，并通过 `log_input` 将数据集与运行关联；文档要求记录能追溯到原始数据的 source 标识、数据上下文和预处理/采集元数据，同时区分数据内容指纹与仅记录元数据的情况。
- 对本项目的适用性：`manifest.id`、面板 SHA-256、原始 URL、schema/quality、`train/validation/test` 上下文可以组成量化数据集对象；baseline 或真实 LLM 运行应保存输入数据集 ID 和上下文，避免只保存最终指标。
- 限制：MLflow 的 digest 语义依赖数据集类型和实现；source 记录并不保证变换前的源文件可以被重新下载，也不解决供应方历史回写、历史股票池或公司行动版本问题。本项目不能因为保存 digest 就宣称 PIT 已验证。

### S3：DVC User Guide

- 标题：User Guide | Data Version Control
- URL：<https://doc.dvc.org/user-guide>
- 发布时间：页面未标注；访问日期 2026-11-02
- 来源类型：DVC 官方文档
- 原文依据：DVC 文档要求用 Git/DVC 元文件共同管理源代码、配置、参数、指标、数据资产和处理过程；数据大文件可以放在仓库外，但其版本元数据应随代码版本化。
- 对本项目的适用性：即使不引入 DVC，也可以采用其设计原则：把数据 manifest、协议配置、候选/运行参数和产物指纹作为同一实验版本的元数据保存；代码提交或工作树状态应成为运行证据的一部分。
- 限制：DVC 的版本指针不等于数据内容永远可取，也不提供公司行动的历史生效规则；当前工作区存在大量未提交并行实现，不能把当前 Git 工作树直接称为稳定代码版本。若未来引入 DVC，还需决定数据许可、远端存储和复现凭据。

### S4：OpenLineage Object Model 与 Job Facets

- 标题：Object Model；Job Facets
- URL：<https://openlineage.io/docs/next/spec/object-model/>；<https://openlineage.io/docs/next/spec/facets/job-facets/>
- 发布时间：页面未标注；访问日期 2026-11-02
- 来源类型：OpenLineage 官方规范文档
- 原文依据：规范区分运行时 `RunEvent`、作业静态元数据 `JobEvent` 和数据集元数据 `DatasetEvent`；Job 可记录 source code location/version，Run 有唯一 run ID 和时间信息，Dataset 可记录 namespace/name、schema、version、输入/输出分区、数据质量和输出统计，并建议用明确的 lineage 关系表达输入与输出，而不是只依赖事件边界推断。
- 对本项目的适用性：可把一次行情采集、baseline 构建或 LLM 因子评估建模为独立 run；把 `manifest.json`、`panel.parquet`、`baselines.json`、因子结果和日志作为输入/输出数据集；把 Git SHA、协议 ID、数据切分、质量检查结果和错误状态分别记录在运行或数据集元数据中。
- 限制：OpenLineage 是通用事件规范，当前文档不能替代本项目的研究协议；它不会自动校验 IC、Rank IC、IR、交易成本或 PIT。`next` 文档页面可能变化，采用时必须锁定规范版本并保存 schema URL。

## 事实、推断与代码检查项

### 外部来源支持的事实

1. 可审计 provenance 至少需要描述实体、处理活动、责任主体、版本和派生关系；单独保存一个文件哈希不足以表达完整运行过程。
2. 数据集指纹应和 source、schema、profile、用途上下文一起保存；数据集指纹与元数据来源是不同维度。
3. 可复现实验需要把代码/配置/参数/指标与数据版本联系起来；数据版本元数据可以和代码版本一起管理。
4. 运行级 lineage 应有唯一 run ID，并明确输入数据集、输出数据集、源代码版本、时间和质量/错误信息；静态数据集元数据与每次运行实际使用的分区或子集应区分。

### 面向本项目的推断

1. `manifest.sha256`、`raw_responses[*].sha256` 和 `normalized_prices_sha256` 已经覆盖“字节/规范化内容完整性”的一部分，但还没有覆盖“哪个代码版本和参数生成了该面板或归档”。因此证据 API 当前可以回答“文件是否被改动”，不能完整回答“能否用同一代码和参数重新产生”。
2. `baselines.json` 现在通过 `dataset.id` 与面板 SHA-256 连接数据和结果，这是一个有效的最小连接；下一步应把它扩展为明确的 `run_id`、`input_dataset_ids`、`output_artifact_ids`、`code_revision`、`config_digest`、`protocol_digest` 和 `environment_digest`。
3. 真实 LLM 运行还应把 provider、模型标识、请求参数摘要、响应/产物哈希、失败状态和凭据存在性分开记录。没有真实 provider 凭据时，只能记录 `llm_unavailable` 或 `unverified`，不能用经典 baseline 代替。

## 可转化为系统开发检查项

| 检查项 | 最小证据 | 当前状态 |
| --- | --- | --- |
| 数据实体可定位 | `dataset_id`、source/endpoint、请求区间、股票池、schema、数据版本或内容指纹 | 部分已有；PIT 版本/历史股票池仍未验证 |
| 原始到规范化派生 | 每个原始响应、规范化面板、清洗规则版本和派生关系 | 原始/规范化哈希已有；清洗规则版本未单独登记 |
| 运行可复现 | `run_id`、代码 revision、配置/协议 digest、环境或依赖锁定信息 | 未完整登记 |
| 输入输出可追溯 | 运行的 input dataset IDs、output artifact IDs、父运行/重试关系 | 目前主要只有归档数据集链接 |
| 质量与限制可见 | 行数、缺失、重复、日历、PIT 状态、survivorship 标记和失败原因 | 数据质量与限制已有；未绑定到独立 run |
| LLM 真实性边界 | provider/model、真实请求状态、响应产物、失败原因、凭据状态 | 当前无真实凭据，必须保持未验证 |
| 公共 API 安全 | 只公开摘要、哈希和状态，不泄露凭据、原始标签或测试目标 | 现有 `public_json` 已有边界；本轮未运行测试 |

## 未决项与限制

- 本轮没有修改实现，没有运行测试、typecheck/build、worker/API smoke 或最终测试；开发检查项仍未代码级通过。
- 当前工作区有大量未提交实现，无法直接把工作树状态写成稳定 release 版本；后续需要约定以提交 SHA、可重放补丁或发布归档哪一种作为运行代码凭据。
- 当前数据仍是供应方 qfq 快照，缺少逐事件复权因子、历史时点股票池和完整公司行动版本；provenance 完整不等于 PIT 通过。
- MLflow、DVC、OpenLineage 都是外部工具/规范参考，本轮没有引入依赖，也没有决定采用其中任何一个。
- `pending decision`：后续实现采用轻量自定义 manifest 扩展，还是引入 DVC/MLflow/OpenLineage 的部分格式；在没有明确数据许可、存储和部署边界前，先保留自定义 JSON 兼容层。
- `pending decision`：运行环境证据采用 Python 依赖锁文件、容器镜像摘要，还是两者并存；未决定前不能声称跨机器可复现。

## 下一轮唯一优先检索方向

继续围绕**因子生成与代码校验**检索一手来源，重点确认如何把生成代码的语法/运行/数值域检查、随机性和失败样本纳入 run manifest，并明确哪些检查只能证明代码可执行、哪些才能支持因子定义与回测协议一致；不修改策略代码，不运行最终测试。

## 本轮结论

本轮新增来源数：**4**。最可复用的开发结论是：把数据集、处理活动、运行、代码版本和产物之间的关系显式记录；将内容哈希、来源标识、schema/quality、运行参数和代码 revision 分开建模；继续保留“完整性 ≠ PIT 验证 ≠ 真实 LLM 生成”的边界。

证据路径：本文、`thesis/evidence.py`、`thesis/acquisition.py`、`runtime/artifacts/baselines.json`。本轮没有把 baseline、经典因子或模拟结果冒充真实 LLM 生成或真实回测结果。
