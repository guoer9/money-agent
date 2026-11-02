# 2026-11-02 第 3 轮：Point-in-Time 数据、公司行动与生存者偏差

## 本轮搜索问题

腾讯 `qfqday` 的下载时点前复权快照，加上一个在下载时点选定的固定 30 只股票池，会给 LLM 量化因子研究带来哪些**可审计的时点偏差风险**？在不改动交易系统实现的前提下，本项目应记录哪些最小数据字段、检查哪些可检测信号，才能避免把历史快照误称为 point-in-time（PIT）数据？

本轮只做资料研究。未修改 `thesis/`、`frontend/`、`tests/`、交易策略代码或 `/Volumes/2tb/mydata/code/Quantitative_trading/qlib_trading`；未运行最终测试。

## 检索与访问记录

- 访问日期：2026-11-02（Asia/Shanghai）
- 检索入口：Crossref API、arXiv 摘要页/API/PDF、CRSP 官方产品页、MDPI 文章页及出版商 PDF、Cambridge Open Engage 摘要页
- 关键词：`point-in-time data`、`corporate actions`、`survivorship bias`、`delisting bias`、`back-adjusted prices`、`historical index constituents`
- 来源分层：
  - **全文已读**：S3 的 25 页 arXiv PDF；S4 的 29 页 MDPI PDF。
  - **仅摘要/元数据已核验**：S1；S5。
  - **仅官方产品页已核验**：S2。
- 访问限制：
  - CRSP US Stock Database Guide 下载入口返回 HTTP 401；未读指南正文。
  - Wiley/OUP/机构仓库的 S1 或 Brown 等 PDF 返回 HTTP 403 或非 PDF 页面；未绕过付费墙。
  - MDPI 网页文章页返回 HTTP 403，但出版商 PDF 可公开下载，已按 PDF 全文核验。
  - SEC、中国交易所及监管网页本轮有 SSL、404 或占位页问题；本轮不以这些页面为依据。

## 来源与原文依据

### S1. Shumway 与 Warther（1999）：退市收益缺失偏差

- 标题：The Delisting Bias in CRSP's Nasdaq Data and Its Implications for the Size Effect
- URL：https://doi.org/10.1111/0022-1082.00192
- 作者：Tyler Shumway、Vincent A. Warther
- 发布时间：1999-12（Crossref `published-print`；在线元数据另记 2002-12-17）
- 访问日期：2026-11-02
- 来源类型：The Journal of Finance 正式论文；本轮核验 Crossref 元数据与公开摘要
- 原文依据（摘要）：
  - 研究 CRSP Nasdaq 数据中退市股票收益缺失造成的偏差。
  - 摘要称缺失收益平均而言“large and negative”，退市股票流动性显著下降。
  - 作者估计，对业绩相关退市使用约 `-55%` 的修正收益可校正该偏差；这是作者在其样本与方法下的估计，不是本项目结果。
  - 作者据此重新审视 Nasdaq 规模效应，并报告校正后没有证据表明曾存在该效应。
- 对本项目的适用性：
  - 支持把“退市/停止交易后的收益如何记录”列为独立的数据治理问题。
  - 支持在股票退出股票池时保留明确原因与退出收益，而不是静默消失。
- 限制：
  - 本轮仅核验公开摘要与元数据，未读正文。
  - `-55%` 不可外推为 A 股、腾讯数据或本项目的缺失收益修正参数。
  - CRSP Nasdaq 的历史制度、样本期与退市定义均不同于本项目。

### S2. CRSP 官方产品页：不活跃证券与永久标识符

- 标题：CRSP US Stock Databases
- URL：https://www.crsp.org/research/crsp-us-stock-databases/
- 发布方：Center for Research in Security Prices（CRSP）
- 发布时间：页面未明确给出；不猜测
- 访问日期：2026-11-02
- 来源类型：官方产品页
- 原文依据（页面正文）：
  - 数据库包括每日/月度市场数据和公司行动，覆盖超过 36,000 个 active and inactive securities。
  - 使用 `PERMNO` 与 `PERMCO` 永久标识符，支持回测、时间序列/事件研究、业绩衡量和证券分析。
- 对本项目的适用性：
  - 提供了一个数据设计参照：股票代码不是稳定实体标识；历史研究需要能跨代码变更、并购和不活跃状态追踪证券/发行人。
  - 支持把“active 与 inactive 证券都在历史股票池中可解释地出现”作为数据源评估项。
- 限制：
  - 该产品页没有给出 survivorship、delisting、point-in-time 或复权方法的完整定义。
  - 不能据此宣称 CRSP 已完全消除生存者偏差，也不能宣称本项目已经具备 PIT 能力。
  - 官方数据库指南本轮返回 HTTP 401，未读正文。

### S3. Ranse（2026）：用当前成分股回测的生存者偏差实验

- 标题：Survivorship Bias in Emerging Market Small-Cap Indices: Evidence from India's NIFTY Smallcap 250
- URL：https://arxiv.org/abs/2603.19380
- DOI：https://doi.org/10.48550/arxiv.2603.19380
- 作者：Harjot Singh Ranse
- 发布时间：2026-03-19（arXiv 首次提交）
- 访问日期：2026-11-02
- 来源类型：arXiv 预印本；本轮下载并读取 25 页 PDF
- 文件核验：
  - 下载字节数：`1,609,650`
  - SHA-256：`03f77a694c32b082a060081805bfcf23eeb4aa924ca6bce8b76347134b25c021`
  - 本地 PDF 元数据页数：25
- 原文依据（全文）：
  - 使用 1,437 只股票、2016–2025 年的数据，比较“只用当前指数成分股”和“包含历史成员的股票池”的等权组合。
  - 作者报告：只用当前成分股的年化收益高估 `4.94` 个百分点（相对高估 `23.3%`），Sharpe 高估 `0.097`（相对高估 `9.1%`）。
  - 作者报告 82.5% 的成员移除率，其中退市 16.1%、升级 33.1%、降级 33.2%；作者强调升级和降级也会造成历史股票池选择偏差。
  - 由于缺乏官方历史成分股，作者以成交额作为自由流通市值的代理，用季末近似调仓日重建股票池；当前成分股匹配 252/252，历史重建准确率估计 85–90%。
  - 作者在限制/未来工作中明确提到：需要公司行动（并购、分拆、代码变更）数据库，以及直接退市收益而非仅从最后成交价推断。
- 对本项目的适用性：
  - 直接支持：固定当前股票池不能代表历史可投资股票池，且偏差不只来自“破产退市”，也来自后续晋级、降级或被移出观察范围。
  - 支持把 `universe_asof`、历史成员关系和退市/迁移原因纳入最小审计字段。
  - 支持在结果中明确区分“当前固定池上的研究结果”和“历史时点股票池上的策略结论”。
- 限制：
  - 单一作者预印本，未同行评审，未在本项目复现。
  - 印度小盘股、NSE 数据和季度重建方法不能直接映射到 A 股或腾讯数据。
  - 历史成分股重建本身只有估计准确率；季末近似、成交额代理、公司行动与直接退市收益均有限制。
  - 上述 `4.94pp`、`23.3%`、`0.097`、`82.5%` 全部是作者结果，不能写成本项目的偏差估计或真实回测结果。

### S4. Fonseca（2026）：PIT 正确性与三类前视偏差

- 标题：Point-in-Time Backtesting of Momentum-Trend Equity Strategies: A Formal Bias Taxonomy, ATR Trailing Stop Analysis, and Investor-Experience Metrics
- URL：https://doi.org/10.3390/math14122182
- 作者：Xavier Fonseca
- 发布时间：2026-06-17
- 访问日期：2026-11-02
- 来源类型：Mathematics 14(12), 2182；开放获取正式期刊论文
- 许可：CC BY 4.0（Crossref 元数据）
- 文件核验：MDPI 出版商 PDF 为 29 页；下载字节数 `1,916,015`；SHA-256 `8a7947f15c92fa9b410f8be403a1df31263f2c4b5d531f32622ac7f5512e9307`
- 原文依据（全文）：
  - 形式定义：若对每个决策时刻 `t`，策略信息集都落在自然滤波 `F_t` 内，则策略是 point-in-time correct。
  - **Class I — universe-membership contamination**：若在时刻 `t` 的合格股票池使用了在 `t` 之后才发布或确定的指数成员名单，则发生股票池污染；当前成分股回测会让后来加入者进入过去、让后来被移除者从过去消失。
  - **Class II — price-data forward leakage**：若用于时刻 `t` 决策的价格序列包含 `t` 之后才发生的拆股、分红或其他公司行动造成的调整，则发生价格前视泄漏。论文明确描述供应商常用回看调整序列，把未来拆股/分红比例回溯乘到历史价格上。
  - 论文建议的缓解方式是：原始未复权价格加显式公司行动处理，或者供应商提供的 point-in-time adjusted series。
  - 论文也承认其自身方法在大型发达市场样本上的价格调整残差较小，但明确说“原始价格 + 显式公司行动”才能完全消除该残差；高股息或频繁拆股的市场应重新测量，不能假定可忽略。
- 对本项目的适用性：
  - 直接支持把腾讯前复权快照视为一个**需要验证的 Class II 风险**，而不是天然正确的历史价格。
  - 支持把固定当前 30 只股票池视为 **Class I 风险**。
  - 支持在未来设计中同时保留原始价、公司行动事件和复权后价格，避免单一 `adj_close` 承担全部语义。
- 限制：
  - 论文的实证市场、策略和收益数字不是本项目结果。
  - 论文的价格调整残差结论来自其 NASDAQ-100 大市值样本，不能据此推断本项目残差大小。
  - 论文也承认其分类不是所有前视偏差的穷尽列表。

### S5. Wei Meng（2026）：PIT 数据治理与可审计停止规则（设计线索）

- 标题：Auditable Halt Rules and Point-in-Time Data Governance in Low-Signal Equity Research: An A-Share Strategy–Forecast Competition Framework with a Single-Stock Proof of Concept
- URL：https://doi.org/10.33774/coe-2026-pk2mg
- 作者：Wei Meng
- 发布时间：2026-08-07
- 访问日期：2026-11-02
- 来源类型：Cambridge Open Engage 预印本（Crossref `posted-content`）；本轮核验公开摘要与元数据
- 原文依据（摘要）：
  - 提出以 point-in-time 数据治理、策略合格性登记、样本外竞争、统计/经济阈值和停止码组成的可审计框架。
  - 摘要称基础设施包括 19 张数据表、13 类来源与 15 个质量门。
  - `VALIDATION-HALT` 约束策略声明，`FORECAST-HALT` 约束预测声明。
  - 确认性设计要求至少 2,000 只 PIT A 股、500 万个股票日、150 万个样本外预测、6 万个公司季度观测和每核心策略 200 笔样本外交易，并包含多重检验控制、风险因子调整、状态分层和冻结前瞻验证。
- 对本项目的适用性：
  - 提供“证据不足时应停止而非继续包装结论”的设计语言。
  - 支持把数据版本、股票池适用性、样本量和多重检验写入审计状态。
- 限制：
  - 预印本，未同行评审；proof of concept 是单股票研究。
  - 19 张表、15 个质量门和样本量要求是作者框架/设计要求，不是本项目已实现能力。
  - 不能把摘要中的收益、RMSE、Sharpe 或样本量当作本项目结果或最低充分条件。

### 补充书目锚点：Brown、Goetzmann、Ibbotson、Ross（1992）

- 标题：Survivorship Bias in Performance Studies
- URL：https://doi.org/10.1093/rfs/5.4.553
- 发布时间：1992-10（OpenAlex/Crossref 元数据）
- 来源类型：Review of Financial Studies 正式论文；OpenAlex 标记为闭源，OUP PDF 返回 HTTP 403
- 本轮用途：只记录书目锚点与访问限制，不引用正文细节，不计入 5 个核心结论来源。

## 事实、作者观点与本项目推断

### 来源支持的事实

1. 历史回测若使用在决策时点之后才确定的股票名单，会把后来加入者带入过去、把后来移出者从过去删除；这是独立于价格调整的股票池污染。
2. 回看调整价格可能把未来公司行动比例写入过去价格；若研究声称“当时可观察”，需要原始未复权价格与显式公司行动，或供应商保证的 PIT 调整序列。
3. CRSP 官方产品页明确把 active and inactive securities、公司行动和永久标识符作为产品能力；但该页面不足以证明具体 PIT 实现细节。
4. 纳指小盘指数案例报告了使用当前成分股造成的收益与 Sharpe 高估；其偏差来源包括退市、晋级与降级，而不仅是破产退市。
5. Nasdaq 退市收益缺失可能存在显著负偏差；该结论来自作者摘要，不能外推为 A 股参数。
6. 文献明确把“直接退市收益”和“公司行动/代码变更数据库”列为提升生存者偏差量化质量的条件。

### 作者观点/作者实验结果

1. Ranse 认为，新兴市场小盘股的生存者偏差可能显著大于发达市场样本；其报告值仅为该印度样本的结果。
2. Fonseca 认为，PIT 正确性需要同时处理股票池、价格调整和日内退出顺序；其分类框架不是所有偏差类型的穷尽列表。
3. Meng 认为，低信号股票研究应以冻结前瞻验证、停止码和可审计数据治理约束声明；其单股票实验不能支持一般化业绩结论。

### 本项目推断（尚未通过代码或实验验证）

1. 本项目当前 30 只股票池是在下载时点选定的便捷池，符合 Class I 股票池污染的结构性风险；这并不证明某个具体因子已经被高估多少。
2. 腾讯 `qfqday` 是下载时点的前复权快照，且归档没有公司行动因子/事件表；因此它不能独立充当完整 PIT 审计证据。后续公司行动是否已经改写该接口返回的历史前复权价格，**本轮没有执行重复下载对比，不能写成已证实现象**。
3. 单一 `adj_close` 不足以同时表达原始成交价、复权因子、公司行动公告日/生效日、证券退出与退市收益。至少应把价格语义和数据版本拆开记录。
4. 印度论文的 `4.94pp`、`23.3%` 和 `0.097` 不能套用为本项目偏差估计；本项目甚至尚未建立历史时点股票池和退市样本。
5. 本轮资料只支持设计检查项，不支持修改策略、重算因子或改变最终测试状态。

## 映射到当前数据与最小审计字段

当前可从 `runtime/data/manifest.json` 核验的事实：

- `source` 为 `Tencent fqkline HTTP / qfqday`，`adjustment` 为 `qfq`。
- 下载时间为 `2026-11-01T14:39:10.779278+00:00`；实际面板起止为 2024-03-11 至 2026-10-30。
- `quality.survivorship_bias=true`，`universe_policy` 明确为 `fixed convenience pool selected at acquisition; not point-in-time constituents`。
- `raw_prices_available=false`，`adjusted_ohlc=true`。
- `adjustment_warning` 明确为 `vendor qfq snapshot; adjustment factors and point-in-time corporate actions are not provided`。
- 面板为 19,190 行、30 只股票；请求区间被 640 根上限截断，`single_range_per_instrument=true`、`anchor_splicing=false`。
- 归档包含原始响应 URL、UTC 下载时间、字节数与 SHA-256；这些是数据来源审计基础，但不等于 PIT 事件审计。

建议的最小字段（设计建议，不代表已实现）：

| 字段 | 用途 | 当前状态 |
| --- | --- | --- |
| `data_version` | 唯一标识一次数据快照及其生成规则 | manifest 有 id/hash；未形成统一数据版本对象 |
| `source` / `request_url` | 回溯供应方与原请求 | 已有 |
| `retrieved_at_utc` | 区分下载时点与历史事件时点 | 已有 |
| `raw_sha256` / `panel_sha256` | 原始响应与派生面板可复核 | 已有 |
| `price_adjustment` | `raw` / `qfq` / `hfq` 等口径 | 已有 `qfq` |
| `corporate_action_asof` | 复权计算截至哪个事件日 | 未见 |
| `universe_asof` / `universe_rule` | 股票池的历史有效期与入选规则 | 仅固定池说明，无历史成员表 |
| `listing_date` / `delisting_date` | 证券生命周期边界 | 未见 |
| `event_type` | 分红、拆股、并购、代码变更、退市等 | 未见事件表 |
| `announcement_date` / `event_effective_date` | 区分信息可知日与价格生效日 | 未见 |
| `adjustment_factor` / `provider_event_id` | 复算与供应方事件对齐 | 未见 |
| `survivorship_policy` | 明确当前池偏差及处理方式 | manifest 有说明；未形成可执行策略 |
| `missing_reason` | 停牌、退市、请求失败、非交易日等 | 清洗仅有排除计数，缺少逐行原因字段 |

## 可检测信号（设计，不是已执行结果）

1. **同一历史窗口的未来重下载对比**：在后续日期用同一请求重新下载并比较重叠日期。本次资料研究没有执行，不能声称腾讯历史 qfq 价格会变化。
2. **原始价/前复权价比值检查**：若可取得原始价，比值应主要在公司行动生效日附近变化；异常跳变应按 `event_type`、公告日和生效日解释。
3. **固定池生命周期检查**：检查固定池中是否存在已知停牌、退市、并购或代码变更样本；若全部是当前仍活跃证券，应标记潜在生存者偏差，而不是推断偏差大小。
4. **事件连续性检查**：公司行动公告日、除权日、复权因子和价格比值应形成可解释序列；缺失或不连续应进入 `missing_reason`。
5. **股票池时间戳检查**：每个历史日期应能回答“当时依据什么名单、规则和可知信息入选”，而不是只引用当前 `universe` 数组。
6. **数据重放检查**：固定 `data_version`、`price_adjustment` 和 `corporate_action_asof` 后重新计算派生面板，结果应可重复；本轮未完成此测试。

## 本轮明确未验证

- 未运行后端测试、前端 typecheck/build、worker/API smoke 或最终测试。
- 未验证本项目是否真正实现 PIT 股票池、公司行动表或原始价重放。
- 未执行腾讯 qfq 历史窗口重复下载测试；不声称历史 qfq 已被未来事件改写。
- 未取得真实 LLM provider 凭据，未生成或评价任何真实 LLM 因子。
- 未复现 S3、S4、S5 的任何实验结果；其中数字均属于原作者。
- 未读取 S1 正文、S2 数据库指南或 Brown 等正文；相应结论仅限公开摘要/元数据/产品页。
- 未把经典 baseline、模拟结果或二手解读写成真实 LLM 生成或真实回测净值。

## 新增来源数

**5 个核心来源**：S1 Shumway & Warther、S2 CRSP 官方产品页、S3 Ranse arXiv 预印本、S4 Fonseca 正式期刊论文、S5 Wei Meng 预印本。另记录 1 个仅用于书目锚点且访问受限的 Brown 等（1992），不计入核心结论来源。

## 未决项与 pending decision

- **blocker**：当前没有历史时点股票池、上市/退市日期和公司行动事件表；固定当前 30 只股票无法代表历史可投资集合。
- **blocker**：腾讯 qfq 快照不提供事件级复权因子与 PIT 公司行动；单一 `adj_close` 不能独立完成价格前视审计。
- **blocker**：S1 正文、CRSP 指南、Brown 等正文不可公开访问；相关结论只能停留在摘要/元数据层。
- **blocker**：S3/S5 为预印本，S4 的业绩与残差结果未在本项目复现。
- **pending decision**：是否引入有许可或公开的历史成分股、上市/退市与公司行动数据源，以及如何记录其许可和数据版本；不阻塞本轮资料研究。
- **pending decision**：是否允许后续执行“同一历史窗口重复下载对比”与原始价/复权价字段核验；本轮未执行，避免越过资料研究边界。

## 下一轮唯一优先检索方向

核验**腾讯 `qfqday` 历史价格回写机制**与 A 股官方**除权除息、停牌、退市事件字段**：找到一个可公开访问的一手来源，回答历史前复权价格是否会在后续公司行动后变化、A 股事件数据至少应保存哪些公告日/生效日字段，并形成下一次最小可执行的数据审计方案。此方向只做资料来源核验，不修改策略或运行最终测试。
