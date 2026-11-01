# 当前状态

更新时间：2026-11-02

## Verified facts

- 持续调度已切换为每 30 分钟一次、持续 24 小时；launchd timer 的 `StartInterval=1800`，state 中 `interval_seconds=1800`、时长为 24.0 小时。
- 启动轮消息已进入目标线程队列，内容为网络资料收集与证据记录，不再默认执行 v1.2 验收。

- 真实面板与归档已存在，并通过 API 进程内 overview 读取；归档含 19,190 行、30 只股票、6 个 validation baseline。
- API archive 的数值、字段和协议可被读取；最后测试仍为 `reserved_not_run`。
- 历史测试/构建日志存在，但尚未在本轮重新执行。
- 当前研究模式为每轮一个窄问题的网络资料研究；来源质量、可访问性和结论适用性必须逐条记录。
- 调度切换证据：`docs/research/2026-11-02/01-scheduler-30m-24h-web-research.md`。
- 已核验 5 个相互独立来源：Alpha-GPT 原始论文、scikit-learn 数据泄漏与 TimeSeriesSplit 官方文档、Bailey 等回测过拟合论文的公开摘要/元数据、EvoQuantLab 预印本。
- 已形成首版防泄漏检查表，明确区分“来源支持的事实”“本项目推断”和“需代码核验”。证据：`docs/research/2026-11-02/02-llm-factor-leakage-source-baseline.md`。
- 未配置真实 LLM 凭据；本轮没有生成真实 LLM 因子，也没有运行后端测试、前端构建或 worker/API smoke。

## Pending verification

- 当前后端测试是否仍为 75 passed。
- 当前前端 typecheck/build 是否仍通过。
- 独立 worker 与 API 能否用真实归档完成一次提交、claim、执行和完成。
- 启动/停止流程、端口、PID 与刷新恢复是否可重复。

## Pending decisions

- 真实 LLM provider、模型和凭据尚未配置；不阻塞 baseline 主干验证。
- 最终测试集何时冻结并独立执行，需在候选与评价协议确认后再决定。
- 当前仓库无 remote；发布方式是本地 tag + 可运行归档，不能宣称公网部署。

## Blockers

- 真实 LLM provider、模型与凭据仍未配置；不能宣称完成真实 LLM 生成或端到端回测。
- v1.2 验收基线复核尚未完成，后续如需恢复工程验收，应单独安排新鲜测试和 worker/API smoke，不得引用历史日志。
- Bailey 等 2014 正文本次访问返回 403，只使用了公开摘要/元数据；不能引用未读正文的具体公式或阈值。
- 首版检查表中的“提示词/候选账本、预处理拟合边界、最终测试冻结入口”尚无代码级通过证据。

## Next step

整理 point-in-time 金融数据、公司行动与生存者偏差的一手资料：回答腾讯前复权快照和固定股票池的具体偏差、可检测信号与最小审计字段，并写入下一份来源笔记。
