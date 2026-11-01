# 2026-11-02 语料收集调度切换（30 分钟 × 24 小时）

## 目的

把持续输出循环从“v1.2 工程验收续跑”切换为“网络资料采集与优化语料积累”。本轮只调整研究调度与说明，不改变交易系统实现，不运行最终测试。

## 变更

- 调度器：`/Volumes/2tb/mydata/code/Quantitative_trading/scripts/research/quant_research_scheduler.py`
- 周期：`1800` 秒（30 分钟）
- 持续时长：`86400` 秒（24 小时）
- 启动时间：`2026-11-02 00:46:54 CST`
- 截止时间：`2026-11-03 00:46:54 CST`
- 目标线程：`01a197f4-6269-78a3-a363-c9886fd2d833`
- launchd daemon：`com.guoer.quantitative-trading.research`
- launchd timer：`com.guoer.quantitative-trading.research.tick`
- caffeinate PID：`2472`

## 每轮默认范围

每轮选择一个窄问题，以网络搜索为主，优先检索论文原文/arXiv、官方文档、开源仓库、监管或交易所资料。主题池包括：

- 数据清洗与点时数据
- 因子生成与代码校验
- IC/Rank IC/IR 与分层回测
- 防泄漏与可复现实验
- LLM/Agent 工作流
- 经典因子库与交易成本

每份笔记必须给出来源标题、URL、访问日期、来源类型、原文依据、适用性、限制和下一轮唯一检索方向。事实、作者观点与本项目推断必须分开。baseline、经典因子、模拟结果不得写成真实 LLM 生成或真实回测。

## 校验命令与结果

```bash
python3 -m py_compile scripts/research/quant_research_scheduler.py
plutil -lint runtime/research-scheduler/com.guoer.quantitative-trading.research.plist runtime/research-scheduler/com.guoer.quantitative-trading.research.tick.plist
scripts/research/quant_research_scheduler.py status
sqlite3 -readonly "$HOME/.codex/queue_1.sqlite" \
  "select payload_json from queued_items where thread_id='01a197f4-6269-78a3-a363-c9886fd2d833' order by rowid desc limit 1;"
```

结果：

- Python 编译通过。
- 两个 plist 均为 `OK`。
- state 显示 `interval_seconds=1800`、`duration_hours=24.0`、`status=running`。
- launchd timer 显示 `run interval = 1800 seconds`。
- 首轮消息已排队，消息 ID `01a19801-c39a-7792-a439-987ca2c96258`，内容已核对为新版语料收集提示词。

## 限制与未决项

- 调度器只在目标线程空闲且队列无待处理消息时入队；一轮超时或线程忙时会跳过该次，不并发强推。
- 网络来源质量与可访问性每轮可能变化；需要逐条记录访问日期和限制。
- 当前没有真实 LLM provider 凭据。资料研究可继续，但不能宣称完成真实 LLM 因子生成或端到端回测。
- 2026-11-02 首轮 v1.2 验收基线复核仍处于未完成状态；本次没有用历史日志替代新鲜测试。

## 下一轮唯一优先方向

检索 LLM 量化因子系统中的**防泄漏与可复现实验检查项**，形成可映射到本项目数据、因子代码和评价协议的第一版来源清单。
