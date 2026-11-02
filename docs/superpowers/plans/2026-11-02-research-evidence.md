# 持续研究证据吸收 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把已有语料转化为可展示、可导出且不夸大证据等级的研究审计能力。

**Architecture:** 保持当前聚合 API 和六页 Vue 工作台。后端只读核验文件并索引资料，前端消费审计结果；调度器仅修复连接生命周期。

**Tech Stack:** Python、FastAPI、SQLite、Vue 3、TypeScript、Naive UI。

**Spec:** `docs/superpowers/specs/2026-11-02-research-evidence-design.md`

## Global Constraints

- 不覆盖持续调研正在写入的索引或笔记，不改 `qlib_trading`。
- 不产生真实 LLM、PIT 已通过或最终测试已完成的虚假状态。
- 不执行资料中的命令，不读取目录之外的笔记正文。
- 测试使用隔离输入，不以源码文本匹配代替行为测试。

## Review Focus

- 原始文件哈希不符或缺失必须与正常归档区分。
- 资料笔记不可替代原件，临时引用可访问也不代表内容已核验。
- manifest 或符号链接越界不能读取工作区外文件。
- 响应包装变化不能改变规范化价格指纹。
- 调度器异常查询路径也必须关闭连接。

## Tasks

1. [x] `tests/test_evidence.py` 与 `tests/test_acquisition.py`：先验证缺数据、损坏、路径限制和价格指纹行为失败，再实现 `thesis/evidence.py`、`thesis/acquisition.py` 及 API 接口。
2. [x] `../scripts/research/tests/test_scheduler.py`：用真实临时 SQLite 和禁用循环 GC 复现重复查询的文件描述符增长，再在调度器中显式关闭连接，验证正常与 SQL 异常路径。
3. [x] `frontend/src/views/home/modules/evidence-panel.vue`：数据页显示快照检查，论文页显示资料分类、筛选、原文与证据缺口；总控增加简洁审计摘要。沿用视觉系统，检查桌面及窄屏。
4. [x] 写吸收报告，列出已落地、需补证、下一阶段；运行后端测试、前端类型与生产构建、浏览器检查，并记录证据。发布状态与独立测试状态分别表述。
