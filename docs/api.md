# AlphaLab v1.2 本地 API

API 是本地只读展示和持久任务入口。HTTP 请求只把研究请求写入 SQLite，不在请求进程中执行因子表达式、baseline 或模型调用。默认 HTTP 服务监听 `127.0.0.1`。

## 启动

在项目根目录分别启动 worker 和 API：

```bash
.venv/bin/python -m thesis.worker
.venv/bin/python -m thesis.api --port 8000
```

API 默认使用当前项目根目录，也可以设置 `THESIS_ROOT`。测试和嵌入式使用通过 `thesis.api.create_app(root: Path | None)` 创建应用。worker 应与 API 使用同一个根目录。

研究任务状态保存在 `runtime/research.sqlite3`，worker 锁保存在 `runtime/worker.lock`，心跳每秒写入数据库。worker 启动时会把上一次遗留的 `running` 任务标为 `failed`，添加 `interrupted` 事件，并且不会自动重新调用模型；需要重新提交新的幂等键。

## 配置

LLM 配置从 `thesis.config.load_config()` 读取，首选 `THESIS_*`，并兼容 `OPENAI_*`：

| 用途 | 首选变量 | 兼容变量 |
| --- | --- | --- |
| API key | `THESIS_LLM_API_KEY` | `OPENAI_API_KEY` |
| Base URL | `THESIS_LLM_BASE_URL` | `OPENAI_BASE_URL`、`OPENAI_API_BASE` |
| 模型名 | `THESIS_LLM_MODEL` | `OPENAI_MODEL` |

只有 API key 和模型名同时存在时 `llm_configured` 才为 `true`。环境接口只返回模型名和布尔状态，key 不会出现在响应、事件、错误或导出中。LLM 任务还要求真实的 `thesis.rd_adapter.run_research(request, root, emit)` 可导入；适配器不存在或不可用时返回 `503`，不会以 baseline 冒充模型研究。

## 响应约定

所有响应都是 plain JSON，接口不使用上游演示的 `code/data` 包装。数字必须是有限 JSON 数字；非有限值会转换为 `null`。

### `GET /api/overview`

返回：

```json
{
  "archive": null,
  "runs": [],
  "environment": {
    "llm_configured": false,
    "rdagent_available": false,
    "worker_running": false,
    "model": null
  },
  "final_test": {"status": "not_run", "message": "..."},
  "server_time": "2026-11-01T00:00:00.000000Z",
  "version": "1.2.0"
}
```

`archive` 读取 `runtime/artifacts/baselines.json`；文件不存在时为 `null`，文件损坏或不符合归档契约时返回 `503`。

另返回 `evidence`，与下述审计端点结构相同。

### `GET /api/evidence` 与 `GET /api/evidence/notes/{note_id}`

只读核验 manifest、面板和原始响应 SHA-256，不计算因子或标签。`dataset.status=verified` 仅指文件完整；`archive_link.status=matched` 才表示当前检查的数据与展示评价属于同一 ID/哈希。缺文件、哈希不符、缺原指纹、路径无效和清单不可读分别记录；`pit_status` 仍为 `unverified`。

`corpus` 枚举 `docs/research/YYYY-MM-DD/` 第 2 轮起的实质 Markdown 记录，返回标题、分类、路径、SHA 和字面临时引用的可访问数。笔记标为 `note_only`，文件存在不代表原件已核验。正文只允许读取索引范围内文件，以纯文本返回，不执行 Markdown、HTML 或命令；未知或越界 ID 返回 `404`。

### `GET /api/datasets` 与 `GET /api/factors`

归档存在时分别返回 `archive.dataset` 的单元素列表和 `archive.factors`，无归档时返回空列表。

### `GET /api/runs` 与 `GET /api/runs/{id}`

返回持久化任务及事件。任务字段包括 `id`、`prompt`、`mode`、`iterations`、`status`、创建/更新时间、`error`、`result` 和 `events`。状态为 `queued`、`running`、`complete` 或 `failed`。不存在的 ID 返回 `404`。

### `POST /api/runs`

请求必须携带 `Idempotency-Key` 请求头，JSON 请求体为：

```json
{"prompt": "复算验证集经典因子", "mode": "baseline", "iterations": 1}
```

`mode` 只能是 `baseline` 或 `llm`，`iterations` 只能是整数 `1` 或 `2`。成功入队返回 `202`。同一 key 和完全相同请求会返回同一个任务 ID；同一 key 用于不同请求返回 `409`。

以下情况不会创建任务：

- 缺少幂等键：`400`。
- 请求体不合法：`422`。
- 没有活跃 worker：`503`。
- LLM 未同时配置 key 和模型名：`503`。
- LLM 的真实 RD-Agent 适配器不可用：`503`。

baseline 请求可以在 worker 在线时入队；实际执行由独立 worker 完成。

### `GET /api/reports/export`

导出真实归档、持久任务、公开环境配置、版本、最终测试状态和 `evidence` 审计快照，返回 `Content-Disposition: attachment`。不包含 API key、原始测试标签或其他原始敏感字段。无归档也返回严格 JSON，`archive` 为 `null`；可以单独导出已有资料索引。

## Worker 执行

worker 通过 `runtime/worker.lock` 保证同一根目录只有一个活动进程，通过 SQLite 原子 claim 保证同一时刻只有一个 `running` 任务。baseline 调用 `thesis.baselines.build_baselines(root)`；LLM 调用真实 `thesis.rd_adapter.run_research(...)`。执行过程写入 queued、running、阶段事件和 complete/failed 结果。

worker 只用于本地启动，主 agent 负责启动 API 和 worker；API 本身不启动 worker。
