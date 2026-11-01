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

# 重新获取行情（会替换当前归档）
.venv/bin/python -m thesis.acquisition --output runtime/data --start 2024-01-01 --end 2026-10-30
.venv/bin/python -m thesis.baselines --root .
```

当前归档数据已随本地 v1.2 发布包保留；数据源、请求批次、局限和评价公式见 `docs/data-evaluation.md`，API 见 `docs/api.md`。

## 论文使用注意

可以把本版本作为“真实数据接入、经典基线正确评价、可追溯展示系统”的实验基线。不要把当前的验证集结果写成 LLM 反馈增益或最终测试成绩，也不要把分层统计收益解释为含费用、滑点和可交易性约束的策略回测。固定股票池存在幸存者偏差；腾讯前复权是下载时点快照；这些限制已在数据页和论文材料页展示。

## 许可证

前端底座保留 SoybeanAdmin 的 MIT 许可证与上游信息，见 `frontend/LICENSE` 和 `frontend/UPSTREAM.json`。
