# RD-Agent 接线状态

## v1.2 状态

v1.2 保留了真实接线的服务端能力边界，但没有把 RD-Agent 源码或模型调用伪装成已完成。当前 `thesis.config.load_rd_adapter()` 会尝试导入 `thesis.rd_adapter`；仓库未提供可用适配器时，环境接口返回 `rdagent_available=false`，LLM 任务返回 `503`，不会降级为经典 baseline。

真实接线必须实现以下最小契约：

```python
def availability() -> bool | dict: ...
def run_research(request: dict, root: Path, emit) -> dict: ...
```

`run_research` 必须：

1. 从服务端配置读取模型，不把凭据写入事件、结果或导出；
2. 只读取训练/验证协议允许的数据，不能读取最终测试标签；
3. 记录模型响应、调用用量、父子因子和验证集指标；
4. 返回可序列化结果，并让 worker 以 `complete/failed` 方式持久化；
5. 在候选冻结后另行触发最终测试，不得把测试结果放入反馈上下文。

完成上游固定版本导入、适配器单元测试和真实模型调用后，才可把“待验证”改成“可用”，并补充论文中的反馈消融和调用成本证据。
