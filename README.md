# AlphaLab 数据层改造前备份 — 2026-11-02

本分支专门保存数据恢复归档；代码位于同仓库 `backup/alphalab-20261102` 分支。

- 代码提交：`385d47c185c9a0a5137f23ec3c6463b11105e5b7`
- 代码标签：`alphalab-v1.2.0-data-checkpoint-20261102.1`
- 数据分支：`data/alphalab-20261102`
- 对应数据标签：`alphalab-data-20261102`

保存内容：产品源码与离线 Git 历史、30 只 / 19,190 行原始基线、当前 A 股 5,576 只 / 11,393,279 行窗口日线的全部原文与规范化文件、逐证券记录与质量审计、复权试点、SQLite 一致性备份、已构建 Vue 3 前端、实际依赖版本和调研来源锁文件。

运行数据 17,178 个文件、调研资料 36 个文件已解压并逐文件核验 SHA-256，数据库完整性检查通过。恢复副本的总览 API、证据 API、页面和静态资源均通过，数据身份匹配。全市场原始行情仍 `research_ready=false`，最终独立测试未执行。

## 取回与校验

```sh
git clone --depth 1 --single-branch --branch data/alphalab-20261102 git@github.com:guoer9/money-agent.git alphalab-data-backup
python3 alphalab-data-backup/restore_archives.py packages
cd packages
shasum -a 256 -c SHA256SUMS.txt
```

`packages` 必须是尚不存在的新目录。脚本使用 Python 3.10+，只读本地分片、不访问网络，逐片及整体校验后写出归档。归档按 32 MiB 分片，索引为 `archive-index.json`，完整归档和数据身份见 `checkpoint-manifest.json`。

`source.tar.gz`、`runtime.tar.gz` 和 `research.tar.gz` 可在同一个新建目录解压，源码和运行数据都以 `rdagent_thesis/` 为目录前缀。详细恢复步骤在代码分支的 `docs/rollback.md`。先演练新目录，再安排运行实例切换。

此分支通过 Git SSH 保存，**不依赖尚未上传成功的 Release 附件**。原远端 main、finbert、vllm 和 v1.0.0 保留；原 AlphaLab v1.2.0 与初始检查点标签也保留。凭据、虚拟环境、node_modules、论文 PDF、个人任务书、第三方 Git 克隆缓存与工作区外临时材料不在归档内。
