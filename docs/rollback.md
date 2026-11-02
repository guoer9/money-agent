# 当前代码与数据的版本回退

本次检查点保存于用户指定的 `guoer9/money-agent` 仓库。代码使用 Git 提交、独立备份分支和附注标签；数据、数据库与已构建前端通过 SSH 保存到专用数据分支。归档每片不超过 32 MiB，附带分片及完整归档 SHA-256、合并脚本和逐文件清单。恢复时共同核验代码和数据身份。

## 2026-11-02 检查点

| 项目 | 标识 |
| --- | --- |
| 当前产品版本 | AlphaLab v1.2.0 及 2026-11-02 工作区增量 |
| 当前回退标签 | `alphalab-v1.2.0-data-checkpoint-20261102.1` |
| GitHub 备份分支 | `backup/alphalab-20261102` |
| GitHub 数据分支 | `data/alphalab-20261102` |
| 原 v1.2.0 基点 | `90dbda7`，另以 `alphalab-v1.2.0` 标签保存 |
| 初始代码检查点 | `alphalab-v1.2.0-data-checkpoint-20261102`，保留原标签 |
| 数据地址 | `https://github.com/guoer9/money-agent/tree/data/alphalab-20261102` |
| 产品数据集 | `tencent-qfq-20261101T143908-d1ef67035f`，30 只、19,190 行 |
| 产品面板 SHA-256 | `1ce0f66b3541711961b8cc08abd94d24c434265c4eda94a1c979b99c87b82623` |
| 全市场验证 | 当前 A 股 5,576 只、11,393,279 行窗口日线，独立保存、尚非研究就绪 |

备份分支保留当前产品独立历史，远端已有 `main`、`finbert`、`vllm` 和 `v1.0.0` 保留。此次检查点不代表新的稳定功能版本或新的论文最终测试成绩。

## 归档与范围

| 合并后文件 | 保存内容 |
| --- | --- |
| `AlphaLab-checkpoint-20261102-source.tar.gz` | 该标签的完整源码、文档、前端锁文件和已纳入版本的原始基线数据 |
| `AlphaLab-checkpoint-20261102-code.bundle` | 产品 Git 历史与标签，供离线恢复 |
| `AlphaLab-checkpoint-20261102-runtime.tar.gz` | 现有数据、基线结果、全市场验证原文/Parquet/逐证券记录/质量报告、SQLite 一致性备份、已构建前端、实际依赖版本 |
| `AlphaLab-checkpoint-20261102-research.tar.gz` | 工作区调研笔记、来源锁定文件、AKShare 固定源码快照及许可、研究调度器源码与状态 |
| `checkpoint-manifest.json` | 提交与数据身份、附件 SHA-256、数量、排除范围和恢复核验结果 |
| `SHA256SUMS.txt` | 上述附件的 SHA-256 校验和 |

运行数据和调研附件各有 `FILE-MANIFEST-*.json`，记录每个文件的相对路径、大小和 SHA-256。运行数据中的数据库通过 SQLite `backup()` 取得，并通过完整性检查；不打包运行锁、WAL/SHM 或机器凭据。依赖环境以版本清单保存，虚拟环境和 `node_modules` 按依赖重新建立。

外部开源仓库缓存由原有来源锁文件记录，不重复放入附件；论文 PDF、个人任务书和工作区外临时材料不上传。部分研究笔记引用 `/tmp` 文件，已有“临时证据”标识；本检查点不将这些引用改写为永久归档证据。来源文件的公开可访问性不代表取得新的再分发授权。

## 在新目录恢复，保留当前运行实例

以下步骤在一个新的空目录中操作。数据分支的脚本先验证分片和完整归档，再写到新建的 `packages` 目录；文件已存在时停止，不覆盖已有文件。

```sh
mkdir AlphaLab-restore-20261102
cd AlphaLab-restore-20261102
git clone --single-branch --branch alphalab-v1.2.0-data-checkpoint-20261102.1 \
  git@github.com:guoer9/money-agent.git rdagent_thesis
git clone --depth 1 --single-branch --branch data/alphalab-20261102 \
  git@github.com:guoer9/money-agent.git data-backup
python3 data-backup/restore_archives.py packages
cd packages
shasum -a 256 -c SHA256SUMS.txt
tar -xzf AlphaLab-checkpoint-20261102-runtime.tar.gz -C ..
tar -xzf AlphaLab-checkpoint-20261102-research.tar.gz -C ..
cd ..
```

核验逐文件清单：

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
for manifest in ('FILE-MANIFEST-runtime.json', 'FILE-MANIFEST-research.json'):
    rows = json.loads(Path(manifest).read_text())['files']
    for row in rows:
        path = Path(row['path'])
        assert path.stat().st_size == row['bytes'], path
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == row['sha256'], path
    print(manifest, len(rows), 'verified')
PY
```

之后按产品 README 创建 Python 环境。附件已含构建产物，可先离线展示；修改前端时，再根据 `frontend/pnpm-lock.yaml` 执行锁定依赖安装和构建。实际环境清单位于 `rdagent_thesis/runtime/checkpoints/20261102/`；全市场 probe 的隔离依赖与产品依赖分别记录。

```sh
cd rdagent_thesis
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# 用另一个端口检查恢复副本，确认后再安排正式实例切换。
PORT=18001 ./scripts/start.sh
```

离线代码恢复可使用 `git clone AlphaLab-checkpoint-20261102-code.bundle rdagent_thesis`，再切换到检查点标签并解压相应运行附件。数据库核验或服务启动应使用项目依赖环境；本机系统 Python 的 SQLite 打开该外接卷文件失败，而项目 `.venv` 的 SQLite 已验证可读。

## 后续数据层变更的版本规则

- 每次数据层结构或来源改变前，保存新的代码检查点和对应数据附件；已有标签与附件保留。
- 新行情快照使用新数据集 ID，记录来源、价格/成交量口径、原始响应指纹和适配器版本；已有实验继续引用原数据集。
- 数据库结构改变时，迁移前保留一致性备份，先在恢复副本演练；验证代码、数据库与快照三者匹配后再切换运行实例。
- 新快照未完整通过质量门槛时，页面继续显示最近一次合格真实快照，并单独显示采集失败或待验收状态。
- 后续提交沿开发分支推进；回退优先从已验证标签创建新工作目录，避免在正在运行的目录执行覆盖式恢复。
