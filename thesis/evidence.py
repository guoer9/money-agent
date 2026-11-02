"""Read-only integrity checks and a corpus index. Documents are never executed.

Hash agreement proves local byte integrity only, never PIT or experiment validity.
No factors, labels or evaluation metrics are calculated by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from thesis.config import utc_now


TEMP_REFERENCE = re.compile(r"/(?:private/)?tmp/[\w./%:@+-]+\.(?:jsonl?|pdf|txt|docx|csv|js|html)\b", re.IGNORECASE)
NOTE_NAME = re.compile(r"^(\d+)-[\w-]+\.md$")


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _inside(base: Path, relative: str) -> Path | None:
    try:
        candidate = (base / relative).resolve()
        return candidate if candidate.is_relative_to(base.resolve()) else None
    except (OSError, ValueError, RuntimeError):
        return None


def _file_check(base: Path, relative, expected=None) -> dict:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return {"status": "invalid_path"}
    path = _inside(base, relative)
    if path is None:
        return {"status": "invalid_path"}
    try:
        actual = _sha256(path)
    except FileNotFoundError:
        return {"status": "missing"}
    except OSError:
        return {"status": "unreadable"}
    status = "verified" if expected and expected == actual else "mismatch" if expected else "unrecorded"
    return {"status": status, "expected_sha256": expected, "actual_sha256": actual}


def _dataset(root: Path) -> dict:
    value = {"status": "unavailable", "pit_status": "unverified", "panel": {"status": "missing"},
             "raw": {"total": 0, "verified": 0, "missing": 0, "mismatch": 0,
                     "unreadable": 0, "invalid_path": 0, "unrecorded": 0},
             "normalized_fingerprints": 0, "snapshot_time": None, "manifest_sha256": None, "id": None}
    data = root / "runtime/data"
    try:
        path = data / "manifest.json"
        if _inside(root, "runtime/data/manifest.json") is None:
            value["status"] = "unreadable"
            return value
        content = path.read_bytes()
        manifest = json.loads(content)
        if (not isinstance(manifest, dict) or not isinstance(manifest.get("raw_responses", []), list)
                or any(not isinstance(r, dict) for r in manifest.get("raw_responses", []))):
            raise ValueError("invalid manifest")
    except FileNotFoundError:
        return value
    except (OSError, ValueError, TypeError):
        value["status"] = "unreadable"
        return value
    value["manifest_sha256"] = hashlib.sha256(content).hexdigest()
    value["id"] = manifest.get("id")
    value["snapshot_time"] = manifest.get("downloaded_at")
    value["panel"] = _file_check(data, "panel.parquet", manifest.get("sha256"))
    for record in manifest.get("raw_responses", []):
        check = _file_check(data, record.get("path"), record.get("sha256"))
        value["raw"]["total"] += 1
        value["raw"][check["status"]] += 1
        if record.get("normalized_prices_sha256"):
            value["normalized_fingerprints"] += 1
    total = value["raw"]["total"]
    value["status"] = ("verified" if value["panel"]["status"] == "verified" and total > 0
                       and total == value["raw"]["verified"] else "incomplete")
    return value


def _notes(root: Path):
    base = root / "docs/research"
    if _inside(root, "docs/research") is None:
        return
    for path in sorted(base.glob("????-??-??/*.md"), reverse=True):
        match = NOTE_NAME.fullmatch(path.name)
        if not match or int(match[1]) < 2:
            continue
        # Symlinked files/directories must stay inside the research directory.
        if _inside(base, str(path.relative_to(base))) is None:
            continue
        yield path, int(match[1])


def read_note(root: Path, note_id: str) -> str | None:
    for path, _ in _notes(root):
        if note_id == f"{path.parent.name}--{path.stem}":
            try:
                return path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                return None
    return None


def _temporary_refs(text: str) -> set[str]:
    refs = set()
    for value in TEMP_REFERENCE.findall(text):
        try:
            refs.add(str(Path(value).resolve()))
        except (OSError, ValueError, RuntimeError):
            refs.add(value)  # Keep the failed reference in the missing-source count.
    return refs


def _accessible(path: str) -> bool:
    try:
        return Path(path).is_file()
    except (OSError, ValueError, RuntimeError):
        return False


def _corpus(root: Path) -> dict:
    notes = []
    all_refs = set()
    for path, round_ in _notes(root):
        try:
            content = path.read_bytes()
            text = content.decode("utf-8")
        except (OSError, UnicodeError):
            continue
        title = re.search(r"^#\s+([^\n]+)", text, re.MULTILINE)
        refs = _temporary_refs(text)
        all_refs.update(refs)
        available = sum(_accessible(p) for p in refs)
        category = "数据可信度"
        if "leakage" in path.name or "llm" in path.name:
            category = "生成与评价"
        elif any(word in path.name for word in ("rights", "chinaclear", "refund", "settlement")):
            category = "事件与制度"
        notes.append({"id": f"{path.parent.name}--{path.stem}", "round": round_, "date": path.parent.name,
                      "title": title[1] if title else path.stem, "category": category,
                      "path": str(path.relative_to(root)), "sha256": hashlib.sha256(content).hexdigest(),
                      "evidence_level": "note_only", "temporary_references": {
                          "referenced": len(refs), "available": available, "missing": len(refs) - available}})
    available = sum(_accessible(p) for p in all_refs)
    return {"note_count": len(notes), "notes": notes, "temporary_files": {
        "referenced": len(all_refs), "available": available, "missing": len(all_refs) - available}}


def evidence_snapshot(root: Path, archive: dict | None = None) -> dict:
    root = Path(root).resolve()
    dataset = _dataset(root)
    displayed = archive.get("dataset", {}) if archive else {}
    link = {"status": "unavailable", "displayed_dataset_id": displayed.get("id"),
            "checked_dataset_id": dataset["id"], "displayed_sha256": displayed.get("sha256"),
            "checked_sha256": dataset["panel"].get("actual_sha256")}
    if all(link[key] for key in ("displayed_dataset_id", "checked_dataset_id", "displayed_sha256", "checked_sha256")):
        link["status"] = ("matched" if link["displayed_dataset_id"] == link["checked_dataset_id"]
                          and link["displayed_sha256"] == link["checked_sha256"] else "mismatch")
    return {"schema_version": "1.0", "checked_at": utc_now(), "dataset": dataset, "archive_link": link,
            "corpus": _corpus(root),
            "boundaries": [
                {"title": "前复权快照", "status": "未完成 PIT 验证", "detail": "文件指纹一致只证明本地归档完整；供应方的逐事件因子和历史回写政策尚未核验。"},
                {"title": "固定股票池", "status": "存在选择偏差", "detail": "当前行情采用固定研究池，尚无历史时点成分、上市与退市覆盖证明。"},
                {"title": "文献与实现", "status": "分开记录", "detail": "研究笔记、制度条款和样本观察分别保留；资料数量不代表代码能力或实验已通过。"},
                {"title": "最终测试", "status": "预留未执行", "detail": "当前仅有经典因子验证集评价；尚未冻结候选，也未执行真实 LLM 反馈和独立测试。"},
            ]}
