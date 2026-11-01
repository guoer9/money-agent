"""Small release-contract checks for the public AlphaLab page."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _object_keys(source: str, name: str) -> set[str]:
    match = re.search(rf"const {name}: Record<string, string> = \{{(?P<body>.*?)\n\}};", source, re.S)
    assert match, f"{name} mapping is missing"
    return set(re.findall(r"\b([A-Za-z0-9_]+):", match.group("body")))


def test_alpha_lab_replaces_upstream_home_title():
    locale = (ROOT / "frontend/src/locales/langs/zh-cn.ts").read_text(encoding="utf-8")
    assert "title: 'AlphaLab'" in locale
    assert re.search(r"home:\s*'AlphaLab'", locale)


def test_real_archive_fields_have_chinese_labels():
    archive = json.loads((ROOT / "runtime/artifacts/baselines.json").read_text(encoding="utf-8"))
    quality_keys = set(archive["dataset"]["quality"])
    protocol_keys = set(archive["protocol"])
    source = (ROOT / "frontend/src/views/home/modules/detail-panels.vue").read_text(encoding="utf-8")
    assert quality_keys <= _object_keys(source, "qualityNames")
    assert protocol_keys <= _object_keys(source, "protocolNames")


def test_local_release_has_reproducible_entrypoints():
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "1.2.0"
    for name in ("run-api.sh", "run-worker.sh", "start.sh"):
        script = ROOT / "scripts" / name
        assert script.is_file() and script.stat().st_mode & 0o111
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "AlphaLab v1.2.0" in readme
