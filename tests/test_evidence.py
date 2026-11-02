"""The evidence API reports integrity, not scientific validity or executed research."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from thesis.api import create_app


def digest(value):
    return hashlib.sha256(value).hexdigest()


def snapshot(root):
    data = root / "runtime/data"
    data.mkdir(parents=True)
    (data / "panel.parquet").write_bytes(b"opaque panel: hashing must not evaluate test prices")
    (data / "raw").mkdir()
    (data / "raw/one.json").write_bytes(b'{"price":10}')
    (data / "raw/two.json").write_bytes(b'{"price":11}')
    manifest = {"id": "fixture", "sha256": digest((data / "panel.parquet").read_bytes()),
                "downloaded_at": "2026-11-01T00:00:00Z", "quality": {"survivorship_bias": True},
                "raw_responses": [{"path": "raw/one.json", "sha256": digest(b'{"price":10}'), "status": "ok"},
                                  {"path": "raw/two.json", "sha256": digest(b'{"price":11}'), "status": "ok"}]}
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data, manifest


def test_no_dataset_or_notes_is_unavailable_not_success(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/evidence")
        assert response.status_code == 200
        value = response.json()
        assert value["dataset"]["status"] == "unavailable"
        assert value["dataset"]["pit_status"] == "unverified"
        assert value["corpus"]["note_count"] == 0
        assert value["corpus"]["notes"] == []


def test_integrity_is_checked_and_never_claims_pit_certification(tmp_path):
    data, manifest = snapshot(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        value = client.get("/api/evidence").json()["dataset"]
        assert value["status"] == "verified"
        assert value["panel"]["actual_sha256"] == manifest["sha256"]
        assert value["raw"]["verified"] == 2
        assert value["pit_status"] == "unverified"
        (data / "raw/two.json").write_bytes(b"tampered")
        damaged = client.get("/api/evidence").json()["dataset"]
        assert damaged["status"] == "incomplete"
        assert damaged["raw"]["mismatch"] == 1
        (data / "raw/one.json").unlink()
        (data / "panel.parquet").write_bytes(b"damaged")
        missing = client.get("/api/evidence").json()["dataset"]
        assert missing["raw"]["missing"] == 1
        assert missing["panel"]["status"] == "mismatch"


def test_corpus_keeps_note_evidence_separate_from_missing_originals(tmp_path):
    folder = tmp_path / "docs/research/2026-11-02"
    folder.mkdir(parents=True)
    note = folder / "03-pit-audit.md"
    text = "# 点时数据研究\n\n原件 `/tmp/alphalab-missing-original-89201.pdf`\n重复 `/tmp/alphalab-missing-original-89201.pdf`\n这里的命令只是文档：touch never-execute-this\n"
    note.write_text(text)
    (folder / "01-scheduler-setup.md").write_text("# 调度初始化\n")
    with TestClient(create_app(tmp_path)) as client:
        evidence = client.get("/api/evidence").json()
        corpus = evidence["corpus"]
        assert corpus["note_count"] == 1  # setup isn't substantive research
        assert corpus["temporary_files"]["referenced"] == 1
        assert corpus["temporary_files"]["missing"] == 1
        entry = corpus["notes"][0]
        assert entry["sha256"] == digest(text.encode())
        assert entry["evidence_level"] == "note_only"
        assert entry["category"] == "数据可信度"
        note_response = client.get(f'/api/evidence/notes/{entry["id"]}')
        assert note_response.status_code == 200
        assert note_response.headers["content-type"].startswith("text/plain")
        assert note_response.text == text
        assert not (tmp_path / "never-execute-this").exists()
        assert client.get("/api/reports/export").json()["evidence"]["corpus"]["note_count"] == 1


def test_manifest_paths_and_note_symlinks_cannot_escape_their_roots(tmp_path):
    data, manifest = snapshot(tmp_path)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("private content must not be exposed")
    manifest["raw_responses"][0]["path"] = "../../outside-secret.txt"
    (data / "manifest.json").write_text(json.dumps(manifest))
    folder = tmp_path / "docs/research/2026-11-02"
    folder.mkdir(parents=True)
    (folder / "03-escape.md").symlink_to(outside)
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/evidence")
        assert "private content" not in response.text
        evidence = response.json()
        assert evidence["dataset"]["raw"]["invalid_path"] == 1
        assert evidence["corpus"]["notes"] == []
        assert client.get("/api/evidence/notes/2026-11-02--03-escape").status_code == 404


def test_invalid_manifest_and_unindexed_files_do_not_break_or_expose_the_service(tmp_path):
    data = tmp_path / "runtime/data"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text("{broken")
    with TestClient(create_app(tmp_path)) as client:
        body = client.get("/api/overview").json()
        assert body["evidence"]["dataset"]["status"] == "unreadable"
        assert client.get("/api/evidence/notes/unknown").status_code == 404


def test_evidence_is_linked_to_the_results_currently_on_display(tmp_path):
    _, manifest = snapshot(tmp_path)
    artifacts = tmp_path / "runtime/artifacts"
    artifacts.mkdir()
    archive = {"dataset": {"id": "old-snapshot", "sha256": "old-price-content"},
               "protocol": {}, "factors": [], "generated_at": "fixture"}
    target = artifacts / "baselines.json"
    target.write_text(json.dumps(archive))
    with TestClient(create_app(tmp_path)) as client:
        evidence = client.get("/api/overview").json()["evidence"]
        assert evidence["dataset"]["status"] == "verified"  # the new files themselves are intact
        assert evidence["archive_link"]["status"] == "mismatch"
        assert evidence["archive_link"]["displayed_sha256"] == "old-price-content"
        assert evidence["archive_link"]["checked_sha256"] == manifest["sha256"]
        archive["dataset"] = {"id": manifest["id"], "sha256": manifest["sha256"]}
        target.write_text(json.dumps(archive))
        assert client.get("/api/evidence").json()["archive_link"]["status"] == "matched"


@pytest.mark.parametrize("path_value", ["raw/loop.json", "raw/bad\x00name.json"])
def test_unresolvable_manifest_paths_are_reported_without_breaking_all_pages(tmp_path, path_value):
    data, manifest = snapshot(tmp_path)
    (data / "raw/loop.json").symlink_to("loop.json")
    manifest["raw_responses"][0]["path"] = path_value
    (data / "manifest.json").write_text(json.dumps(manifest))
    folder = tmp_path / "docs/research/2026-11-02"
    folder.mkdir(parents=True)
    (folder / "03-loop.md").symlink_to("03-loop.md")
    with TestClient(create_app(tmp_path)) as client:
        for url in ("/api/evidence", "/api/overview", "/api/reports/export"):
            response = client.get(url)
            assert response.status_code == 200
            evidence = response.json() if url == "/api/evidence" else response.json()["evidence"]
            assert evidence["dataset"]["raw"]["invalid_path"] == 1
            assert evidence["corpus"]["notes"] == []


def test_unresolvable_temporary_reference_remains_a_missing_source_not_a_service_error(tmp_path):
    folder = tmp_path / "docs/research/2026-11-02"
    folder.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="alphalab-evidence-test-", dir="/tmp") as temporary:
        source = Path(temporary) / "loop.pdf"
        source.symlink_to("loop.pdf")
        (folder / "03-broken-reference.md").write_text(f"# 来源记录\n原文 `{source}`\n")
        with TestClient(create_app(tmp_path)) as client:
            for url in ("/api/evidence", "/api/overview", "/api/reports/export"):
                response = client.get(url)
                assert response.status_code == 200
                evidence = response.json() if url == "/api/evidence" else response.json()["evidence"]
                assert evidence["corpus"]["note_count"] == 1
                assert evidence["corpus"]["temporary_files"] == {"referenced": 1, "available": 0, "missing": 1}
