"""API, durable queue and independent worker contract tests (no external services)."""

import importlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_llm_environment(monkeypatch):
    for name in ("THESIS_ROOT", "THESIS_LLM_API_KEY", "THESIS_LLM_BASE_URL", "THESIS_LLM_MODEL",
                 "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)


def baseline_request(**updates):
    return {"prompt": "复算验证集经典因子", "mode": "baseline", "iterations": 1, **updates}


def write_archive(root):
    archive = {
        "dataset": {"id": "fixture", "source": "test fixture", "start": "2024-01-01",
                    "end": "2024-12-31", "rows": 12000, "instruments": 20,
                    "sha256": "abc123", "adjustment": "qfq", "quality": {},
                    "splits": {"validation": ["2024-07-01", "2024-10-01"],
                               "test": ["2024-10-02", "2024-12-31"]}},
        "protocol": {"id": "t1-t2", "label": "t+1进入，t+2退出"},
        "factors": [{"id": "momentum", "name": "动量", "expression": "close",
                     "description": "测试归档", "segment": "validation", "origin": "baseline",
                     "metrics": {"ic_mean": 0.1, "rank_ic_mean": None, "ic_ir": None,
                                 "rank_ic_ir": None, "coverage": 0.9, "valid_ic_days": 20},
                     "daily": [{"date": "2024-08-01", "ic": 0.1, "rank_ic": None}],
                     "quantiles": [{"group": 1, "mean_return": 0.002}]}],
        "generated_at": "2026-11-01T00:00:00Z", "test_evaluated": False,
    }
    target = root / "runtime" / "artifacts" / "baselines.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(archive, ensure_ascii=False), encoding="utf-8")
    return archive


def test_empty_overview_and_read_routes(tmp_path, monkeypatch):
    for name in ("THESIS_LLM_API_KEY", "THESIS_LLM_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    from thesis.api import create_app

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/overview")
        assert response.status_code == 200
        body = response.json()
        assert body["archive"] is None
        assert body["runs"] == []
        assert body["version"] == "1.2.0"
        assert body["environment"]["llm_configured"] is False
        assert body["environment"]["worker_running"] is False
        assert body["environment"]["model"] is None
        assert body["final_test"]["status"] == "not_run"
        assert body["final_test"]["message"]
        assert body["server_time"]
        for path in ("datasets", "factors", "runs"):
            assert client.get(f"/api/{path}").json() == []
        assert client.get("/api/runs/missing").status_code == 404


def test_archive_routes_and_read_only_export(tmp_path):
    from thesis.api import create_app

    archive = write_archive(tmp_path)
    target = tmp_path / "runtime" / "artifacts" / "baselines.json"
    before = target.read_bytes()
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/overview").json()["archive"] == archive
        assert client.get("/api/datasets").json() == [archive["dataset"]]
        assert client.get("/api/factors").json() == archive["factors"]
        exported = client.get("/api/reports/export")
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith("application/json")
        assert "attachment" in exported.headers["content-disposition"]
        body = exported.json()
        assert body["archive"] == archive
        assert body["runs"] == []
        assert body["config"]["version"] == "1.2.0"
        assert body["final_test"]["status"] == "not_run"
    assert target.read_bytes() == before


def test_export_empty_state_is_honest(tmp_path):
    from thesis.api import create_app

    with TestClient(create_app(tmp_path)) as client:
        body = client.get("/api/reports/export").json()
        assert body["archive"] is None
        assert body["runs"] == []
        assert body["final_test"]["status"] == "not_run"


def test_export_redacts_secrets_test_labels_and_nonfinite_values(tmp_path, monkeypatch):
    from thesis.api import create_app

    monkeypatch.setenv("THESIS_LLM_API_KEY", "sk-private-fixture")
    monkeypatch.setenv("THESIS_LLM_MODEL", "fixture-model")
    archive = write_archive(tmp_path)
    archive.update({"test_labels": [123.0], "api_key": "sk-private-fixture"})
    archive["dataset"]["quality"] = {"raw_test_labels": [3.0], "notes": "sk-private-fixture"}
    archive["factors"][0]["metrics"]["ic_mean"] = float("nan")
    archive["factors"][0]["metrics"]["rank_ic_mean"] = float("inf")
    (tmp_path / "runtime" / "artifacts" / "baselines.json").write_text(json.dumps(archive))
    with TestClient(create_app(tmp_path)) as client:
        for path in ("/api/overview", "/api/reports/export", "/api/factors", "/api/datasets"):
            response = client.get(path)
            assert response.status_code == 200
            assert "sk-private-fixture" not in response.text
            assert "test_labels" not in response.text
            json.loads(response.text, parse_constant=lambda value: pytest.fail(f"non-JSON: {value}"))
        body = client.get("/api/reports/export").json()
        assert body["archive"]["factors"][0]["metrics"]["ic_mean"] is None
        assert body["archive"]["protocol"]["label"] == "t+1进入，t+2退出"


def test_corrupt_archive_is_explicit_service_error(tmp_path):
    from thesis.api import create_app

    write_archive(tmp_path)
    (tmp_path / "runtime" / "artifacts" / "baselines.json").write_text("{broken")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/overview").status_code == 503
        assert client.get("/api/reports/export").status_code == 503


def test_no_worker_rejects_new_runs_without_enqueuing(tmp_path):
    from thesis.api import create_app

    with TestClient(create_app(tmp_path)) as client:
        for mode in ("baseline", "llm"):
            response = client.post("/api/runs", json=baseline_request(mode=mode),
                                   headers={"Idempotency-Key": mode})
            assert response.status_code == 503
        assert client.get("/api/runs").json() == []


def test_http_enqueues_baseline_and_never_executes_it(tmp_path):
    from thesis.api import create_app
    from thesis.worker import Worker

    with Worker(tmp_path), TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/runs", json=baseline_request(), headers={"Idempotency-Key": "one"})
        assert response.status_code == 202
        run = response.json()
        assert run["mode"] == "baseline"
        assert run["status"] == "queued"
        assert run["events"][0]["stage"] == "queued"
        assert client.get(f"/api/runs/{run['id']}").json()["status"] == "queued"
        assert not (tmp_path / "runtime" / "artifacts" / "baselines.json").exists()


def test_llm_rejected_without_credentials_or_real_adapter(tmp_path, monkeypatch):
    from thesis.api import create_app
    from thesis.worker import Worker
    import thesis.config as config

    with Worker(tmp_path), TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/runs", json=baseline_request(mode="llm"),
                               headers={"Idempotency-Key": "no-key"})
        assert response.status_code == 503
        assert client.get("/api/runs").json() == []
    monkeypatch.setenv("THESIS_LLM_API_KEY", "sk-private-fixture")
    monkeypatch.setenv("THESIS_LLM_MODEL", "fixture-model")
    monkeypatch.setattr(config, "load_rd_adapter", lambda: None)
    with Worker(tmp_path), TestClient(create_app(tmp_path)) as client:
        environment = client.get("/api/overview").json()["environment"]
        assert environment["llm_configured"] is True
        assert environment["rdagent_available"] is False
        response = client.post("/api/runs", json=baseline_request(mode="llm"),
                               headers={"Idempotency-Key": "no-adapter"})
        assert response.status_code == 503
        assert "sk-private-fixture" not in response.text


def test_idempotent_replay_and_conflicts_survive_worker_and_api_restart(tmp_path):
    from thesis.api import create_app
    from thesis.worker import Worker

    with Worker(tmp_path), TestClient(create_app(tmp_path)) as client:
        first = client.post("/api/runs", json=baseline_request(), headers={"Idempotency-Key": "same"})
        assert first.status_code == 202
        run_id = first.json()["id"]
    with TestClient(create_app(tmp_path)) as client:
        replay = client.post("/api/runs", json=baseline_request(), headers={"Idempotency-Key": "same"})
        assert replay.status_code == 202
        assert replay.json()["id"] == run_id
        for changed in ({"prompt": "另一个请求"}, {"iterations": 2}, {"mode": "llm"}):
            conflict = client.post("/api/runs", json=baseline_request(**changed),
                                   headers={"Idempotency-Key": "same"})
            assert conflict.status_code == 409
        assert len(client.get("/api/runs").json()) == 1


@pytest.mark.parametrize("updates", [{"iterations": 0}, {"iterations": 3}, {"iterations": True},
                                      {"iterations": "1"}, {"mode": "other"}, {"prompt": "  "},
                                      {"expression": "__import__('os')"}])
def test_submit_validation(tmp_path, updates):
    from thesis.api import create_app

    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/runs", json=baseline_request(**updates),
                           headers={"Idempotency-Key": "bad"}).status_code == 422


def test_idempotency_key_is_required(tmp_path):
    from thesis.api import create_app

    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/runs", json=baseline_request()).status_code == 400


def test_concurrent_submissions_share_one_durable_id(tmp_path):
    from thesis.api import create_app
    from thesis.worker import Worker

    with Worker(tmp_path), TestClient(create_app(tmp_path)) as client:
        def submit(_):
            response = client.post("/api/runs", json=baseline_request(),
                                   headers={"Idempotency-Key": "concurrent"})
            assert response.status_code == 202
            return response.json()["id"]
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(submit, range(8)))
        assert len(set(ids)) == 1
        assert len(client.get("/api/runs").json()) == 1


def test_claim_is_atomic_and_only_one_run_can_be_active(tmp_path):
    from thesis.store import Store

    store = Store(tmp_path)
    store.heartbeat("a")
    store.heartbeat("b")
    first = store.enqueue(baseline_request(), "one")
    second = store.enqueue(baseline_request(iterations=2), "two")
    barrier = threading.Barrier(2)
    def claim(worker_id):
        barrier.wait(timeout=5)
        return worker_id, Store(tmp_path).claim(worker_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ("a", "b")))
    active = [(owner, run) for owner, run in claims if run is not None]
    assert len(active) == 1
    owner, run = active[0]
    assert run["id"] == first["id"]
    assert Store(tmp_path).claim("a") is None
    assert Store(tmp_path).claim("b") is None
    with pytest.raises(ValueError):
        store.finish(run["id"], "not-owner", result={})
    store.finish(run["id"], owner, result={"ok": True})
    assert Store(tmp_path).claim("b")["id"] == second["id"]


def test_heartbeat_expiry_controls_submission(tmp_path):
    from thesis.api import create_app
    from thesis.store import Store

    store = Store(tmp_path)
    store.heartbeat("stale")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE heartbeats SET heartbeat_at = ?", (time.time() - 60,))
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/overview").json()["environment"]["worker_running"] is False
        assert client.post("/api/runs", json=baseline_request(),
                           headers={"Idempotency-Key": "stale"}).status_code == 503
    store.heartbeat("stale")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/overview").json()["environment"]["worker_running"] is True


def test_worker_heartbeats_during_a_slow_task_and_stops_on_close(tmp_path, monkeypatch):
    from thesis.worker import Worker
    from thesis.store import Store

    import thesis.baselines as baselines
    entered = threading.Event()
    release = threading.Event()
    def slow_baseline(root):
        assert root == tmp_path
        entered.set()
        assert release.wait(timeout=6)
        return write_archive(root)
    monkeypatch.setattr(baselines, "build_baselines", slow_baseline)
    with Worker(tmp_path) as worker:
        worker.store.enqueue(baseline_request(), "slow")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker.run_once)
            try:
                assert entered.wait(timeout=3)
                with sqlite3.connect(worker.store.path) as connection:
                    before = connection.execute("SELECT MAX(heartbeat_at) FROM heartbeats").fetchone()[0]
                time.sleep(1.2)
                with sqlite3.connect(worker.store.path) as connection:
                    after = connection.execute("SELECT MAX(heartbeat_at) FROM heartbeats").fetchone()[0]
                assert after > before
                assert Store(tmp_path).worker_running() is True
            finally:
                release.set()
            assert future.result(timeout=5)["status"] == "complete"
    assert Store(tmp_path).worker_running() is False


def test_second_worker_is_rejected_even_if_heartbeat_is_stale(tmp_path):
    from thesis.worker import Worker, WorkerAlreadyRunning

    with Worker(tmp_path) as first:
        with sqlite3.connect(first.store.path) as connection:
            connection.execute("UPDATE heartbeats SET heartbeat_at = ?", (time.time() - 60,))
        with pytest.raises(WorkerAlreadyRunning):
            Worker(tmp_path).start()


def test_recovery_marks_old_running_interrupted_and_never_requeues_it(tmp_path):
    from thesis.api import create_app
    from thesis.store import Store
    from thesis.worker import Worker

    store = Store(tmp_path)
    store.heartbeat("dead-worker")
    old = store.enqueue(baseline_request(mode="llm"), "old-model-call")
    assert store.claim("dead-worker")["id"] == old["id"]
    pending = store.enqueue(baseline_request(), "pending")
    with Worker(tmp_path) as restarted:
        recovered = restarted.store.get_run(old["id"])
        assert recovered["status"] == "failed"
        assert "interrupted" in recovered["error"]
        assert any(event["stage"] == "interrupted" for event in recovered["events"])
        assert restarted.store.get_run(pending["id"])["status"] == "queued"
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(f"/api/runs/{old['id']}").json()["status"] == "failed"
        replay = client.post("/api/runs", json=baseline_request(mode="llm"),
                             headers={"Idempotency-Key": "old-model-call"})
        assert replay.json()["id"] == old["id"]
        assert replay.json()["status"] == "failed"


def test_api_restart_does_not_interrupt_live_worker(tmp_path):
    from thesis.api import create_app
    from thesis.worker import Worker

    with Worker(tmp_path) as worker:
        run = worker.store.enqueue(baseline_request(), "live")
        worker.store.claim(worker.worker_id)
        with TestClient(create_app(tmp_path)) as client:
            assert client.get(f"/api/runs/{run['id']}").json()["status"] == "running"


def test_worker_executes_real_baseline_and_persists_result(tmp_path):
    import pandas as pd
    from thesis.api import create_app
    from thesis.worker import Worker

    days = pd.bdate_range("2024-01-01", periods=90)
    rows = [{"datetime": date, "instrument": f"S{s}", "open": 10 + s + d * .01,
             "high": 12 + s + d * .01, "low": 9 + s + d * .01,
             "close": 11 + s + d * .01, "adj_close": 11 + s + d * .01,
             "volume": 1000 + 20 * s + d} for d, date in enumerate(days) for s in range(6)]
    data = tmp_path / "runtime" / "data"
    data.mkdir(parents=True)
    pd.DataFrame(rows).set_index(["datetime", "instrument"]).to_parquet(data / "panel.parquet")
    (data / "manifest.json").write_text(json.dumps({"id": "integration-fixture", "source": "fixture",
        "quality": {}, "splits": {"train": [str(days[0].date()), str(days[49].date())],
        "validation": [str(days[50].date()), str(days[74].date())],
        "test": [str(days[75].date()), str(days[-1].date())]}}))
    with Worker(tmp_path) as worker, TestClient(create_app(tmp_path)) as client:
        queued = client.post("/api/runs", json=baseline_request(), headers={"Idempotency-Key": "real"}).json()
        completed = worker.run_once()
        assert completed["id"] == queued["id"]
        assert completed["status"] == "complete"
        assert completed["result"]["factors"]
        assert completed["result"]["test_evaluated"] is False
        assert {"queued", "running", "complete"} <= {e["stage"] for e in completed["events"]}
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(f"/api/runs/{queued['id']}").json()["status"] == "complete"
        assert client.get("/api/reports/export").json()["runs"][0]["result"]["factors"]


def test_worker_failure_is_durable_and_redacts_credentials(tmp_path, monkeypatch):
    from thesis.worker import Worker
    from thesis.api import create_app
    import thesis.baselines as baselines

    monkeypatch.setenv("THESIS_LLM_API_KEY", "sk-private-fixture")
    def broken(root):
        raise RuntimeError("failure with sk-private-fixture")
    monkeypatch.setattr(baselines, "build_baselines", broken)
    with Worker(tmp_path) as worker:
        queued = worker.store.enqueue(baseline_request(), "failed")
        finished = worker.run_once()
        assert finished["status"] == "failed"
        assert "sk-private-fixture" not in json.dumps(finished)
        assert worker.run_once() is None
    with TestClient(create_app(tmp_path)) as client:
        run = client.get(f"/api/runs/{queued['id']}").json()
        assert run["status"] == "failed"
        assert run["error"]
        assert run["events"][-1]["stage"] == "failed"


def test_config_env_precedence_public_values_and_default_root(tmp_path, monkeypatch):
    from thesis.api import create_app
    from thesis.config import load_config

    monkeypatch.setenv("THESIS_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "fallback-model")
    config = load_config()
    assert config.root == tmp_path
    assert config.llm_api_key == "fallback-key"
    assert config.llm_base_url == "https://example.invalid/v1"
    assert config.llm_model == "fallback-model"
    assert config.llm_configured is True
    monkeypatch.setenv("THESIS_LLM_API_KEY", "preferred-key")
    monkeypatch.setenv("THESIS_LLM_MODEL", "preferred-model")
    monkeypatch.setenv("THESIS_LLM_BASE_URL", "https://preferred.invalid/v1")
    preferred = load_config()
    assert preferred.llm_api_key == "preferred-key"
    assert preferred.llm_base_url == "https://preferred.invalid/v1"
    with TestClient(create_app()) as client:
        response = client.get("/api/overview")
        assert response.json()["environment"]["model"] == "preferred-model"
        assert "preferred-key" not in response.text
        assert "fallback-key" not in response.text


def test_production_static_files_spa_fallback_and_unknown_api_404(tmp_path):
    from thesis.api import create_app

    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>AlphaLab fixture</title>")
    (dist / "assets" / "app.js").write_text("window.fixture = true;")
    (tmp_path / "outside.txt").write_text("do not serve")
    with TestClient(create_app(tmp_path)) as client:
        assert "AlphaLab fixture" in client.get("/").text
        assert "AlphaLab fixture" in client.get("/research/deep-link").text
        assert "window.fixture" in client.get("/assets/app.js").text
        for path in ("/api", "/api/unknown", "/api/unknown/deep", "/assets/missing.js",
                     "/%2e%2e/%2e%2e/outside.txt"):
            response = client.get(path)
            assert response.status_code == 404
            assert "AlphaLab fixture" not in response.text
