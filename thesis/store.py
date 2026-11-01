"""SQLite run queue, append-only events and worker heartbeats.

Each operation owns a connection. BEGIN IMMEDIATE and a partial unique index
serialize claims across processes; the worker additionally holds an OS lock.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from thesis.config import HEARTBEAT_TIMEOUT, load_config, public_json, utc_now


class IdempotencyConflict(ValueError):
    pass


class WorkerUnavailable(RuntimeError):
    pass


class Store:
    def __init__(self, root: Path):
        self.config = load_config(root)
        self.path = self.config.root / "runtime" / "research.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK (mode IN ('baseline', 'llm')),
                    iterations INTEGER NOT NULL CHECK (iterations IN (1, 2)),
                    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'complete', 'failed')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    worker_id TEXT,
                    result_json TEXT,
                    error TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS single_active_run
                    ON runs ((1)) WHERE status = 'running';
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    time TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_by_run ON events(run_id, id);
                CREATE TABLE IF NOT EXISTS heartbeats (
                    worker_id TEXT PRIMARY KEY,
                    heartbeat_at REAL NOT NULL,
                    active INTEGER NOT NULL CHECK (active IN (0, 1))
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _clean(self, value):
        return public_json(value, self.config.secret_values)

    @staticmethod
    def _fingerprint(request):
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _event(self, db, run_id, stage, message):
        now = utc_now()
        db.execute("INSERT INTO events(run_id, time, stage, message) VALUES (?, ?, ?, ?)",
                   (run_id, now, self._clean(str(stage))[:80], self._clean(str(message))[:4000]))
        db.execute("UPDATE runs SET updated_at = ? WHERE id = ?", (now, run_id))

    def _run(self, db, row):
        if row is None:
            return None
        run = {key: row[key] for key in ("id", "prompt", "mode", "iterations", "status", "created_at",
                                        "updated_at", "worker_id", "error")}
        run["result"] = json.loads(row["result_json"]) if row["result_json"] is not None else None
        run["events"] = [dict(event) for event in db.execute(
            "SELECT time, stage, message FROM events WHERE run_id = ? ORDER BY id", (row["id"],))]
        return self._clean(run)

    def _replay(self, db, request, key):
        row = db.execute("SELECT * FROM runs WHERE idempotency_key = ?", (key,)).fetchone()
        if row is not None and row["request_hash"] != self._fingerprint(request):
            raise IdempotencyConflict("相同 Idempotency-Key 已用于不同的研究请求")
        return self._run(db, row)

    def replay(self, request: dict, key: str):
        with self.connection() as db:
            return self._replay(db, request, key)

    def enqueue(self, request: dict, key: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self._replay(db, request, key)
            if previous is not None:
                return previous
            if not self._worker_running(db):
                raise WorkerUnavailable("研究 worker 未在线，请先启动独立 worker")
            run_id, now = uuid4().hex, utc_now()
            db.execute("""INSERT INTO runs
                (id, idempotency_key, request_hash, prompt, mode, iterations, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
                (run_id, key, self._fingerprint(request), self._clean(request["prompt"]), request["mode"],
                 request["iterations"], now, now))
            self._event(db, run_id, "queued", "研究已入队，等待独立 worker")
            return self._run(db, db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())

    def get_run(self, run_id: str):
        with self.connection() as db:
            return self._run(db, db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())

    def list_runs(self):
        with self.connection() as db:
            return [self._run(db, row) for row in db.execute("SELECT * FROM runs ORDER BY created_at DESC, id DESC")]

    @staticmethod
    def _worker_running(db, worker_id=None):
        sql = "SELECT 1 FROM heartbeats WHERE active = 1 AND heartbeat_at >= ?"
        params = [time.time() - HEARTBEAT_TIMEOUT]
        if worker_id is not None:
            sql += " AND worker_id = ?"
            params.append(worker_id)
        return db.execute(sql + " LIMIT 1", params).fetchone() is not None

    def worker_running(self):
        with self.connection() as db:
            return self._worker_running(db)

    def heartbeat(self, worker_id: str):
        with self.connection() as db:
            db.execute("""INSERT INTO heartbeats(worker_id, heartbeat_at, active) VALUES (?, ?, 1)
                ON CONFLICT(worker_id) DO UPDATE SET heartbeat_at=excluded.heartbeat_at, active=1""",
                       (worker_id, time.time()))

    def _interrupt(self, db, row):
        message = "interrupted: worker 执行中断；为避免重复模型调用，不会自动重试，请检查事件后重新提交"
        db.execute("UPDATE runs SET status='failed', error=? WHERE id=?", (message, row["id"]))
        self._event(db, row["id"], "interrupted", message)

    def start_worker(self, worker_id: str):
        """Only call after acquiring runtime/worker.lock, never from API startup."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT id FROM runs WHERE status='running'").fetchall():
                self._interrupt(db, row)
            db.execute("UPDATE heartbeats SET active=0")
            db.execute("INSERT INTO heartbeats(worker_id, heartbeat_at, active) VALUES (?, ?, 1)",
                       (worker_id, time.time()))

    def stop_worker(self, worker_id: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT id FROM runs WHERE status='running' AND worker_id=?",
                                  (worker_id,)).fetchall():
                self._interrupt(db, row)
            db.execute("UPDATE heartbeats SET active=0 WHERE worker_id=?", (worker_id,))

    def claim(self, worker_id: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._worker_running(db, worker_id):
                return None
            if db.execute("SELECT 1 FROM runs WHERE status='running'").fetchone():
                return None
            row = db.execute("SELECT * FROM runs WHERE status='queued' ORDER BY created_at, id LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE runs SET status='running', worker_id=? WHERE id=?", (worker_id, row["id"]))
            self._event(db, row["id"], "running", "独立 worker 已领取研究任务")
            return self._run(db, db.execute("SELECT * FROM runs WHERE id=?", (row["id"],)).fetchone())

    def append_event(self, run_id: str, worker_id: str, stage: str, message: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, run_id, worker_id)
            self._event(db, run_id, stage, message)

    @staticmethod
    def _owned(db, run_id, worker_id):
        row = db.execute("SELECT status, worker_id FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None or row["status"] != "running" or row["worker_id"] != worker_id:
            raise ValueError("只有当前领取任务的 worker 可以更新 running 任务")

    def finish(self, run_id: str, worker_id: str, *, result=None, error=None):
        cleaned_result = json.dumps(self._clean(result), ensure_ascii=False, allow_nan=False) if error is None else None
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, run_id, worker_id)
            status = "complete" if error is None else "failed"
            db.execute("UPDATE runs SET status=?, result_json=?, error=? WHERE id=?",
                       (status, cleaned_result, self._clean(str(error))[:4000] if error is not None else None, run_id))
            self._event(db, run_id, status, "研究已完成" if error is None else str(error))
            return self._run(db, db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())
