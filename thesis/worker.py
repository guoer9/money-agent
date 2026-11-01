"""Independent, single-process research executor: python -m thesis.worker."""

import fcntl
import logging
from pathlib import Path
import signal
import threading
from uuid import uuid4

from thesis import config as configuration
from thesis.config import HEARTBEAT_INTERVAL, load_config
from thesis.store import Store


class WorkerAlreadyRunning(RuntimeError):
    pass


class Worker:
    def __init__(self, root: Path | None = None):
        self.config = load_config(root)
        self.store = Store(self.config.root)
        self.worker_id = uuid4().hex
        self._lock = None
        self._stop = threading.Event()
        self._heartbeat_thread = None
        self._execution_lock = threading.Lock()

    def start(self):
        if self._lock is not None:
            return self
        lock = (self.config.root / "runtime" / "worker.lock").open("a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise WorkerAlreadyRunning("此研究目录已有 worker 持有执行锁") from None
        try:
            self.store.start_worker(self.worker_id)
            self._lock = lock
            self._stop.clear()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat, name="alphalab-heartbeat", daemon=True)
            self._heartbeat_thread.start()
        except BaseException:
            self._lock = None
            lock.close()
            raise
        return self

    def _heartbeat(self):
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self.store.heartbeat(self.worker_id)
            except Exception:
                # A failed heartbeat expires on the API; never log credentials.
                logging.getLogger(__name__).warning("Worker heartbeat could not be persisted")

    def close(self):
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join()
            self._heartbeat_thread = None
        # Never release the process lock while another thread executes research.
        with self._execution_lock:
            if self._lock is not None:
                try:
                    self.store.stop_worker(self.worker_id)
                finally:
                    self._lock.close()
                    self._lock = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def run_once(self):
        if self._lock is None:
            raise RuntimeError("Worker 必须先 start 并持有执行锁")
        with self._execution_lock:
            run = self.store.claim(self.worker_id)
            if run is None:
                return None
            def emit(stage, message=None):
                if isinstance(stage, dict):
                    message, stage = stage.get("message", ""), stage.get("stage", "progress")
                self.store.append_event(run["id"], self.worker_id, str(stage), str(message or ""))
            try:
                if run["mode"] == "baseline":
                    from thesis.baselines import build_baselines
                    emit("baseline", "仅复算经典因子的验证集评价；不调用模型或最终测试")
                    result = build_baselines(self.config.root)
                else:
                    if not self.config.llm_configured:
                        raise RuntimeError("LLM 未配置，任务未调用模型")
                    adapter = configuration.load_rd_adapter()
                    if adapter is None:
                        raise RuntimeError("RD-Agent 适配器未可用，任务未调用模型")
                    request = {key: run[key] for key in ("id", "prompt", "mode", "iterations")}
                    result = adapter.run_research(request, self.config.root, emit)
                if not isinstance(result, dict):
                    raise TypeError("研究执行器必须返回真实结果 JSON 对象")
                return self.store.finish(run["id"], self.worker_id, result=result)
            except Exception as exc:
                return self.store.finish(run["id"], self.worker_id, error=f"{type(exc).__name__}: {exc}")
            except BaseException:
                self.store.finish(run["id"], self.worker_id,
                                  error="interrupted: worker 执行中断，不会自动重试模型调用")
                raise

    def run_forever(self):
        with self:
            while not self._stop.is_set():
                if self.run_once() is None:
                    self._stop.wait(0.25)


def main():
    worker = Worker()
    def stop(signum, frame):
        # Interrupt execution so an in-flight model call is never requeued.
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        pass
    except WorkerAlreadyRunning as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
