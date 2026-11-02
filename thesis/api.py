"""Local research API. No research is executed by request handlers."""

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from thesis.config import VERSION, load_config, public_json, utc_now
from thesis.store import IdempotencyConflict, Store, WorkerUnavailable
from thesis.evidence import evidence_snapshot, read_note


FINAL_TEST = {"status": "not_run", "message": "最终独立测试尚未执行；当前归档仅用于验证集研究。"}


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: StrictStr = Field(min_length=1, max_length=8000)
    mode: Literal["llm", "baseline"]
    iterations: StrictInt = Field(ge=1, le=2)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value):
        if not value.strip():
            raise ValueError("研究描述不能为空")
        return value


def create_app(root: Path | None = None) -> FastAPI:
    config = load_config(root)
    store = Store(config.root)
    app = FastAPI(title="AlphaLab Research API", version=VERSION)
    app.state.config, app.state.store = config, store

    def response(value, *, status_code=200, headers=None):
        return JSONResponse(public_json(value, config.secret_values), status_code=status_code, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo arbitrary request values/credentials in validation errors.
        errors = [{"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in exc.errors()]
        return response({"detail": errors}, status_code=422)

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_error(request, exc):
        return response({"detail": str(exc)}, status_code=409)

    @app.exception_handler(WorkerUnavailable)
    async def unavailable_error(request, exc):
        return response({"detail": str(exc)}, status_code=503)

    def archive():
        path = config.root / "runtime" / "artifacts" / "baselines.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not all(key in value for key in ("dataset", "protocol", "factors", "generated_at")):
                raise ValueError("Archive does not match the research contract")
            return public_json(value, config.secret_values)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            raise HTTPException(status_code=503, detail="研究归档不可读或格式错误，请检查 runtime/artifacts/baselines.json") from None

    @app.get("/api/overview")
    def overview():
        current = archive()
        return response({"archive": current, "runs": store.list_runs(),
                         "evidence": evidence_snapshot(config.root, current),
                         "environment": config.environment(worker_running=store.worker_running()),
                         "final_test": FINAL_TEST, "server_time": utc_now(), "version": VERSION})

    @app.get("/api/datasets")
    def datasets():
        value = archive()
        return response([value["dataset"]] if value is not None else [])

    @app.get("/api/evidence")
    def evidence():
        return response(evidence_snapshot(config.root, archive()))

    @app.get("/api/evidence/notes/{note_id}")
    def evidence_note(note_id: str):
        value = read_note(config.root, note_id)
        if value is None:
            raise HTTPException(status_code=404, detail="研究笔记不存在或不可读")
        return PlainTextResponse(public_json(value, config.secret_values),
                                 headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/api/factors")
    def factors():
        value = archive()
        return response(value["factors"] if value is not None else [])

    @app.get("/api/runs")
    def runs():
        return response(store.list_runs())

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="研究任务不存在")
        return response(run)

    @app.post("/api/runs", status_code=202)
    def submit(body: RunRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        if not idempotency_key or not idempotency_key.strip() or len(idempotency_key) > 200:
            raise HTTPException(status_code=400, detail="必须提供长度 1–200 的 Idempotency-Key")
        request = body.model_dump()
        previous = store.replay(request, idempotency_key)
        if previous is not None:
            return response(previous, status_code=202)
        if not store.worker_running():
            raise WorkerUnavailable("研究 worker 未在线，请先启动独立 worker")
        if body.mode == "llm":
            if not config.llm_configured:
                raise WorkerUnavailable("LLM 未配置：请设置 API key 和模型名，并在 API 与 worker 进程中使用相同配置")
            if not config.environment(worker_running=True)["rdagent_available"]:
                raise WorkerUnavailable("RD-Agent 适配器未可用；不能执行真实 LLM 研究")
        return response(store.enqueue(request, idempotency_key), status_code=202)

    @app.get("/api/reports/export")
    def export_report():
        environment = config.environment(worker_running=store.worker_running())
        current = archive()
        return response({"version": VERSION, "server_time": utc_now(), "archive": current,
                         "evidence": evidence_snapshot(config.root, current),
                         "runs": store.list_runs(), "config": {"version": VERSION, **environment},
                         "final_test": FINAL_TEST},
                        headers={"Content-Disposition": 'attachment; filename="alphalab-research-v1.2.0.json"'})

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API 路径不存在")
        dist = (config.root / "frontend" / "dist").resolve()
        candidate = (dist / path).resolve()
        if not candidate.is_relative_to(dist):
            raise HTTPException(status_code=404, detail="文件不存在")
        if candidate.is_file():
            return FileResponse(candidate)
        index = dist / "index.html"
        if not Path(path).suffix and index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="前端尚未构建或文件不存在")

    return app


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AlphaLab local HTTP API")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
