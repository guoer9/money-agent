"""Explicit environment configuration and the public JSON boundary.

No dotenv files are read and credentials are never part of public configuration.
Both the HTTP process and the independent worker call ``load_config``.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
import math
from numbers import Integral, Real
import os
from pathlib import Path
from typing import Any


VERSION = "1.2.0"
HEARTBEAT_INTERVAL = 1.0
HEARTBEAT_TIMEOUT = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Settings:
    root: Path
    llm_api_key: str | None = field(default=None, repr=False)
    llm_base_url: str = field(default="https://api.openai.com/v1", repr=False)
    llm_model: str | None = None
    secret_values: tuple[str, ...] = field(default=(), repr=False)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    def environment(self, *, worker_running: bool) -> dict:
        return {
            "llm_configured": self.llm_configured,
            "rdagent_available": load_rd_adapter() is not None,
            "worker_running": worker_running,
            "model": self.llm_model,
        }


def load_config(root: Path | None = None) -> Settings:
    def first(*names, default=None):
        return next((os.environ[name].strip() for name in names if os.environ.get(name, "").strip()), default)

    selected_root = root if root is not None else first("THESIS_ROOT", default=Path(__file__).resolve().parents[1])
    return Settings(
        root=Path(selected_root).expanduser().resolve(),
        llm_api_key=first("THESIS_LLM_API_KEY", "OPENAI_API_KEY"),
        llm_base_url=first("THESIS_LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                           default="https://api.openai.com/v1"),
        llm_model=first("THESIS_LLM_MODEL", "OPENAI_MODEL"),
        secret_values=tuple(os.environ[name].strip() for name in ("THESIS_LLM_API_KEY", "OPENAI_API_KEY")
                            if os.environ.get(name, "").strip()),
    )


def load_rd_adapter():
    """Load the real adapter, optionally honoring its upstream availability probe.

    The adapter contract is run_research(request, root, emit) -> dict. Missing or
    broken imports/probes are unavailable, never replaced with a baseline.
    """
    try:
        module = import_module("thesis.rd_adapter")
        if not callable(getattr(module, "run_research", None)):
            return None
        probe = getattr(module, "availability", None)
        if callable(probe):
            status = probe()
            available = status.get("available", False) if isinstance(status, dict) else status
            if not available:
                return None
        return module
    except Exception:
        return None


def public_json(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    """Remove credentials/raw labels and map nonfinite numbers to JSON null.

    This boundary applies to archives, run results, events, errors and exports.
    It does not rewrite the original experiment artifacts.
    """
    if isinstance(value, dict):
        def private(key):
            normalized = str(key).lower().replace("-", "_")
            return (any(part in normalized for part in ("api_key", "apikey", "secret", "password", "credential"))
                    or normalized in {"authorization", "access_token", "refresh_token", "token", "bearer",
                                      "labels", "raw_labels", "y_test", "test_targets", "test_returns"}
                    or ("label" in normalized and ("test" in normalized or "raw" in normalized)))
        return {str(key): public_json(item, secrets) for key, item in value.items() if not private(key)}
    if isinstance(value, (list, tuple)):
        return [public_json(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    raise TypeError(f"Unsupported public JSON type: {type(value).__name__}")
