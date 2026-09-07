from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


TERMINAL_MODES = {"sandbox", "host", "disabled"}


def _terminal_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in TERMINAL_MODES:
        choices = ", ".join(sorted(TERMINAL_MODES))
        raise ValueError(f"MODEL_TERMINAL_MODE must be one of: {choices}")
    return mode


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def validate_agent_state_path(file_root: Path, path: Path) -> Path:
    file_root = file_root.resolve()
    path = path.resolve()
    try:
        inside_workspace = os.path.commonpath([file_root, path]) == str(file_root)
    except ValueError:
        inside_workspace = False
    if inside_workspace:
        raise ValueError("MODEL_AGENT_STATE_FILE must be outside MODEL_FILE_ROOT")
    return path


def _agent_state_path(file_root: Path) -> Path:
    configured = os.getenv("MODEL_AGENT_STATE_FILE", "").strip()
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else (
            file_root.parent / ".tff_agent_runs" / f"{file_root.name}.json"
        ).resolve()
    )
    return validate_agent_state_path(file_root, path)


def _memory_path(file_root: Path) -> Path:
    configured = os.getenv("MODEL_MEMORY_FILE", "").strip()
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else (
            file_root.parent / ".tff_memory" / f"{file_root.name}.json"
        ).resolve()
    )
    return validate_agent_state_path(file_root, path)


def _eval_path(file_root: Path) -> Path:
    configured = os.getenv("MODEL_EVAL_FILE", "").strip()
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else (
            file_root.parent / ".tff_evals" / f"{file_root.name}.json"
        ).resolve()
    )
    return validate_agent_state_path(file_root, path)


@dataclass(frozen=True)
class Settings:
    model_base_url: str
    model_name: str | None
    model_api_key: str | None
    model_file_root: Path = Path("model_workspace")
    model_state_file: Path | None = None
    terminal_mode: str = "sandbox"
    sandbox_image: str = "tff-harness-sandbox:latest"
    agent_state_file: Path | None = None
    memory_file: Path | None = None
    eval_file: Path | None = None
    model_input_cost_per_million: float = 0.0
    model_output_cost_per_million: float = 0.0
    agent_max_tool_rounds: int = 320
    agent_max_tool_calls: int = 640
    agent_max_seconds: int = 9000
    agent_max_consecutive_failures: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        file_root = Path(
            os.getenv("MODEL_FILE_ROOT", "model_workspace")
        ).expanduser().resolve()
        configured_state_file = os.getenv("MODEL_STATE_FILE", "").strip()
        return cls(
            model_base_url=os.getenv(
                "MODEL_BASE_URL", "http://127.0.0.1:8080"
            ).rstrip("/"),
            model_name=os.getenv("MODEL_NAME", "").strip() or None,
            model_api_key=os.getenv("MODEL_API_KEY", "").strip() or None,
            model_file_root=file_root,
            model_state_file=(
                Path(configured_state_file).expanduser().resolve()
                if configured_state_file
                else file_root / ".harness_state.json"
            ),
            terminal_mode=_terminal_mode(
                os.getenv("MODEL_TERMINAL_MODE", "sandbox")
            ),
            sandbox_image=(
                os.getenv(
                    "MODEL_SANDBOX_IMAGE", "tff-harness-sandbox:latest"
                ).strip()
                or "tff-harness-sandbox:latest"
            ),
            agent_state_file=_agent_state_path(file_root),
            memory_file=_memory_path(file_root),
            eval_file=_eval_path(file_root),
            model_input_cost_per_million=_bounded_float(
                "MODEL_INPUT_COST_PER_MILLION", 0.0, 0.0, 100_000.0
            ),
            model_output_cost_per_million=_bounded_float(
                "MODEL_OUTPUT_COST_PER_MILLION", 0.0, 0.0, 100_000.0
            ),
            agent_max_tool_rounds=_bounded_int(
                "AGENT_MAX_TOOL_ROUNDS", 320, 1, 1000
            ),
            agent_max_tool_calls=_bounded_int(
                "AGENT_MAX_TOOL_CALLS", 640, 1, 5000
            ),
            agent_max_seconds=_bounded_int(
                "AGENT_MAX_SECONDS", 9000, 10, 86_400
            ),
            agent_max_consecutive_failures=_bounded_int(
                "AGENT_MAX_CONSECUTIVE_FAILURES", 3, 1, 20
            ),
        )
