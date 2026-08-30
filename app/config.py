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


@dataclass(frozen=True)
class Settings:
    model_base_url: str
    model_name: str | None
    model_api_key: str | None
    model_file_root: Path = Path("model_workspace")
    model_state_file: Path | None = None
    terminal_mode: str = "sandbox"
    sandbox_image: str = "tff-harness-sandbox:latest"

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
        )
