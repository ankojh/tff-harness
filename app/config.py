from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_base_url: str
    model_name: str | None
    model_api_key: str | None
    model_file_root: Path = Path("model_workspace")
    model_state_file: Path | None = None

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
        )
