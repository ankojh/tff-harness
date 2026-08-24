from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
import uuid


MAX_STATE_ENTRIES = 128
MAX_STATE_KEY_CHARS = 80
MAX_STATE_VALUE_BYTES = 8 * 1024
MAX_STATE_TOTAL_BYTES = 64 * 1024
MAX_STATE_FILE_BYTES = 128 * 1024
STATE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

STATE_SYSTEM_PROMPT = """You have persistent state tools for durable context across conversations in this workspace.
- When earlier work, preferences, decisions, or task status may matter, call list_state, then read only relevant keys.
- Write or replace state when durable information changes. Use concise, topic-specific keys.
- Do not store chat transcripts, large tool outputs, temporary details, or secrets.
- Delete entries that are obsolete. Do not duplicate information across keys.
- State is bounded, so summarize durable conclusions rather than accumulating history."""


class StateToolError(Exception):
    """A state tool request failed validation or execution."""


class StateTools:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._lock = threading.Lock()

    @property
    def definitions(self) -> list[dict[str, Any]]:
        key = {
            "type": "string",
            "description": (
                "Topic-specific state key, for example 'project/decisions' or "
                "'user/preferences'."
            ),
            "maxLength": MAX_STATE_KEY_CHARS,
        }
        return [
            self._definition(
                "list_state",
                "List persistent state keys and metadata. Values are omitted; read only relevant keys afterward.",
                {},
                [],
            ),
            self._definition(
                "read_state",
                "Read one persistent state value by key.",
                {"key": key},
                ["key"],
            ),
            self._definition(
                "write_state",
                "Create or replace one concise persistent state value for future conversations.",
                {
                    "key": key,
                    "value": {
                        "type": "string",
                        "description": (
                            "Concise durable context. Replace prior summaries instead "
                            "of appending history."
                        ),
                    },
                },
                ["key", "value"],
            ),
            self._definition(
                "delete_state",
                "Delete a persistent state entry that is no longer useful or accurate.",
                {"key": key},
                ["key"],
            ),
        ]

    @property
    def names(self) -> set[str]:
        return {item["function"]["name"] for item in self.definitions}

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "list_state":
                result = self.list_state()
            elif name == "read_state":
                result = self.read_state(self._key(arguments))
            elif name == "write_state":
                result = self.write_state(
                    self._key(arguments),
                    self._string(arguments, "value", allow_empty=True),
                )
            elif name == "delete_state":
                result = self.delete_state(self._key(arguments))
            else:
                raise StateToolError(f"Unknown state tool: {name}")
            return json.dumps({"ok": True, **result}, ensure_ascii=False)
        except StateToolError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except OSError as exc:
            return json.dumps(
                {"ok": False, "error": f"State operation failed: {exc}"},
                ensure_ascii=False,
            )

    def list_state(self) -> dict[str, Any]:
        with self._lock:
            document = self._load()
            entries = [
                {
                    "key": key,
                    "bytes": len(entry["value"].encode("utf-8")),
                    "updated_at": entry["updated_at"],
                }
                for key, entry in sorted(document["entries"].items())
            ]
        return {
            "entries": entries,
            "count": len(entries),
            "limits": {
                "max_entries": MAX_STATE_ENTRIES,
                "max_value_bytes": MAX_STATE_VALUE_BYTES,
                "max_total_value_bytes": MAX_STATE_TOTAL_BYTES,
            },
        }

    def read_state(self, key: str) -> dict[str, Any]:
        with self._lock:
            entry = self._load()["entries"].get(key)
        if entry is None:
            raise StateToolError(f"State key does not exist: {key}")
        return {
            "key": key,
            "value": entry["value"],
            "updated_at": entry["updated_at"],
        }

    def write_state(self, key: str, value: str) -> dict[str, Any]:
        value_bytes = len(value.encode("utf-8"))
        if value_bytes > MAX_STATE_VALUE_BYTES:
            raise StateToolError(
                f"State value is {value_bytes} bytes; the per-entry limit is "
                f"{MAX_STATE_VALUE_BYTES} bytes. Summarize it before saving."
            )
        with self._lock:
            document = self._load()
            entries = document["entries"]
            created = key not in entries
            if created and len(entries) >= MAX_STATE_ENTRIES:
                raise StateToolError(
                    f"State already has {MAX_STATE_ENTRIES} entries. Delete or consolidate stale entries."
                )
            previous_bytes = (
                len(entries[key]["value"].encode("utf-8")) if key in entries else 0
            )
            total_bytes = self._total_value_bytes(entries) - previous_bytes + value_bytes
            if total_bytes > MAX_STATE_TOTAL_BYTES:
                raise StateToolError(
                    f"State values would total {total_bytes} bytes; the limit is "
                    f"{MAX_STATE_TOTAL_BYTES} bytes. Delete or consolidate stale entries."
                )
            entries[key] = {
                "value": value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save(document)
        return {
            "key": key,
            "bytes": value_bytes,
            "created": created,
            "replaced": not created,
            "total_value_bytes": total_bytes,
        }

    def delete_state(self, key: str) -> dict[str, Any]:
        with self._lock:
            document = self._load()
            if key not in document["entries"]:
                raise StateToolError(f"State key does not exist: {key}")
            del document["entries"][key]
            self._save(document)
        return {"key": key, "deleted": True}

    @staticmethod
    def display_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        key = arguments.get("key")
        return {"key": key if isinstance(key, str) else ""}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "entries": {}}
        file_bytes = self.path.stat().st_size
        if file_bytes > MAX_STATE_FILE_BYTES:
            raise StateToolError(
                f"State file is {file_bytes} bytes; the file limit is "
                f"{MAX_STATE_FILE_BYTES} bytes. It was not loaded or modified."
            )
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StateToolError(
                f"State file is invalid and was not modified: {self.path}"
            ) from exc
        if (
            not isinstance(document, dict)
            or document.get("version") != 1
            or not isinstance(document.get("entries"), dict)
        ):
            raise StateToolError(
                f"State file has an unsupported structure and was not modified: {self.path}"
            )
        entries = document["entries"]
        if len(entries) > MAX_STATE_ENTRIES:
            raise StateToolError(
                f"State file has {len(entries)} entries; the limit is "
                f"{MAX_STATE_ENTRIES}. It was not loaded or modified."
            )
        total_value_bytes = 0
        for key, entry in entries.items():
            if (
                not isinstance(key, str)
                or len(key) > MAX_STATE_KEY_CHARS
                or not STATE_KEY_PATTERN.fullmatch(key)
                or not isinstance(entry, dict)
                or not isinstance(entry.get("value"), str)
                or not isinstance(entry.get("updated_at"), str)
            ):
                raise StateToolError(
                    f"State file has an invalid entry and was not modified: {self.path}"
                )
            value_bytes = len(entry["value"].encode("utf-8"))
            if value_bytes > MAX_STATE_VALUE_BYTES:
                raise StateToolError(
                    f"State entry '{key}' is {value_bytes} bytes; the per-entry "
                    f"limit is {MAX_STATE_VALUE_BYTES}. The file was not modified."
                )
            total_value_bytes += value_bytes
        if total_value_bytes > MAX_STATE_TOTAL_BYTES:
            raise StateToolError(
                f"State values total {total_value_bytes} bytes; the limit is "
                f"{MAX_STATE_TOTAL_BYTES}. The file was not modified."
            )
        return document

    def _save(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _total_value_bytes(entries: dict[str, Any]) -> int:
        return sum(len(entry["value"].encode("utf-8")) for entry in entries.values())

    @staticmethod
    def _definition(
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _key(arguments: dict[str, Any]) -> str:
        key = StateTools._string(arguments, "key")
        if len(key) > MAX_STATE_KEY_CHARS or not STATE_KEY_PATTERN.fullmatch(key):
            raise StateToolError(
                "key must be 1-80 characters using letters, numbers, '.', '_', '/', or '-'."
            )
        return key

    @staticmethod
    def _string(
        arguments: dict[str, Any], key: str, allow_empty: bool = False
    ) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or (not value and not allow_empty):
            raise StateToolError(f"{key} must be a string.")
        return value
