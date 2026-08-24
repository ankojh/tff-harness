from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


MAX_READ_BYTES = 256 * 1024
MAX_WRITE_BYTES = 1024 * 1024
MAX_LIST_ENTRIES = 500
MAX_SEARCH_RESULTS = 200
MAX_SEARCH_LINE_CHARS = 500


class FileToolError(Exception):
    """A file tool request failed validation or execution."""


class FileTools:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            self._definition(
                "list_files",
                "List files and directories in the workspace.",
                {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory, or '.' for the root.",
                        "default": ".",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to include nested entries.",
                        "default": False,
                    },
                },
                [],
            ),
            self._definition(
                "search_files",
                "Search UTF-8 text files for a literal string.",
                {
                    "query": {
                        "type": "string",
                        "description": "Literal text to find.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file or directory to search.",
                        "default": ".",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional file glob such as '*.py'.",
                    },
                },
                ["query"],
            ),
            self._definition(
                "read_file",
                "Read a UTF-8 text file from the isolated workspace.",
                {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    }
                },
                ["path"],
            ),
            self._definition(
                "create_file",
                "Create a new UTF-8 text file. Existing files are never overwritten.",
                {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path for the new file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 contents of the new file.",
                    },
                },
                ["path", "content"],
            ),
            self._definition(
                "write_file",
                "Write complete UTF-8 contents to a file, creating or replacing it.",
                {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 contents for the file.",
                    },
                },
                ["path", "content"],
            ),
            self._definition(
                "replace_in_file",
                "Replace exact text in an existing UTF-8 file. By default the old text must occur exactly once.",
                {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of requiring one.",
                        "default": False,
                    },
                },
                ["path", "old_text", "new_text"],
            ),
            self._definition(
                "delete_file",
                "Move a file into the workspace trash so it can be recovered.",
                {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to trash.",
                    }
                },
                ["path"],
            ),
        ]

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

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "list_files":
                result = self.list_files(
                    self._optional_string(arguments, "path", "."),
                    self._boolean(arguments, "recursive", False),
                )
            elif name == "search_files":
                result = self.search_files(
                    self._string(arguments, "query"),
                    self._optional_string(arguments, "path", "."),
                    self._optional_string(arguments, "glob", None),
                )
            elif name == "read_file":
                result = self.read_file(self._string(arguments, "path"))
            elif name == "create_file":
                result = self.create_file(
                    self._string(arguments, "path"),
                    self._string(arguments, "content"),
                )
            elif name == "write_file":
                result = self.write_file(
                    self._string(arguments, "path"),
                    self._string(arguments, "content", allow_empty=True),
                )
            elif name == "replace_in_file":
                result = self.replace_in_file(
                    self._string(arguments, "path"),
                    self._string(arguments, "old_text"),
                    self._string(arguments, "new_text", allow_empty=True),
                    self._boolean(arguments, "replace_all", False),
                )
            elif name == "delete_file":
                result = self.delete_file(self._string(arguments, "path"))
            else:
                raise FileToolError(f"Unknown file tool: {name}")
            return json.dumps({"ok": True, **result}, ensure_ascii=False)
        except FileToolError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except OSError as exc:
            return json.dumps(
                {"ok": False, "error": f"File operation failed: {exc}"},
                ensure_ascii=False,
            )

    def list_files(self, path: str = ".", recursive: bool = False) -> dict[str, Any]:
        target = self._resolve(path, allow_root=True)
        if not target.is_dir():
            raise FileToolError(f"Directory does not exist: {path}")

        candidates = target.rglob("*") if recursive else target.iterdir()
        entries: list[dict[str, Any]] = []
        truncated = False
        for candidate in sorted(candidates, key=lambda item: item.as_posix()):
            relative = candidate.relative_to(self.root)
            if ".trash" in relative.parts or candidate.is_symlink():
                continue
            if len(entries) == MAX_LIST_ENTRIES:
                truncated = True
                break
            entry: dict[str, Any] = {
                "path": relative.as_posix(),
                "type": "directory" if candidate.is_dir() else "file",
            }
            if candidate.is_file():
                entry["bytes"] = candidate.stat().st_size
            entries.append(entry)
        return {"path": path, "entries": entries, "truncated": truncated}

    def search_files(
        self,
        query: str,
        path: str = ".",
        glob: str | None = None,
    ) -> dict[str, Any]:
        target = self._resolve(path, allow_root=True)
        if not target.exists():
            raise FileToolError(f"Path does not exist: {path}")
        if glob is not None and (not glob or "/" in glob or "\\" in glob):
            raise FileToolError("glob must be a non-empty filename pattern.")

        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = target.rglob(glob or "*")
        else:
            raise FileToolError(f"Path is not a file or directory: {path}")

        matches: list[dict[str, Any]] = []
        truncated = False
        for candidate in sorted(candidates, key=lambda item: item.as_posix()):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(self.root)
            if ".trash" in relative.parts or candidate.stat().st_size > MAX_READ_BYTES:
                continue
            try:
                content = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if query not in line:
                    continue
                if len(matches) == MAX_SEARCH_RESULTS:
                    truncated = True
                    break
                matches.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "text": line[:MAX_SEARCH_LINE_CHARS],
                    }
                )
            if truncated:
                break
        return {
            "query": query,
            "path": path,
            "matches": matches,
            "truncated": truncated,
        }

    def read_file(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise FileToolError(f"File does not exist: {path}")
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise FileToolError(
                f"File is {size} bytes; the read limit is {MAX_READ_BYTES} bytes."
            )
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise FileToolError("Only UTF-8 text files can be read.") from exc
        return {"path": path, "content": content, "bytes": size}

    def create_file(self, path: str, content: str) -> dict[str, Any]:
        encoded = self._validate_content(content)
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("x", encoding="utf-8") as file:
                file.write(content)
        except FileExistsError as exc:
            raise FileToolError(f"File already exists: {path}") from exc
        return {"path": path, "bytes": len(encoded), "created": True}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        encoded = self._validate_content(content)
        target = self._resolve(path)
        if target.exists() and not target.is_file():
            raise FileToolError(f"Path is not a file: {path}")
        created = not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "path": path,
            "bytes": len(encoded),
            "created": created,
            "overwritten": not created,
        }

    def replace_in_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise FileToolError(f"File does not exist: {path}")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise FileToolError("Only UTF-8 text files can be edited.") from exc
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise FileToolError("old_text was not found in the file.")
        if occurrences > 1 and not replace_all:
            raise FileToolError(
                f"old_text occurs {occurrences} times; set replace_all to replace all occurrences."
            )
        replacements = occurrences if replace_all else 1
        updated = content.replace(old_text, new_text, -1 if replace_all else 1)
        encoded = self._validate_content(updated)
        target.write_text(updated, encoding="utf-8")
        return {"path": path, "bytes": len(encoded), "replacements": replacements}

    def delete_file(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise FileToolError(f"File does not exist: {path}")
        trash = self.root / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        destination = trash / f"{uuid.uuid4().hex}-{target.name}"
        shutil.move(str(target), str(destination))
        return {
            "path": path,
            "deleted": True,
            "recoverable_from": str(destination.relative_to(self.root)),
        }

    def display_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        path = arguments.get("path")
        display: dict[str, Any] = {"path": path if isinstance(path, str) else ""}
        if name in {"create_file", "write_file"}:
            content = arguments.get("content")
            if isinstance(content, str):
                display["bytes"] = len(content.encode("utf-8"))
                display["preview"] = content[:1000]
        elif name == "replace_in_file":
            old_text = arguments.get("old_text")
            new_text = arguments.get("new_text")
            if isinstance(old_text, str) and isinstance(new_text, str):
                display["preview"] = f"Replace:\n{old_text[:450]}\n\nWith:\n{new_text[:450]}"
        elif name == "search_files":
            query = arguments.get("query")
            if isinstance(query, str):
                display["query"] = query
        return display

    def requires_approval(self, name: str) -> bool:
        return name in {"create_file", "write_file", "replace_in_file", "delete_file"}

    @staticmethod
    def _string(
        arguments: dict[str, Any], key: str, allow_empty: bool = False
    ) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or (not value and not allow_empty):
            raise FileToolError(f"{key} must be a non-empty string.")
        return value

    @staticmethod
    def _optional_string(
        arguments: dict[str, Any], key: str, default: str | None
    ) -> str | None:
        if key not in arguments:
            return default
        value = arguments[key]
        if not isinstance(value, str):
            raise FileToolError(f"{key} must be a string.")
        return value

    @staticmethod
    def _boolean(arguments: dict[str, Any], key: str, default: bool) -> bool:
        value = arguments.get(key, default)
        if not isinstance(value, bool):
            raise FileToolError(f"{key} must be a boolean.")
        return value

    @staticmethod
    def _validate_content(content: str) -> bytes:
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise FileToolError(
                f"Content is {len(encoded)} bytes; the write limit is "
                f"{MAX_WRITE_BYTES} bytes."
            )
        return encoded

    def _resolve(self, raw_path: str, allow_root: bool = False) -> Path:
        if allow_root and raw_path == ".":
            self.root.mkdir(parents=True, exist_ok=True)
            return self.root
        path = PurePosixPath(raw_path)
        if path.is_absolute() or not path.parts:
            raise FileToolError("Paths must be relative to the model workspace.")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise FileToolError("Path traversal is not allowed.")
        if path.parts[0] == ".trash":
            raise FileToolError("The workspace trash is reserved.")

        self.root.mkdir(parents=True, exist_ok=True)
        candidate = (self.root / Path(*path.parts)).resolve()
        try:
            inside_root = os.path.commonpath([self.root, candidate]) == str(self.root)
        except ValueError:
            inside_root = False
        if not inside_root:
            raise FileToolError("Path escapes the model workspace.")
        return candidate
