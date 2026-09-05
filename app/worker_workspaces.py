from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any
import uuid

from app.agent_runs import MAX_WORKER_CHANGES, AgentRunError, AgentRunStore


MAX_WORKER_WORKSPACE_FILES = 2000
MAX_WORKER_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_REVIEW_FILE_BYTES = 1024 * 1024
MAX_REVIEW_DIFF_CHARS = 64_000
MAX_REVIEW_HUNKS = 500
PROTECTED_DIRECTORIES = {".git", ".trash"}


class WorkerConflictError(AgentRunError):
    """A worker change cannot be applied because the root workspace diverged."""


class WorkerWorkspaceManager:
    def __init__(self, store: AgentRunStore, workspace_root: Path) -> None:
        self.store = store
        self.workspace_root = workspace_root.resolve()
        self.artifact_root = store.worker_artifact_root

    def prepare(self, run_id: str, task: dict[str, Any]) -> Path:
        task_root = self._task_root(run_id, task["id"])
        if task_root.exists():
            shutil.rmtree(task_root)
        private_workspace = task_root / "workspace"
        private_workspace.mkdir(parents=True)
        source = self._scope_source(task["scope"])
        baseline = self._manifest(source)
        for relative in baseline:
            source_file = source / relative
            destination = private_workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        (task_root / "baseline.json").write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return private_workspace

    def changes(self, run_id: str, task: dict[str, Any]) -> list[dict[str, Any]]:
        task_root = self._task_root(run_id, task["id"])
        private_workspace = task_root / "workspace"
        baseline_path = task_root / "baseline.json"
        if not private_workspace.is_dir() or not baseline_path.is_file():
            raise AgentRunError("The private worker workspace is unavailable.")
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AgentRunError("The worker baseline manifest is invalid.") from exc
        current = self._manifest(private_workspace)
        changes = []
        scope_prefix = "" if task["scope"] == "." else f"{task['scope']}/"
        for relative in sorted(set(baseline) | set(current)):
            before = baseline.get(relative)
            after = current.get(relative)
            if before == after:
                continue
            if before is None:
                action = "create"
            elif after is None:
                action = "delete"
            else:
                action = "update"
            changes.append(
                {
                    "path": f"{scope_prefix}{relative}",
                    "action": action,
                    "before_sha256": before["sha256"] if before else None,
                    "after_sha256": after["sha256"] if after else None,
                    "bytes": (after or before)["bytes"],
                }
            )
        if len(changes) > MAX_WORKER_CHANGES:
            raise AgentRunError(
                f"Worker produced {len(changes)} changed files; the limit is "
                f"{MAX_WORKER_CHANGES}."
            )
        return changes

    def review(self, run_id: str, task: dict[str, Any]) -> dict[str, Any]:
        if (
            task["mode"] != "isolated_write"
            or task["status"] != "completed"
            or task["integration_status"] not in {"pending", "conflict"}
            or not task["change_set"]
        ):
            raise AgentRunError("This worker has no reviewable private changes.")
        private_workspace = self._task_root(run_id, task["id"]) / "workspace"
        if not private_workspace.is_dir():
            raise AgentRunError("The private worker workspace is unavailable.")
        files = []
        remaining = MAX_REVIEW_DIFF_CHARS
        review_truncated = False
        conflicts = []
        for change in task["change_set"]:
            target = self._workspace_target(change["path"])
            current = self._file_info(target)
            current_hash = current["sha256"] if current else None
            conflict = current_hash != change["before_sha256"]
            if conflict:
                conflicts.append(change["path"])
            before = self._bounded_bytes(target) if current is not None else b""
            staged = (
                self._staged_target(private_workspace, task, change["path"])
                if change["after_sha256"] is not None
                else None
            )
            if staged is not None:
                staged_info = self._file_info(staged)
                if (
                    staged_info is None
                    or staged_info["sha256"] != change["after_sha256"]
                ):
                    raise AgentRunError(
                        f"Staged worker file no longer matches its manifest: {change['path']}"
                    )
            after = self._bounded_bytes(staged) if staged is not None else b""
            binary = before is None or after is None
            too_large = binary and (
                (current is not None and current["bytes"] > MAX_REVIEW_FILE_BYTES)
                or (
                    staged is not None
                    and staged.is_file()
                    and staged.stat().st_size > MAX_REVIEW_FILE_BYTES
                )
            )
            hunks = []
            if binary:
                diff = (
                    "Diff unavailable because the file exceeds the 1 MB review limit."
                    if too_large
                    else "Binary file changed."
                )
            else:
                before_text = self._decode_review_text(before)
                after_text = self._decode_review_text(after)
                if before_text is None or after_text is None:
                    binary = True
                    diff = "Binary file changed."
                else:
                    hunks = self._diff_hunks(change, before_text, after_text)
                    diff = (
                        f"--- a/{change['path']}\n+++ b/{change['path']}\n"
                        + "".join(hunk["diff"] for hunk in hunks)
                    )
            truncated = len(diff) > remaining
            if truncated:
                diff = diff[: max(0, remaining)]
                review_truncated = True
            remaining = max(0, remaining - len(diff))
            files.append(
                {
                    "path": change["path"],
                    "action": change["action"],
                    "bytes": change["bytes"],
                    "binary": binary,
                    "conflict": conflict,
                    "truncated": truncated,
                    "diff": diff,
                    "hunks": [
                        {key: value for key, value in hunk.items() if key != "diff"}
                        for hunk in hunks
                    ],
                }
            )
            if remaining == 0:
                review_truncated = True
        return {
            "task_id": task["id"],
            "title": task["title"],
            "scope": task["scope"],
            "change_count": len(task["change_set"]),
            "conflicts": conflicts,
            "truncated": review_truncated,
            "files": files,
        }

    def integration_selection(
        self,
        run_id: str,
        task: dict[str, Any],
        accepted_paths: list[str] | None,
        accepted_hunks: dict[str, list[str]] | None,
    ) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
        changes_by_path = {change["path"]: change for change in task["change_set"]}
        full_paths = list(changes_by_path) if accepted_paths is None and accepted_hunks is None else (accepted_paths or [])
        hunk_selection = accepted_hunks or {}
        if (
            len(full_paths) != len(set(full_paths))
            or any(path not in changes_by_path for path in full_paths)
            or any(path not in changes_by_path for path in hunk_selection)
            or set(full_paths) & set(hunk_selection)
        ):
            raise AgentRunError("The worker review selection contains invalid paths.")
        if any(
            not isinstance(hunks, list)
            or not hunks
            or len(hunks) != len(set(hunks))
            or any(not isinstance(hunk, str) or not hunk for hunk in hunks)
            for hunks in hunk_selection.values()
        ):
            raise AgentRunError("The worker review selection contains invalid hunk ids.")
        task_root = self._task_root(run_id, task["id"])
        private_workspace = task_root / "workspace"
        selection_workspace = task_root / "review-selection"
        if selection_workspace.exists():
            shutil.rmtree(selection_workspace)
        selection_workspace.mkdir(parents=True)
        selected_changes = []
        rejected_paths = []
        try:
            for path, change in changes_by_path.items():
                target = self._workspace_target(path)
                current = self._file_info(target)
                current_hash = current["sha256"] if current else None
                if current_hash != change["before_sha256"]:
                    raise WorkerConflictError(
                        "Workspace changed after the worker snapshot; integration conflicts: "
                        + path
                    )
                if path in full_paths:
                    selected_changes.append(change)
                    if change["after_sha256"] is not None:
                        source = self._staged_target(private_workspace, task, path)
                        destination = self._staged_target(selection_workspace, task, path)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                    continue
                requested_hunks = hunk_selection.get(path)
                if requested_hunks is None:
                    rejected_paths.append(path)
                    continue
                before = target.read_bytes() if current is not None else b""
                staged = (
                    self._staged_target(private_workspace, task, path)
                    if change["after_sha256"] is not None
                    else None
                )
                after = staged.read_bytes() if staged is not None else b""
                before_text = self._decode_review_text(before)
                after_text = self._decode_review_text(after)
                if before_text is None or after_text is None:
                    raise AgentRunError(
                        f"Binary or oversized files require whole-file acceptance: {path}"
                    )
                output = self._select_hunks(
                    change,
                    before_text,
                    after_text,
                    requested_hunks,
                ).encode("utf-8")
                if output == before:
                    rejected_paths.append(path)
                    continue
                after_hash = hashlib.sha256(output).hexdigest()
                action = "create" if current is None else "update"
                if not output and change["after_sha256"] is None:
                    action = "delete"
                    after_hash = None
                selected = {
                    "path": path,
                    "action": action,
                    "before_sha256": change["before_sha256"],
                    "after_sha256": after_hash,
                    "bytes": len(output) if after_hash is not None else len(before),
                }
                selected_changes.append(selected)
                if after_hash is not None:
                    destination = self._staged_target(selection_workspace, task, path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(output)
            if not selected_changes:
                raise AgentRunError(
                    "Select at least one file or hunk, or discard the worker changes."
                )
        except Exception:
            shutil.rmtree(selection_workspace, ignore_errors=True)
            raise
        decision = {
            "original_change_count": len(task["change_set"]),
            "integrated_change_count": len(selected_changes),
            "accepted_paths": sorted(full_paths),
            "accepted_hunks": {
                path: list(hunks) for path, hunks in sorted(hunk_selection.items())
            },
            "rejected_paths": sorted(rejected_paths),
        }
        return selected_changes, selection_workspace, decision

    def integrate(
        self,
        run_id: str,
        task: dict[str, Any],
        *,
        selected_changes: list[dict[str, Any]] | None = None,
        staged_workspace: Path | None = None,
    ) -> dict[str, Any]:
        if task["mode"] != "isolated_write" or task["status"] != "completed":
            raise AgentRunError("Only a completed implementation worker can be integrated.")
        if task["integration_status"] != "pending" or not task["change_set"]:
            raise AgentRunError("This worker has no pending changes to integrate.")
        task_root = self._task_root(run_id, task["id"])
        private_workspace = staged_workspace or (task_root / "workspace")
        checkpoint = task_root / "checkpoint"
        if not private_workspace.is_dir():
            raise AgentRunError("The private worker workspace is unavailable.")

        prepared: list[tuple[dict[str, Any], Path, Path | None]] = []
        conflicts = []
        integration_changes = selected_changes or task["change_set"]
        for change in integration_changes:
            target = self._workspace_target(change["path"])
            current = self._file_info(target)
            current_hash = current["sha256"] if current else None
            if current_hash != change["before_sha256"]:
                conflicts.append(change["path"])
                continue
            staged = None
            if change["after_sha256"] is not None:
                staged = self._staged_target(private_workspace, task, change["path"])
                staged_info = self._file_info(staged)
                if (
                    staged_info is None
                    or staged_info["sha256"] != change["after_sha256"]
                ):
                    raise AgentRunError(
                        f"Staged worker file no longer matches its manifest: {change['path']}"
                    )
            prepared.append((change, target, staged))
        if conflicts:
            raise WorkerConflictError(
                "Workspace changed after the worker snapshot; integration conflicts: "
                + ", ".join(conflicts[:20])
            )

        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        (checkpoint / "original").mkdir(parents=True)
        for change, target, _ in prepared:
            if change["before_sha256"] is None:
                continue
            backup = checkpoint / "original" / change["path"]
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        (checkpoint / "changes.json").write_text(
            json.dumps(integration_changes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        try:
            for change, target, staged in prepared:
                if change["action"] == "delete":
                    target.unlink()
                    continue
                assert staged is not None
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                try:
                    shutil.copy2(staged, temporary)
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
        except Exception as exc:
            self._restore(integration_changes, checkpoint)
            raise AgentRunError(f"Worker integration failed and was rolled back: {exc}") from exc

        return {
            "task_id": task["id"],
            "integrated": True,
            "changes": len(integration_changes),
            "change_set": integration_changes,
        }

    def finalize_integration(self, run_id: str, task_id: str) -> None:
        task_root = self._task_root(run_id, task_id)
        for name in ("workspace", "review-selection"):
            directory = task_root / name
            if directory.exists():
                shutil.rmtree(directory)

    def clear_selection(self, run_id: str, task_id: str) -> None:
        selection = self._task_root(run_id, task_id) / "review-selection"
        if selection.exists():
            shutil.rmtree(selection)

    def revert_unrecorded_integration(
        self,
        run_id: str,
        task: dict[str, Any],
        changes: list[dict[str, Any]] | None = None,
    ) -> None:
        checkpoint = self._task_root(run_id, task["id"]) / "checkpoint"
        if not checkpoint.is_dir():
            return
        self._restore(changes or self._checkpoint_changes(checkpoint), checkpoint)
        shutil.rmtree(checkpoint)

    def recover_pending_integration(
        self,
        run_id: str,
        task: dict[str, Any],
    ) -> str | None:
        checkpoint = self._task_root(run_id, task["id"]) / "checkpoint"
        if task["integration_status"] != "pending" or not checkpoint.is_dir():
            return None
        changes = self._checkpoint_changes(checkpoint)
        for change in changes:
            target = self._workspace_target(change["path"])
            current = self._file_info(target)
            current_hash = current["sha256"] if current else None
            if current_hash not in {
                change["before_sha256"],
                change["after_sha256"],
            }:
                return "conflict"
        self._restore(changes, checkpoint)
        shutil.rmtree(checkpoint)
        return "reverted"

    def rollback(self, run_id: str, task: dict[str, Any]) -> dict[str, Any]:
        if task["integration_status"] != "integrated":
            raise AgentRunError("Only integrated worker changes can be rolled back.")
        task_root = self._task_root(run_id, task["id"])
        checkpoint = task_root / "checkpoint"
        if not checkpoint.is_dir():
            raise AgentRunError("The integration checkpoint is unavailable.")
        conflicts = []
        for change in task["change_set"]:
            target = self._workspace_target(change["path"])
            current = self._file_info(target)
            current_hash = current["sha256"] if current else None
            if current_hash != change["after_sha256"]:
                conflicts.append(change["path"])
        if conflicts:
            raise WorkerConflictError(
                "Workspace changed after integration; rollback conflicts: "
                + ", ".join(conflicts[:20])
            )
        self._restore(task["change_set"], checkpoint)
        shutil.rmtree(task_root)
        return {
            "task_id": task["id"],
            "rolled_back": True,
            "changes": len(task["change_set"]),
        }

    def discard(self, run_id: str, task_id: str) -> None:
        task_root = self._task_root(run_id, task_id)
        if task_root.exists():
            shutil.rmtree(task_root)

    def _restore(self, changes: list[dict[str, Any]], checkpoint: Path) -> None:
        for change in reversed(changes):
            target = self._workspace_target(change["path"])
            if change["before_sha256"] is None:
                if target.exists():
                    target.unlink()
                continue
            backup = checkpoint / "original" / change["path"]
            if not backup.is_file():
                raise AgentRunError(
                    f"Checkpoint is missing the original file: {change['path']}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copy2(backup, temporary)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()

    @staticmethod
    def _checkpoint_changes(checkpoint: Path) -> list[dict[str, Any]]:
        try:
            changes = json.loads((checkpoint / "changes.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AgentRunError("The integration checkpoint manifest is invalid.") from exc
        if not isinstance(changes, list):
            raise AgentRunError("The integration checkpoint manifest is invalid.")
        return changes

    @staticmethod
    def _bounded_bytes(path: Path) -> bytes | None:
        if path.stat().st_size > MAX_REVIEW_FILE_BYTES:
            return None
        return path.read_bytes()

    @staticmethod
    def _decode_review_text(content: bytes) -> str | None:
        if len(content) > MAX_REVIEW_FILE_BYTES or b"\x00" in content:
            return None
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @classmethod
    def _diff_hunks(
        cls,
        change: dict[str, Any],
        before_text: str,
        after_text: str,
    ) -> list[dict[str, Any]]:
        before = before_text.splitlines(keepends=True)
        after = after_text.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, before, after)
        groups = list(matcher.get_grouped_opcodes(3))
        if len(groups) > MAX_REVIEW_HUNKS:
            raise AgentRunError(
                f"Review for {change['path']} exceeds the {MAX_REVIEW_HUNKS}-hunk limit."
            )
        hunks = []
        for group in groups:
            hunk_id = cls._hunk_id(change, group)
            old_start, old_end = group[0][1], group[-1][2]
            new_start, new_end = group[0][3], group[-1][4]
            header = (
                f"@@ -{cls._format_range(old_start, old_end)} "
                f"+{cls._format_range(new_start, new_end)} @@\n"
            )
            lines = [header]
            for tag, first_start, first_end, second_start, second_end in group:
                if tag == "equal":
                    lines.extend(cls._prefixed_lines(" ", before[first_start:first_end]))
                elif tag == "delete":
                    lines.extend(cls._prefixed_lines("-", before[first_start:first_end]))
                elif tag == "insert":
                    lines.extend(cls._prefixed_lines("+", after[second_start:second_end]))
                else:
                    lines.extend(cls._prefixed_lines("-", before[first_start:first_end]))
                    lines.extend(cls._prefixed_lines("+", after[second_start:second_end]))
            hunks.append(
                {
                    "id": hunk_id,
                    "header": header.strip(),
                    "old_start": old_start + 1,
                    "old_lines": old_end - old_start,
                    "new_start": new_start + 1,
                    "new_lines": new_end - new_start,
                    "diff": "".join(lines),
                }
            )
        return hunks

    @classmethod
    def _select_hunks(
        cls,
        change: dict[str, Any],
        before_text: str,
        after_text: str,
        requested_hunks: list[str],
    ) -> str:
        before = before_text.splitlines(keepends=True)
        after = after_text.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, before, after)
        opcodes = matcher.get_opcodes()
        groups = list(matcher.get_grouped_opcodes(3))
        selected_edits: set[tuple[str, int, int, int, int]] = set()
        available = {}
        for group in groups:
            hunk_id = cls._hunk_id(change, group)
            edits = {opcode for opcode in group if opcode[0] != "equal"}
            available[hunk_id] = edits
        unknown = sorted(set(requested_hunks) - set(available))
        if unknown:
            raise AgentRunError(
                f"Review selection contains stale or unknown hunks for {change['path']}."
            )
        for hunk_id in requested_hunks:
            selected_edits.update(available[hunk_id])
        output = []
        for opcode in opcodes:
            tag, first_start, first_end, second_start, second_end = opcode
            if tag == "equal" or opcode not in selected_edits:
                output.extend(before[first_start:first_end])
            elif tag in {"replace", "insert"}:
                output.extend(after[second_start:second_end])
        return "".join(output)

    @staticmethod
    def _hunk_id(
        change: dict[str, Any],
        group: list[tuple[str, int, int, int, int]],
    ) -> str:
        fingerprint = json.dumps(
            {
                "path": change["path"],
                "before": change["before_sha256"],
                "after": change["after_sha256"],
                "group": group,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _format_range(start: int, stop: int) -> str:
        beginning = start + 1
        length = stop - start
        if length == 1:
            return str(beginning)
        if length == 0:
            beginning -= 1
        return f"{beginning},{length}"

    @staticmethod
    def _prefixed_lines(prefix: str, lines: list[str]) -> list[str]:
        return [prefix + line if line.endswith("\n") else prefix + line + "\n" for line in lines]

    def _manifest(self, root: Path) -> dict[str, dict[str, Any]]:
        manifest: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        if not root.is_dir():
            raise AgentRunError("Worker scope is not an existing directory.")
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            retained_directories = []
            for name in directory_names:
                child = directory_path / name
                if name in PROTECTED_DIRECTORIES:
                    continue
                if child.is_symlink():
                    raise AgentRunError(
                        f"Worker snapshots do not support symbolic links: {child.relative_to(root)}"
                    )
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in file_names:
                path = directory_path / name
                relative = path.relative_to(root).as_posix()
                if path.is_symlink() or not path.is_file():
                    raise AgentRunError(
                        f"Worker snapshots support regular files only: {relative}"
                    )
                size = path.stat().st_size
                total_bytes += size
                if len(manifest) + 1 > MAX_WORKER_WORKSPACE_FILES:
                    raise AgentRunError(
                        f"Worker scope exceeds the {MAX_WORKER_WORKSPACE_FILES}-file limit."
                    )
                if total_bytes > MAX_WORKER_WORKSPACE_BYTES:
                    raise AgentRunError(
                        "Worker scope exceeds the 64 MB private-workspace limit."
                    )
                manifest[relative] = {
                    "sha256": self._sha256(path),
                    "bytes": size,
                }
        return manifest

    def _scope_source(self, scope: str) -> Path:
        return (
            self.workspace_root
            if scope == "."
            else self._workspace_target(scope)
        )

    def _task_root(self, run_id: str, task_id: str) -> Path:
        if not run_id.isalnum() or not task_id.isalnum():
            raise AgentRunError("Worker artifact id is invalid.")
        target = (self.artifact_root / run_id / task_id).resolve()
        if os.path.commonpath([self.artifact_root, target]) != str(self.artifact_root):
            raise AgentRunError("Worker artifact path escaped its storage root.")
        return target

    def _workspace_target(self, raw_path: str) -> Path:
        path = PurePosixPath(raw_path)
        if path.is_absolute() or not path.parts or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise AgentRunError("Worker change path is invalid.")
        target = (self.workspace_root / path.as_posix()).resolve()
        if os.path.commonpath([self.workspace_root, target]) != str(self.workspace_root):
            raise AgentRunError("Worker change escaped the model workspace.")
        return target

    @staticmethod
    def _staged_target(private_workspace: Path, task: dict[str, Any], path: str) -> Path:
        relative = (
            path
            if task["scope"] == "."
            else PurePosixPath(path).relative_to(PurePosixPath(task["scope"])).as_posix()
        )
        target = (private_workspace / relative).resolve()
        if os.path.commonpath([private_workspace, target]) != str(private_workspace):
            raise AgentRunError("Staged worker path escaped its private workspace.")
        return target

    @staticmethod
    def _file_info(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise AgentRunError(f"Expected a regular file: {path}")
        return {"sha256": WorkerWorkspaceManager._sha256(path), "bytes": path.stat().st_size}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
