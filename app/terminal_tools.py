from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120
MAX_OUTPUT_BYTES = 64 * 1024
READ_CHUNK_BYTES = 4096
TERMINAL_MODES = {"sandbox", "host", "disabled"}
DEFAULT_SANDBOX_IMAGE = "tff-harness-sandbox:latest"
SANDBOX_WORKSPACE = "/workspace"
SANDBOX_MEMORY = "512m"
SANDBOX_CPUS = "1.0"
SANDBOX_PIDS_LIMIT = 128
SANDBOX_TMPFS_SIZE = "64m"


class TerminalToolError(Exception):
    """A terminal tool request failed validation or execution."""


class TerminalTools:
    def __init__(
        self,
        root: Path,
        mode: str = "sandbox",
        sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
        docker_executable: str = "docker",
    ) -> None:
        self.root = root.resolve()
        normalized_mode = mode.strip().lower()
        if normalized_mode not in TERMINAL_MODES:
            choices = ", ".join(sorted(TERMINAL_MODES))
            raise ValueError(f"terminal mode must be one of: {choices}")
        if normalized_mode == "sandbox" and not sandbox_image.strip():
            raise ValueError("sandbox_image must be a non-empty string")
        self.mode = normalized_mode
        self.sandbox_image = sandbox_image.strip()
        self.docker_executable = docker_executable

    @property
    def definitions(self) -> list[dict[str, Any]]:
        if self.mode == "disabled":
            return []
        execution_description = (
            "inside an isolated, network-disabled container"
            if self.mode == "sandbox"
            else "on the host with the harness process's permissions"
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": (
                        f"Run a shell command {execution_description}, starting "
                        "from the model workspace, and return stdout, stderr, "
                        "exit code, and duration. This operation requires user "
                        "approval."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute.",
                            },
                            "cwd": {
                                "type": "string",
                                "description": (
                                    "Workspace-relative directory to start in, "
                                    "or '.' for the workspace root."
                                ),
                                "default": ".",
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": "Timeout from 1 to 120 seconds.",
                                "minimum": 1,
                                "maximum": MAX_TIMEOUT_SECONDS,
                                "default": DEFAULT_TIMEOUT_SECONDS,
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    @property
    def names(self) -> set[str]:
        return {"run_command"} if self.mode != "disabled" else set()

    def requires_approval(self, name: str) -> bool:
        return name in self.names

    @staticmethod
    def display_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments.get("command")
        cwd = arguments.get("cwd", ".")
        timeout = arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        return {
            "command": command if isinstance(command, str) else "",
            "cwd": cwd if isinstance(cwd, str) else "",
            "timeout_seconds": timeout if isinstance(timeout, int) else "",
            "preview": command[:2000] if isinstance(command, str) else "",
        }

    @staticmethod
    def display_result(result: dict[str, Any]) -> dict[str, Any]:
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        return {
            "exit_code": result.get("exit_code"),
            "stdout": stdout[:8000] if isinstance(stdout, str) else "",
            "stderr": stderr[:8000] if isinstance(stderr, str) else "",
            "timed_out": result.get("timed_out", False),
            "duration_ms": result.get("duration_ms"),
            "execution_mode": result.get("execution_mode"),
            "truncated": bool(
                result.get("stdout_truncated") or result.get("stderr_truncated")
            ),
        }

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name != "run_command":
                raise TerminalToolError(f"Unknown terminal tool: {name}")
            command = self._string(arguments, "command")
            cwd = self._optional_string(arguments, "cwd", ".")
            timeout = self._integer(
                arguments,
                "timeout_seconds",
                DEFAULT_TIMEOUT_SECONDS,
                minimum=1,
                maximum=MAX_TIMEOUT_SECONDS,
            )
            if self.mode == "disabled":
                raise TerminalToolError("Terminal access is disabled.")
            result = await self.run_command(command, cwd, timeout)
            ok = result["exit_code"] == 0 and not result["timed_out"]
            payload: dict[str, Any] = {"ok": ok, **result}
            if result["timed_out"]:
                payload["error"] = f"Command timed out after {timeout} seconds."
            elif result["exit_code"] != 0:
                payload["error"] = f"Command exited with code {result['exit_code']}."
            return json.dumps(payload, ensure_ascii=False)
        except TerminalToolError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except OSError as exc:
            if self.mode == "sandbox":
                message = (
                    "Could not start the Docker sandbox. Ensure Docker is "
                    f"installed and running, and that image '{self.sandbox_image}' "
                    f"exists: {exc}"
                )
            else:
                message = f"Could not run command: {exc}"
            return json.dumps({"ok": False, "error": message}, ensure_ascii=False)

    async def run_command(
        self,
        command: str,
        cwd: str = ".",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        working_directory = self._resolve_cwd(cwd)
        if self.mode == "disabled":
            raise TerminalToolError("Terminal access is disabled.")
        if self.mode == "sandbox":
            return await self._run_sandboxed(
                command,
                cwd,
                timeout_seconds,
            )
        return await self._run_host(
            command,
            cwd,
            timeout_seconds,
            working_directory,
        )

    async def _run_host(
        self,
        command: str,
        cwd: str,
        timeout_seconds: int,
        working_directory: Path,
    ) -> dict[str, Any]:
        return await self._run_process(
            ["/bin/sh", "-lc", command],
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            process_cwd=working_directory,
            execution_mode="host",
        )

    async def _run_sandboxed(
        self,
        command: str,
        cwd: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        container_name = f"tff-sandbox-{uuid.uuid4().hex}"
        container_cwd = (
            SANDBOX_WORKSPACE
            if cwd == "."
            else f"{SANDBOX_WORKSPACE}/{PurePosixPath(cwd).as_posix()}"
        )
        mount = f"type=bind,source={self.root},target={SANDBOX_WORKSPACE}"
        arguments = [
            self.docker_executable,
            "run",
            "--rm",
            "--init",
            "--name",
            container_name,
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(SANDBOX_PIDS_LIMIT),
            "--memory",
            SANDBOX_MEMORY,
            "--cpus",
            SANDBOX_CPUS,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--hostname",
            "sandbox",
            "--env",
            "HOME=/tmp",
            "--workdir",
            container_cwd,
            "--mount",
            mount,
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={SANDBOX_TMPFS_SIZE}",
            self.sandbox_image,
            "/bin/sh",
            "-lc",
            command,
        ]
        return await self._run_process(
            arguments,
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            execution_mode="sandbox",
            container_name=container_name,
        )

    async def _run_process(
        self,
        arguments: list[str],
        *,
        command: str,
        cwd: str,
        timeout_seconds: int,
        execution_mode: str,
        process_cwd: Path | None = None,
        container_name: str | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=str(process_cwd) if process_cwd is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout_task = asyncio.create_task(self._read_output(process.stdout))
        stderr_task = asyncio.create_task(self._read_output(process.stderr))
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            await self._stop_process(process)
            if container_name is not None:
                await self._remove_container(container_name)
        except asyncio.CancelledError:
            await self._stop_process(process)
            if container_name is not None:
                await self._remove_container(container_name)
            await asyncio.gather(stdout_task, stderr_task)
            raise

        stdout, stdout_truncated = await stdout_task
        stderr, stderr_truncated = await stderr_task
        duration_ms = round((time.monotonic() - started) * 1000)
        return {
            "command": command,
            "cwd": cwd,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "execution_mode": execution_mode,
        }

    async def _remove_container(self, name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                self.docker_executable,
                "rm",
                "--force",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except (OSError, asyncio.TimeoutError):
            return

    @staticmethod
    async def _read_output(
        stream: asyncio.StreamReader | None,
    ) -> tuple[str, bool]:
        if stream is None:
            return "", False
        output = bytearray()
        truncated = False
        while True:
            chunk = await stream.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            remaining = MAX_OUTPUT_BYTES - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        return output.decode("utf-8", errors="replace"), truncated

    @staticmethod
    def _signal_process_group(pid: int, requested_signal: signal.Signals) -> None:
        try:
            os.killpg(pid, requested_signal)
        except ProcessLookupError:
            pass

    async def _stop_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        self._signal_process_group(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            self._signal_process_group(process.pid, signal.SIGKILL)
            await process.wait()

    def _resolve_cwd(self, raw_path: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        if raw_path == ".":
            return self.root
        path = PurePosixPath(raw_path)
        if path.is_absolute() or not path.parts:
            raise TerminalToolError("cwd must be relative to the model workspace.")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise TerminalToolError("cwd traversal is not allowed.")
        candidate = (self.root / Path(*path.parts)).resolve()
        try:
            inside_root = os.path.commonpath([self.root, candidate]) == str(self.root)
        except ValueError:
            inside_root = False
        if not inside_root:
            raise TerminalToolError("cwd escapes the model workspace.")
        if not candidate.is_dir():
            raise TerminalToolError(f"cwd does not exist or is not a directory: {raw_path}")
        return candidate

    @staticmethod
    def _string(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TerminalToolError(f"{key} must be a non-empty string.")
        return value

    @staticmethod
    def _optional_string(arguments: dict[str, Any], key: str, default: str) -> str:
        if key not in arguments:
            return default
        value = arguments[key]
        if not isinstance(value, str) or not value:
            raise TerminalToolError(f"{key} must be a non-empty string.")
        return value

    @staticmethod
    def _integer(
        arguments: dict[str, Any],
        key: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        value = arguments.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TerminalToolError(f"{key} must be an integer.")
        if not minimum <= value <= maximum:
            raise TerminalToolError(f"{key} must be between {minimum} and {maximum}.")
        return value
