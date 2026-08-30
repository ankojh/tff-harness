import json

import pytest

from app.terminal_tools import MAX_OUTPUT_BYTES, TerminalTools


@pytest.mark.asyncio
async def test_run_command_captures_stdout_stderr_and_exit_code(tmp_path):
    tools = TerminalTools(tmp_path, mode="host")

    result = json.loads(
        await tools.execute(
            "run_command",
            {"command": "printf hello; printf warning >&2; exit 7"},
        )
    )

    assert result["ok"] is False
    assert result["exit_code"] == 7
    assert result["stdout"] == "hello"
    assert result["stderr"] == "warning"
    assert result["error"] == "Command exited with code 7."
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_run_command_uses_workspace_relative_cwd(tmp_path):
    (tmp_path / "nested").mkdir()
    tools = TerminalTools(tmp_path, mode="host")

    result = json.loads(
        await tools.execute(
            "run_command",
            {"command": "pwd; printf data > result.txt", "cwd": "nested"},
        )
    )

    assert result["ok"] is True
    assert result["stdout"].strip() == str(tmp_path / "nested")
    assert (tmp_path / "nested" / "result.txt").read_text() == "data"


@pytest.mark.asyncio
async def test_run_command_times_out_and_stops_process(tmp_path):
    tools = TerminalTools(tmp_path, mode="host")

    result = json.loads(
        await tools.execute(
            "run_command",
            {"command": "sleep 5", "timeout_seconds": 1},
        )
    )

    assert result["ok"] is False
    assert result["timed_out"] is True
    assert "timed out" in result["error"]
    assert result["duration_ms"] < 3000


@pytest.mark.asyncio
async def test_run_command_truncates_large_output_but_still_drains_it(tmp_path):
    tools = TerminalTools(tmp_path, mode="host")

    result = json.loads(
        await tools.execute(
            "run_command",
            {"command": "yes x | head -c 70000"},
        )
    )

    assert result["ok"] is True
    assert len(result["stdout"].encode("utf-8")) == MAX_OUTPUT_BYTES
    assert result["stdout_truncated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("cwd", ["..", "nested/../../outside", "/tmp"])
async def test_run_command_rejects_cwd_outside_workspace(tmp_path, cwd):
    tools = TerminalTools(tmp_path, mode="host")

    result = json.loads(
        await tools.execute("run_command", {"command": "pwd", "cwd": cwd})
    )

    assert result["ok"] is False


def test_run_command_displays_command_for_tool_events(tmp_path):
    tools = TerminalTools(tmp_path, mode="host")

    assert tools.display_arguments("run_command", {"command": "pytest"})["preview"] == "pytest"


@pytest.mark.asyncio
async def test_sandbox_uses_docker_with_isolation_flags(tmp_path):
    fake_docker = tmp_path / "fake-docker"
    fake_docker.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    fake_docker.chmod(0o755)
    workspace = tmp_path / "workspace"
    (workspace / "nested").mkdir(parents=True)
    tools = TerminalTools(
        workspace,
        mode="sandbox",
        sandbox_image="sandbox:test",
        docker_executable=str(fake_docker),
    )

    result = json.loads(
        await tools.execute(
            "run_command",
            {"command": "printf hello", "cwd": "nested"},
        )
    )

    assert result["ok"] is True
    assert result["execution_mode"] == "sandbox"
    arguments = result["stdout"].splitlines()
    assert arguments[0] == "run"
    assert arguments[arguments.index("--pull") + 1] == "never"
    assert arguments[arguments.index("--network") + 1] == "none"
    assert "--read-only" in arguments
    assert arguments[arguments.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in arguments
    assert "--pids-limit" in arguments
    assert "--memory" in arguments
    assert "--cpus" in arguments
    assert "--user" in arguments
    assert arguments[arguments.index("--workdir") + 1] == "/workspace/nested"
    mount = arguments[arguments.index("--mount") + 1]
    assert f"source={workspace.resolve()}" in mount
    assert "target=/workspace" in mount
    assert "noexec" in arguments[arguments.index("--tmpfs") + 1]
    assert arguments[-4:] == ["sandbox:test", "/bin/sh", "-lc", "printf hello"]


def test_disabled_terminal_is_not_exposed(tmp_path):
    tools = TerminalTools(tmp_path, mode="disabled")

    assert tools.definitions == []
    assert tools.names == set()
    assert tools.requires_approval("run_command") is False


def test_terminal_commands_always_require_approval(tmp_path):
    assert TerminalTools(tmp_path, mode="sandbox").requires_approval("run_command")
    assert TerminalTools(tmp_path, mode="host").requires_approval("run_command")
