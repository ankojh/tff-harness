import json

import pytest

from app.terminal_tools import MAX_OUTPUT_BYTES, TerminalTools


@pytest.mark.asyncio
async def test_run_command_captures_stdout_stderr_and_exit_code(tmp_path):
    tools = TerminalTools(tmp_path)

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
    tools = TerminalTools(tmp_path)

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
    tools = TerminalTools(tmp_path)

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
    tools = TerminalTools(tmp_path)

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
    tools = TerminalTools(tmp_path)

    result = json.loads(
        await tools.execute("run_command", {"command": "pwd", "cwd": cwd})
    )

    assert result["ok"] is False


def test_run_command_displays_command_for_tool_events(tmp_path):
    tools = TerminalTools(tmp_path)

    assert tools.display_arguments("run_command", {"command": "pytest"})["preview"] == "pytest"
