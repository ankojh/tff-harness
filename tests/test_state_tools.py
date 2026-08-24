import json

import pytest

import app.state_tools as state_module
from app.state_tools import MAX_STATE_VALUE_BYTES, StateToolError, StateTools


def test_state_persists_across_instances_and_list_omits_values(tmp_path):
    path = tmp_path / "state.json"
    first = StateTools(path)
    written = first.write_state("project/decision", "Use SQLite for the prototype.")

    second = StateTools(path)
    listing = second.list_state()
    read = second.read_state("project/decision")

    assert written["created"] is True
    assert listing["count"] == 1
    assert listing["entries"][0]["key"] == "project/decision"
    assert "value" not in listing["entries"][0]
    assert read["value"] == "Use SQLite for the prototype."


def test_write_state_replaces_instead_of_appending(tmp_path):
    tools = StateTools(tmp_path / "state.json")
    tools.write_state("task/status", "first")

    result = tools.write_state("task/status", "second")

    assert result["replaced"] is True
    assert result["total_value_bytes"] == len("second")
    assert tools.read_state("task/status")["value"] == "second"


def test_delete_state_removes_obsolete_entry(tmp_path):
    tools = StateTools(tmp_path / "state.json")
    tools.write_state("task/status", "done")

    result = tools.delete_state("task/status")

    assert result["deleted"] is True
    with pytest.raises(StateToolError, match="does not exist"):
        tools.read_state("task/status")


def test_state_rejects_oversized_value(tmp_path):
    tools = StateTools(tmp_path / "state.json")

    result = json.loads(
        tools.execute(
            "write_state",
            {"key": "large", "value": "x" * (MAX_STATE_VALUE_BYTES + 1)},
        )
    )

    assert result["ok"] is False
    assert "Summarize" in result["error"]


def test_state_enforces_entry_and_total_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(state_module, "MAX_STATE_ENTRIES", 2)
    monkeypatch.setattr(state_module, "MAX_STATE_TOTAL_BYTES", 6)
    tools = StateTools(tmp_path / "state.json")
    tools.write_state("one", "abc")
    tools.write_state("two", "def")

    entry_limit = json.loads(
        tools.execute("write_state", {"key": "three", "value": ""})
    )
    total_limit = json.loads(
        tools.execute("write_state", {"key": "two", "value": "toolong"})
    )

    assert entry_limit["ok"] is False
    assert "Delete or consolidate" in entry_limit["error"]
    assert total_limit["ok"] is False
    assert "Delete or consolidate" in total_limit["error"]


@pytest.mark.parametrize("key", ["", "/leading", "spaces are bad", "x" * 81])
def test_state_rejects_invalid_keys(tmp_path, key):
    tools = StateTools(tmp_path / "state.json")

    result = json.loads(tools.execute("read_state", {"key": key}))

    assert result["ok"] is False


def test_invalid_state_file_is_preserved(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json")
    tools = StateTools(path)

    result = json.loads(tools.execute("list_state", {}))

    assert result["ok"] is False
    assert path.read_text() == "not json"


def test_externally_oversized_state_is_rejected_on_load(tmp_path, monkeypatch):
    monkeypatch.setattr(state_module, "MAX_STATE_TOTAL_BYTES", 3)
    path = tmp_path / "state.json"
    document = {
        "version": 1,
        "entries": {
            "external": {
                "value": "too large",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        },
    }
    path.write_text(json.dumps(document))
    tools = StateTools(path)

    result = json.loads(tools.execute("list_state", {}))

    assert result["ok"] is False
    assert "not modified" in result["error"]
