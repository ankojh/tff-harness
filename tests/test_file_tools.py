import json

import pytest

from app.file_tools import FileToolError, FileTools


def test_create_read_and_recoverable_delete(tmp_path):
    tools = FileTools(tmp_path)

    created = tools.create_file("notes/hello.txt", "hello")
    assert created["created"] is True
    assert tools.read_file("notes/hello.txt")["content"] == "hello"

    deleted = tools.delete_file("notes/hello.txt")
    assert deleted["deleted"] is True
    assert not (tmp_path / "notes" / "hello.txt").exists()
    assert (tmp_path / deleted["recoverable_from"]).read_text() == "hello"


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "a/../../secret"])
def test_paths_cannot_escape_workspace(tmp_path, path):
    tools = FileTools(tmp_path)

    with pytest.raises(FileToolError):
        tools.read_file(path)


def test_create_never_overwrites(tmp_path):
    tools = FileTools(tmp_path)
    tools.create_file("existing.txt", "first")

    result = json.loads(tools.execute("create_file", {
        "path": "existing.txt",
        "content": "second",
    }))

    assert result["ok"] is False
    assert (tmp_path / "existing.txt").read_text() == "first"


def test_list_files_non_recursive_and_recursive(tmp_path):
    tools = FileTools(tmp_path)
    tools.create_file("top.txt", "top")
    tools.create_file("nested/note.txt", "note")

    shallow = tools.list_files()
    assert [entry["path"] for entry in shallow["entries"]] == ["nested", "top.txt"]

    recursive = tools.list_files(recursive=True)
    assert [entry["path"] for entry in recursive["entries"]] == [
        "nested",
        "nested/note.txt",
        "top.txt",
    ]


def test_search_files_supports_path_and_glob(tmp_path):
    tools = FileTools(tmp_path)
    tools.create_file("src/app.py", "first\nneedle here\n")
    tools.create_file("src/app.txt", "needle ignored by glob\n")
    tools.create_file("other.py", "needle outside path\n")

    result = tools.search_files("needle", path="src", glob="*.py")

    assert result["matches"] == [
        {"path": "src/app.py", "line": 2, "text": "needle here"}
    ]


def test_write_file_creates_and_overwrites(tmp_path):
    tools = FileTools(tmp_path)

    created = tools.write_file("note.txt", "one")
    overwritten = tools.write_file("note.txt", "two")

    assert created["created"] is True
    assert overwritten["overwritten"] is True
    assert (tmp_path / "note.txt").read_text() == "two"


def test_replace_in_file_requires_unique_match_by_default(tmp_path):
    tools = FileTools(tmp_path)
    tools.create_file("note.txt", "old old")

    with pytest.raises(FileToolError, match="occurs 2 times"):
        tools.replace_in_file("note.txt", "old", "new")

    result = tools.replace_in_file("note.txt", "old", "new", replace_all=True)

    assert result["replacements"] == 2
    assert (tmp_path / "note.txt").read_text() == "new new"


def test_write_and_replace_allow_empty_content(tmp_path):
    tools = FileTools(tmp_path)
    tools.write_file("empty.txt", "")
    tools.create_file("note.txt", "remove me")

    result = json.loads(
        tools.execute(
            "replace_in_file",
            {"path": "note.txt", "old_text": "remove me", "new_text": ""},
        )
    )

    assert result["ok"] is True
    assert (tmp_path / "empty.txt").read_text() == ""
    assert (tmp_path / "note.txt").read_text() == ""


def test_execute_returns_structured_error_for_invalid_optional_arguments(tmp_path):
    tools = FileTools(tmp_path)

    result = json.loads(tools.execute("list_files", {"path": None}))

    assert result == {"ok": False, "error": "path must be a string."}


def test_symlink_cannot_escape_workspace(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    tools = FileTools(tmp_path)

    with pytest.raises(FileToolError):
        tools.create_file("escape/file.txt", "blocked")
