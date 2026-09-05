import json
import hashlib

import httpx
import pytest

from app.agent_runs import AgentRunStore
from app.config import Settings
from app.context_compaction import COMPACTION_MARKER, ContextCompactor
from app.main import create_app
from app.memory_tools import MemoryStore, MemoryTools


def make_run(store):
    return store.create(
        "Remember verified repository conventions",
        model="model-test",
        temperature=0.2,
        max_tokens=1024,
        budgets={
            "max_tool_rounds": 10,
            "max_tool_calls": 20,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "goal"},
        ],
    )


def successful_receipt(store, run_id):
    receipt, _ = store.begin_tool_receipt(
        run_id,
        "read-call",
        "read_file",
        {"path": "guide.txt"},
        {"path": "guide.txt"},
        False,
    )
    store.complete_tool_receipt(run_id, "read-call", '{"ok":true}', True)
    return receipt


def test_repository_index_is_searchable_and_skips_sensitive_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "guide.txt").write_text("Use atomic replacements for durable writes.")
    (workspace / ".env").write_text("API_KEY='should-not-be-indexed'")
    store = MemoryStore(tmp_path / "memory.json", workspace)

    refreshed = store.refresh_repository()
    result = store.search("atomic replacements", refresh_repository=False)

    assert refreshed["files"] == 1
    assert refreshed["skipped_sensitive"] == 1
    assert result["results"][0]["kind"] == "repository"
    assert result["results"][0]["provenance"]["path"] == "guide.txt"
    assert "untrusted evidence" in result["security"]
    assert "should-not-be-indexed" not in json.dumps(result)


def test_repository_memory_enforces_scope_and_direct_read_freshness(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "allowed").mkdir(parents=True)
    (workspace / "private").mkdir()
    source = workspace / "allowed" / "guide.txt"
    source.write_text("Scoped repository evidence.")
    (workspace / "private" / "notes.txt").write_text("Scoped repository evidence.")
    store = MemoryStore(tmp_path / "memory.json", workspace)
    store.refresh_repository()
    scoped = MemoryTools(store, read_only=True, scope="allowed")

    found = json.loads(
        scoped.execute("search_memory", {"query": "Scoped repository evidence"})
    )
    assert found["ok"] is True
    assert len(found["results"]) == 1
    assert found["results"][0]["provenance"]["path"] == "allowed/guide.txt"

    memory_id = found["results"][0]["id"]
    source.write_text("Changed after indexing.")
    stale = store.read(memory_id, allow_stale=True)
    denied = json.loads(
        scoped.execute("read_memory", {"memory_id": memory_id})
    )

    assert stale["stale"] is True
    assert stale["stale_reason"] == "source changed: allowed/guide.txt"
    assert denied["ok"] is False
    assert "stale" in denied["error"]


def test_lesson_requires_receipts_and_becomes_stale_when_source_changes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "guide.txt"
    source.write_text("The project uses atomic writes.")
    runs = AgentRunStore(tmp_path / "agent.json")
    run = make_run(runs)
    receipt = successful_receipt(runs, run["id"])
    memory = MemoryStore(tmp_path / "memory.json", workspace)
    tools = MemoryTools(memory, runs, run["id"])

    rejected = json.loads(
        tools.execute(
            "save_lesson",
            {
                "title": "Unsupported",
                "content": "This should fail.",
                "confidence": 0.8,
                "expires_in_days": 30,
                "receipt_ids": ["missing"],
            },
        )
    )
    saved = json.loads(
        tools.execute(
            "save_lesson",
            {
                "title": "Durable writes",
                "content": "Use atomic writes for persistent JSON state.",
                "confidence": 0.9,
                "expires_in_days": 30,
                "receipt_ids": [receipt["id"]],
                "source_paths": ["guide.txt"],
            },
        )
    )

    assert rejected["ok"] is False
    assert saved["ok"] is True
    assert saved["trust"] == "agent_inference"
    assert saved["provenance"]["receipts"][0]["receipt_id"] == receipt["id"]

    source.write_text("The convention changed.")
    current = memory.search(
        "persistent JSON state",
        kinds=["lesson"],
        refresh_repository=False,
    )
    historical = memory.search(
        "persistent JSON state",
        kinds=["lesson"],
        include_stale=True,
        refresh_repository=False,
    )

    assert current["results"] == []
    assert historical["results"][0]["stale"] is True
    assert historical["results"][0]["stale_reason"] == "source changed: guide.txt"


def test_completed_run_memory_is_immutable_to_model_tools(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs = AgentRunStore(tmp_path / "agent.json")
    run = make_run(runs)
    completed = {**run, "status": "completed", "summary": "Verified the convention."}
    memory = MemoryStore(tmp_path / "memory.json", workspace)
    saved = memory.save_run(completed)
    tools = MemoryTools(memory, runs, run["id"])

    found = memory.search(
        "Verified convention",
        kinds=["run"],
        refresh_repository=False,
    )
    forgotten = json.loads(
        tools.execute("forget_lesson", {"memory_id": saved["id"]})
    )

    assert found["results"][0]["trust"] == "harness_record"
    assert forgotten["ok"] is False
    assert "Only agent-authored lessons" in forgotten["error"]


def test_completed_graph_artifacts_are_searchable_and_exact_duplicates_collapse(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs = AgentRunStore(tmp_path / "agent.json")
    run = make_run(runs)
    artifact_content = {"strategy": "Rollback restores the previous snapshot safely."}
    encoded = json.dumps(
        artifact_content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    artifact = {
        "id": "artifact-result",
        "name": "recovery_strategy",
        "type": "json",
        "content": artifact_content,
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "created_at": run["created_at"],
    }
    secret_content = {"api_key": "should-not-enter-memory"}
    secret_encoded = json.dumps(
        secret_content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    secret_artifact = {
        **artifact,
        "id": "artifact-secret",
        "name": "credentials",
        "content": secret_content,
        "content_sha256": hashlib.sha256(secret_encoded).hexdigest(),
        "bytes": len(secret_encoded),
    }
    completed = {
        **run,
        "status": "completed",
        "summary": "Captured recovery guidance.",
        "worker_tasks": {
            "task-a": {"id": "task-a", "title": "Recovery", "artifacts": [artifact]},
            "task-b": {
                "id": "task-b",
                "title": "Recovery duplicate",
                "artifacts": [{**artifact, "id": "artifact-duplicate"}],
            },
            "task-c": {"id": "task-c", "title": "Sensitive", "artifacts": [secret_artifact]},
        },
        "task_graph": {
            "id": "graph-a",
            "nodes": {
                "recover": {
                    "task_id": "task-a",
                    "attempts": [{"task_id": "task-b"}],
                    "loop_iterations": [],
                }
            },
        },
    }
    memory = MemoryStore(tmp_path / "memory.json", workspace)

    captured = memory.save_run(completed)
    found = memory.search(
        "undo changes using the prior snapshot",
        kinds=["artifact"],
        refresh_repository=False,
    )
    remembered = memory.read(found["results"][0]["id"])

    assert captured["artifacts_indexed"] == 1
    assert len(found["results"]) == 1
    assert found["results"][0]["score_components"]["semantic"] > 0
    assert remembered["kind"] == "artifact"
    assert remembered["consolidation_count"] == 2
    assert len(remembered["provenance"]["occurrences"]) == 2
    assert remembered["provenance"]["artifact_content_sha256"] == artifact["content_sha256"]
    assert "should-not-enter-memory" not in json.dumps(found)


def test_hybrid_retrieval_consolidates_lessons_and_feedback_can_quarantine(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs = AgentRunStore(tmp_path / "agent.json")
    run = make_run(runs)
    receipt = successful_receipt(runs, run["id"])
    memory = MemoryStore(tmp_path / "memory.json", workspace)
    arguments = {
        "content": "Rollback modifications safely after verifying the snapshot.",
        "confidence": 0.8,
        "expires_in_days": 30,
        "run_id": run["id"],
        "receipt_ids": [receipt["id"]],
        "source_paths": [],
        "receipts": runs.get(run["id"])["tool_receipts"],
    }

    first = memory.save_lesson(title="Safe recovery", **arguments)
    second = memory.save_lesson(title="Safe recovery procedure", **arguments)
    semantic = memory.search(
        "undo changes and test the saved state",
        kinds=["lesson"],
        refresh_repository=False,
    )
    for _ in range(3):
        feedback = memory.rate_lesson(
            first["id"], rating="unhelpful", reason="Outdated for this project."
        )
    hidden = memory.search(
        "rollback modifications",
        kinds=["lesson"],
        refresh_repository=False,
    )
    historical = memory.search(
        "rollback modifications",
        kinds=["lesson"],
        include_stale=True,
        refresh_repository=False,
    )

    assert second["consolidated"] is True
    assert second["id"] == first["id"]
    assert memory.read(first["id"], allow_stale=True)["consolidation_count"] == 2
    assert semantic["retrieval"]["mode"] == "hybrid"
    assert semantic["results"][0]["score_components"]["semantic"] > 0
    assert feedback["stale_reason"] == "rejected by user feedback"
    assert hidden["results"] == []
    assert historical["results"][0]["stale_reason"] == "rejected by user feedback"


def test_v1_memory_migrates_feedback_and_consolidation_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs = AgentRunStore(tmp_path / "agent.json")
    run = make_run(runs)
    completed = {**run, "status": "completed", "summary": "Legacy memory."}
    path = tmp_path / "memory.json"
    memory = MemoryStore(path, workspace)
    saved = memory.save_run(completed)
    document = json.loads(path.read_text())
    document["version"] = 1
    document["entries"][saved["id"]].pop("feedback")
    document["entries"][saved["id"]].pop("consolidation_count")
    path.write_text(json.dumps(document))

    migrated = MemoryStore(path, workspace).read(saved["id"])

    assert migrated["feedback"]["helpful"] == 0
    assert migrated["consolidation_count"] == 1
    assert json.loads(path.read_text())["version"] == 2


def test_context_compaction_preserves_recent_tool_group_and_archives_summary(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = MemoryStore(tmp_path / "memory.json", workspace)
    compactor = ContextCompactor(memory)
    run = {
        "id": "a" * 32,
        "goal": "Compact safely",
        "status": "working",
        "plan": [{"text": "Inspect", "status": "in_progress"}],
        "context_compactions": [],
    }
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "old analysis " * 80},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-old",
                    "type": "function",
                    "function": {"name": "search_files", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-old",
            "content": json.dumps({"ok": True, "count": 4}),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-new",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-new",
            "content": json.dumps({"ok": True, "path": "current.py"}),
        },
    ]

    compacted, record = compactor.compact(
        run,
        messages,
        threshold=500,
        target_recent=250,
    )

    assert record is not None
    assert record["removed_messages"] >= 1
    assert compacted[2]["content"].startswith(COMPACTION_MARKER)
    retained_ids = {
        message.get("tool_call_id")
        for message in compacted
        if message.get("role") == "tool"
    }
    assistant_ids = {
        call["id"]
        for message in compacted
        for call in message.get("tool_calls", [])
    }
    assert retained_ids <= assistant_ids
    archived = memory.read(record["archive_id"])
    assert archived["kind"] == "context"
    assert archived["trust"] == "harness_record"
    assert "Compacted activity" in archived["content"]


@pytest.mark.asyncio
async def test_memory_status_search_and_refresh_apis(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "architecture.md").write_text("Provenance protects durable memory.")
    settings = Settings(
        model_base_url="http://model.test",
        model_name="model-test",
        model_api_key=None,
        model_file_root=workspace,
        model_state_file=tmp_path / "state.json",
        agent_state_file=tmp_path / "agent.json",
        memory_file=tmp_path / "memory.json",
        terminal_mode="disabled",
    )
    runs = AgentRunStore(settings.agent_state_file)
    run = make_run(runs)
    receipt = successful_receipt(runs, run["id"])
    memory = MemoryStore(settings.memory_file, workspace)
    lesson = memory.save_lesson(
        title="Provenance guidance",
        content="Validate provenance before trusting durable memory.",
        confidence=0.9,
        expires_in_days=30,
        run_id=run["id"],
        receipt_ids=[receipt["id"]],
        source_paths=[],
        receipts=runs.get(run["id"])["tool_receipts"],
    )
    app = create_app(settings=settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        refreshed = await client.post("/api/memory/index/refresh")
        status = await client.get("/api/memory/status")
        searched = await client.get(
            "/api/memory/search",
            params={"q": "Provenance durable memory", "kinds": "repository"},
        )
        read = await client.get(f"/api/memory/entries/{lesson['id']}")
        rated = await client.post(
            f"/api/memory/lessons/{lesson['id']}/feedback",
            json={"rating": "helpful", "reason": "Confirmed by the operator."},
        )
        consolidated = await client.post("/api/memory/lessons/consolidate")

    assert refreshed.status_code == 200
    assert refreshed.json()["files"] == 1
    assert status.json()["repository"]["chunks"] == 1
    assert searched.json()["results"][0]["provenance"]["path"] == "architecture.md"
    assert read.status_code == 200
    assert read.json()["id"] == lesson["id"]
    assert rated.json()["feedback"]["helpful"] == 1
    assert consolidated.status_code == 200
