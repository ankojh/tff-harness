import asyncio
from contextlib import suppress
import json

import httpx
import pytest

from app.agent_loop import AgentService
from app.agent_runs import AgentControlTools, AgentRunStore
from app.agent_workers import AgentWorkerTools
from app.approvals import ApprovalBroker
from app.config import Settings
from app.file_tools import FileTools
from app.model_gateway import ModelGateway
from app.pdf_tools import PdfTools
from app.state_tools import StateTools
from app.terminal_tools import TerminalTools
from app.web_tools import WebTools


def settings(tmp_path):
    return Settings(
        model_base_url="http://model.test",
        model_name="model-test",
        model_api_key=None,
        model_file_root=tmp_path / "workspace",
        agent_state_file=tmp_path / "agent.json",
        terminal_mode="disabled",
    )


def content_stream(text):
    payload = {
        "choices": [{"delta": {"content": text}, "finish_reason": "stop"}]
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()


def tool_stream(call_id, name, arguments):
    called = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    finished = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
    return (
        f"data: {json.dumps(called)}\n\n"
        f"data: {json.dumps(finished)}\n\n"
        "data: [DONE]\n\n"
    ).encode()


def worker_tools(tmp_path, handler):
    config = settings(tmp_path)
    config.model_file_root.mkdir()
    store = AgentRunStore(config.agent_state_file)
    run = store.create(
        "Root goal",
        model="model-test",
        temperature=0.4,
        max_tokens=4096,
        budgets={
            "max_tool_rounds": 20,
            "max_tool_calls": 40,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
        messages=[{"role": "user", "content": "Root goal"}],
    )
    AgentControlTools(store, run["id"]).execute(
        "agent_set_plan", {"steps": ["Delegate inspection"]}
    )
    tools = AgentWorkerTools(
        ModelGateway(config, httpx.MockTransport(handler)),
        FileTools(config.model_file_root),
        PdfTools(config.model_file_root),
        WebTools(),
        StateTools(tmp_path / "memory.json"),
        store,
        run["id"],
    )
    return tools, store, run


@pytest.mark.asyncio
async def test_worker_batch_runs_in_parallel_with_read_only_tools(tmp_path):
    active = 0
    peak = 0
    offered_names = []

    async def handler(request):
        nonlocal active, peak
        payload = json.loads(request.content)
        offered_names.append(
            {tool["function"]["name"] for tool in payload["tools"]}
        )
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return httpx.Response(200, content=content_stream("Scoped findings."))

    tools, store, run = worker_tools(tmp_path, handler)
    result = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": f"Inspect area {index}",
                        "instruction": "Find relevant implementation details.",
                        "role": role,
                        "scope": ".",
                    }
                    for index, role in enumerate(
                        ["researcher", "reviewer", "tester"], start=1
                    )
                ]
            },
        )
    )

    assert result["ok"] is True
    assert peak == 3
    assert {task["status"] for task in result["tasks"]} == {"completed"}
    assert len(store.worker_tasks(run["id"])) == 3
    assert all("write_file" not in names for names in offered_names)
    assert all("run_command" not in names for names in offered_names)
    assert all("agent_complete" not in names for names in offered_names)
    observed = store.get(run["id"])
    assert observed["usage"]["model_calls"] == 3
    assert observed["usage"]["estimated_token_calls"] == 3
    assert sum(span["kind"] == "worker_model" for span in observed["trace"]["spans"]) == 3


@pytest.mark.asyncio
async def test_worker_reads_only_inside_its_scoped_subtree(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        payload = json.loads(request.content)
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream("read", "read_file", {"path": "inside.txt"}),
            )
        tool_result = next(
            message["content"]
            for message in payload["messages"]
            if message.get("role") == "tool"
        )
        assert "scoped content" in tool_result
        return httpx.Response(200, content=content_stream("Scoped file confirmed."))

    tools, store, run = worker_tools(tmp_path, handler)
    scope = tools.file_tools.root / "package"
    scope.mkdir()
    (scope / "inside.txt").write_text("scoped content")
    (tools.file_tools.root / "outside.txt").write_text("outside")

    result = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Inspect package",
                        "instruction": "Read the package note.",
                        "role": "researcher",
                        "scope": "package",
                    }
                ]
            },
        )
    )

    assert result["ok"] is True
    task = store.worker_tasks(run["id"])[0]
    assert task["result"] == "Scoped file confirmed."
    assert task["usage"] == {"model_rounds": 2, "tool_calls": 1}


@pytest.mark.asyncio
async def test_worker_rejects_scope_outside_workspace(tmp_path):
    tools, store, run = worker_tools(
        tmp_path,
        lambda request: httpx.Response(200, content=content_stream("unused")),
    )

    result = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Escape",
                        "instruction": "Inspect outside.",
                        "role": "reviewer",
                        "scope": "../",
                    }
                ]
            },
        )
    )

    assert result["ok"] is False
    assert "inside" in result["error"]
    assert store.worker_tasks(run["id"]) == []


@pytest.mark.asyncio
async def test_cancelled_worker_is_persisted_as_stopped(tmp_path):
    started = asyncio.Event()

    async def handler(request):
        started.set()
        await asyncio.sleep(10)
        return httpx.Response(200, content=content_stream("too late"))

    tools, store, run = worker_tools(tmp_path, handler)
    execution = asyncio.create_task(
        tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Slow inspection",
                        "instruction": "Wait for context.",
                        "role": "tester",
                    }
                ]
            },
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    execution.cancel()
    with suppress(asyncio.CancelledError):
        await execution

    task = store.worker_tasks(run["id"])[0]
    assert task["status"] == "stopped"
    assert "parent run stopped" in task["error"]


def test_restart_stops_persisted_running_workers(tmp_path):
    tools, store, run = worker_tools(
        tmp_path,
        lambda request: httpx.Response(200, content=content_stream("unused")),
    )
    task = store.create_worker_tasks(
        run["id"],
        [
            {
                "title": "Interrupted",
                "instruction": "Inspect state.",
                "role": "reviewer",
                "scope": ".",
                "mode": "read_only",
            }
        ],
    )[0]
    store.mutate_worker_task(
        run["id"],
        task["id"],
        lambda record: record.__setitem__("status", "running"),
        event_type="worker_task_started",
    )

    AgentRunStore(store.path).mark_interrupted()
    recovered = AgentRunStore(store.path).worker_tasks(run["id"])[0]

    assert recovered["status"] == "stopped"
    assert "restarted" in recovered["error"]


@pytest.mark.asyncio
async def test_worker_round_budget_is_independent_and_bounded(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=tool_stream(f"list-{calls}", "list_files", {"path": "."}),
        )

    tools, store, run = worker_tools(tmp_path, handler)
    result = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Never-ending inspection",
                        "instruction": "Keep listing forever.",
                        "role": "researcher",
                    }
                ]
            },
        )
    )

    task = store.worker_tasks(run["id"])[0]
    assert result["ok"] is False
    assert task["status"] == "failed"
    assert task["usage"] == {"model_rounds": 6, "tool_calls": 6}
    assert "6-round budget" in task["error"]


@pytest.mark.asyncio
async def test_implementer_changes_private_snapshot_then_integrates_and_rolls_back(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write",
                    "write_file",
                    {"path": "note.txt", "content": "worker version"},
                ),
            )
        return httpx.Response(
            200,
            content=content_stream("Updated note.txt in the private workspace."),
        )

    tools, store, run = worker_tools(tmp_path, handler)
    note = tools.file_tools.root / "note.txt"
    note.write_text("root version")
    delegated = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Implement note update",
                        "instruction": "Update note.txt.",
                        "role": "implementer",
                        "mode": "isolated_write",
                        "scope": ".",
                    }
                ]
            },
        )
    )
    task = store.worker_tasks(run["id"])[0]

    assert delegated["ok"] is True
    assert note.read_text() == "root version"
    assert task["integration_status"] == "pending"
    assert task["change_set"][0]["action"] == "update"
    assert task["change_set"][0]["path"] == "note.txt"

    integrated = json.loads(
        await tools.execute("agent_integrate_worker", {"task_id": task["id"]})
    )
    integrated_task = store.worker_tasks(run["id"])[0]

    assert integrated["ok"] is True
    assert note.read_text() == "worker version"
    assert integrated_task["integration_status"] == "integrated"

    rolled_back = json.loads(
        await tools.execute("agent_rollback_worker", {"task_id": task["id"]})
    )
    rolled_back_task = store.worker_tasks(run["id"])[0]

    assert rolled_back["ok"] is True
    assert note.read_text() == "root version"
    assert rolled_back_task["integration_status"] == "rolled_back"


@pytest.mark.asyncio
async def test_implementer_integration_refuses_diverged_root_file(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write",
                    "write_file",
                    {"path": "note.txt", "content": "worker version"},
                ),
            )
        return httpx.Response(200, content=content_stream("Prepared update."))

    tools, store, run = worker_tools(tmp_path, handler)
    note = tools.file_tools.root / "note.txt"
    note.write_text("snapshot version")
    await tools.execute(
        "agent_delegate_tasks",
        {
            "tasks": [
                {
                    "title": "Implement update",
                    "instruction": "Update note.txt.",
                    "role": "implementer",
                    "mode": "isolated_write",
                }
            ]
        },
    )
    task = store.worker_tasks(run["id"])[0]
    note.write_text("new root version")

    result = json.loads(
        await tools.execute("agent_integrate_worker", {"task_id": task["id"]})
    )

    assert result["ok"] is False
    assert "conflicts" in result["error"]
    assert note.read_text() == "new root version"
    assert store.worker_tasks(run["id"])[0]["integration_status"] == "conflict"


@pytest.mark.asyncio
async def test_pending_implementer_changes_must_be_discarded_before_completion(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "create",
                    "create_file",
                    {"path": "draft.txt", "content": "private draft"},
                ),
            )
        return httpx.Response(200, content=content_stream("Draft prepared."))

    tools, store, run = worker_tools(tmp_path, handler)
    await tools.execute(
        "agent_delegate_tasks",
        {
            "tasks": [
                {
                    "title": "Prepare optional draft",
                    "instruction": "Create draft.txt.",
                    "role": "implementer",
                    "mode": "isolated_write",
                }
            ]
        },
    )
    task = store.worker_tasks(run["id"])[0]
    controls = AgentControlTools(store, run["id"])
    controls.execute(
        "agent_update_step",
        {"index": 0, "status": "completed"},
    )
    controls.execute("agent_begin_verification", {})
    receipt, _ = store.begin_tool_receipt(
        run["id"], "verify", "list_files", {"path": "."}, {"path": "."}, False
    )
    store.complete_tool_receipt(run["id"], "verify", '{"ok": true}', True)
    controls.execute(
        "agent_record_verification",
        {"evidence": "Root workspace inspected.", "receipt_id": receipt["id"]},
    )

    blocked = json.loads(
        controls.execute("agent_complete", {"summary": "Finished without draft."})
    )
    discarded = json.loads(
        await tools.execute("agent_discard_worker", {"task_id": task["id"]})
    )
    completed = json.loads(
        controls.execute("agent_complete", {"summary": "Finished without draft."})
    )

    assert blocked["ok"] is False
    assert "Integrate or discard" in blocked["error"]
    assert discarded["ok"] is True
    assert completed["ok"] is True
    assert not (tools.file_tools.root / "draft.txt").exists()
    assert store.worker_tasks(run["id"])[0]["integration_status"] == "discarded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_mode", "terminal_offered"),
    [("host", False), ("sandbox", True)],
)
async def test_implementer_never_receives_host_terminal(
    tmp_path,
    terminal_mode,
    terminal_offered,
):
    offered = set()

    def handler(request):
        payload = json.loads(request.content)
        offered.update(tool["function"]["name"] for tool in payload["tools"])
        return httpx.Response(200, content=content_stream("No changes needed."))

    tools, store, run = worker_tools(tmp_path, handler)
    tools.terminal_tools = TerminalTools(
        tools.file_tools.root,
        mode=terminal_mode,
        sandbox_image="sandbox:test",
        docker_executable=str(tmp_path / "fake-docker"),
    )
    await tools.execute(
        "agent_delegate_tasks",
        {
            "tasks": [
                {
                    "title": "Inspect implementation",
                    "instruction": "Check whether changes are needed.",
                    "role": "implementer",
                    "mode": "isolated_write",
                }
            ]
        },
    )

    assert ("run_command" in offered) is terminal_offered


@pytest.mark.asyncio
async def test_restart_reverts_integration_applied_before_state_commit(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write",
                    "write_file",
                    {"path": "note.txt", "content": "uncommitted integration"},
                ),
            )
        return httpx.Response(200, content=content_stream("Ready."))

    tools, store, run = worker_tools(tmp_path, handler)
    note = tools.file_tools.root / "note.txt"
    note.write_text("root version")
    await tools.execute(
        "agent_delegate_tasks",
        {
            "tasks": [
                {
                    "title": "Prepare update",
                    "instruction": "Update note.txt.",
                    "role": "implementer",
                    "mode": "isolated_write",
                }
            ]
        },
    )
    task = store.worker_tasks(run["id"])[0]
    tools.workspaces.integrate(run["id"], task)
    assert note.read_text() == "uncommitted integration"

    config = settings(tmp_path)
    AgentService(
        ModelGateway(config, httpx.MockTransport(handler)),
        FileTools(config.model_file_root),
        PdfTools(config.model_file_root),
        WebTools(),
        TerminalTools(config.model_file_root, mode="disabled"),
        StateTools(tmp_path / "memory.json"),
        ApprovalBroker(),
        AgentRunStore(config.agent_state_file),
        {
            "max_tool_rounds": 20,
            "max_tool_calls": 40,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
    )

    recovered_store = AgentRunStore(config.agent_state_file)
    recovered_task = recovered_store.worker_tasks(run["id"])[0]
    event_types = [event["type"] for event in recovered_store.events(run["id"])]
    assert note.read_text() == "root version"
    assert recovered_task["integration_status"] == "pending"
    assert "worker_integration_reverted_after_restart" in event_types


@pytest.mark.asyncio
async def test_task_graph_runs_ready_nodes_in_parallel_then_dependencies(tmp_path):
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        payload = json.loads(request.content)
        instruction = payload["messages"][1]["content"]
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.03)
        active -= 1
        if "Task: Combine findings" in instruction:
            assert "[alpha] Alpha report." in instruction
            assert "[beta] Beta report." in instruction
            return httpx.Response(200, content=content_stream("Combined report."))
        report = "Alpha report." if "Task: Inspect alpha" in instruction else "Beta report."
        return httpx.Response(200, content=content_stream(report))

    tools, store, run = worker_tools(tmp_path, handler)
    result = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "alpha",
                        "title": "Inspect alpha",
                        "instruction": "Inspect alpha independently.",
                        "role": "researcher",
                    },
                    {
                        "key": "beta",
                        "title": "Inspect beta",
                        "instruction": "Inspect beta independently.",
                        "role": "reviewer",
                    },
                    {
                        "key": "combine",
                        "title": "Combine findings",
                        "instruction": "Combine the dependency findings.",
                        "role": "tester",
                        "depends_on": ["alpha", "beta"],
                    },
                ]
            },
        )
    )

    assert result["ok"] is True
    assert result["graph"]["status"] == "completed"
    assert peak == 2
    assert {node["status"] for node in result["graph"]["nodes"].values()} == {
        "completed"
    }
    assert len(store.worker_tasks(run["id"])) == 3


@pytest.mark.asyncio
async def test_task_graph_waits_for_integration_then_unlocks_successor(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        instruction = payload["messages"][1]["content"]
        if "Task: Implement package" in instruction and calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write",
                    "write_file",
                    {"path": "note.txt", "content": "integrated version"},
                ),
            )
        if "Task: Implement package" in instruction:
            return httpx.Response(200, content=content_stream("Package updated."))
        assert (tools.file_tools.root / "pkg" / "note.txt").read_text() == "integrated version"
        return httpx.Response(200, content=content_stream("Integrated package verified."))

    tools, store, run = worker_tools(tmp_path, handler)
    package = tools.file_tools.root / "pkg"
    package.mkdir()
    note = package / "note.txt"
    note.write_text("root version")
    created = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "implement",
                        "title": "Implement package",
                        "instruction": "Update note.txt.",
                        "role": "implementer",
                        "mode": "isolated_write",
                        "scope": "pkg",
                    },
                    {
                        "key": "verify",
                        "title": "Verify package",
                        "instruction": "Inspect the integrated package.",
                        "role": "tester",
                        "scope": "pkg",
                        "depends_on": ["implement"],
                    },
                ]
            },
        )
    )
    implementer = store.worker_tasks(run["id"])[0]

    assert created["graph"]["status"] == "waiting_for_integration"
    assert created["graph"]["nodes"]["verify"]["status"] == "queued"
    assert note.read_text() == "root version"

    integrated = json.loads(
        await tools.execute("agent_integrate_worker", {"task_id": implementer["id"]})
    )

    assert integrated["ok"] is True
    assert integrated["graph"]["status"] == "completed"
    assert integrated["graph"]["nodes"]["implement"]["status"] == "integrated"
    assert integrated["graph"]["nodes"]["verify"]["status"] == "completed"
    assert len(store.worker_tasks(run["id"])) == 2


@pytest.mark.asyncio
async def test_task_graph_rejects_overlapping_independent_implementers(tmp_path):
    tools, store, run = worker_tools(
        tmp_path,
        lambda request: httpx.Response(200, content=content_stream("unused")),
    )
    (tools.file_tools.root / "src" / "api").mkdir(parents=True)

    result = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "src",
                        "title": "Implement source",
                        "instruction": "Change source.",
                        "role": "implementer",
                        "mode": "isolated_write",
                        "scope": "src",
                    },
                    {
                        "key": "api",
                        "title": "Implement API",
                        "instruction": "Change API.",
                        "role": "implementer",
                        "mode": "isolated_write",
                        "scope": "src/api",
                    },
                ]
            },
        )
    )

    assert result["ok"] is False
    assert "non-overlapping scopes" in result["error"]
    assert store.get(run["id"])["task_graph"] is None


@pytest.mark.asyncio
async def test_failed_task_graph_can_be_replanned_with_replacement_node(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=content_stream("" if calls == 1 else "Recovered report."),
        )

    tools, store, run = worker_tools(tmp_path, handler)
    created = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "broken",
                        "title": "Broken inspection",
                        "instruction": "Attempt inspection.",
                        "role": "researcher",
                    }
                ]
            },
        )
    )
    assert created["graph"]["status"] == "failed"
    controls = AgentControlTools(store, run["id"])
    controls.execute("agent_update_step", {"index": 0, "status": "completed"})
    premature_verification = json.loads(
        controls.execute("agent_begin_verification", {})
    )

    replanned = json.loads(
        await tools.execute(
            "agent_replan_task_graph",
            {
                "revision": 1,
                "reason": "Replace the failed approach.",
                "cancel_keys": ["broken"],
                "add_tasks": [
                    {
                        "key": "recovery",
                        "title": "Recovery inspection",
                        "instruction": "Use the revised approach.",
                        "role": "reviewer",
                    }
                ],
            },
        )
    )

    assert replanned["ok"] is True
    assert replanned["graph"]["revision"] == 2
    assert replanned["graph"]["status"] == "completed"
    assert replanned["graph"]["nodes"]["broken"]["status"] == "cancelled"
    assert replanned["graph"]["nodes"]["recovery"]["result"] == "Recovered report."
    verification = json.loads(controls.execute("agent_begin_verification", {}))
    assert premature_verification["ok"] is False
    assert "task graph" in premature_verification["error"]
    assert verification["ok"] is True


@pytest.mark.asyncio
async def test_conflicted_graph_worker_can_retry_from_fresh_snapshot(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            return httpx.Response(
                200,
                content=tool_stream(
                    f"write-{calls}",
                    "write_file",
                    {
                        "path": "note.txt",
                        "content": "first attempt" if calls == 1 else "retried version",
                    },
                ),
            )
        return httpx.Response(200, content=content_stream("Prepared update."))

    tools, store, run = worker_tools(tmp_path, handler)
    note = tools.file_tools.root / "note.txt"
    note.write_text("original")
    await tools.execute(
        "agent_create_task_graph",
        {
            "tasks": [
                {
                    "key": "implementation",
                    "title": "Implement update",
                    "instruction": "Update note.txt.",
                    "role": "implementer",
                    "mode": "isolated_write",
                }
            ]
        },
    )
    task = store.worker_tasks(run["id"])[0]
    note.write_text("new root")
    conflicted = json.loads(
        await tools.execute("agent_integrate_worker", {"task_id": task["id"]})
    )
    assert conflicted["ok"] is False
    assert store.get(run["id"])["task_graph"]["nodes"]["implementation"]["status"] == "conflict"

    retried = json.loads(
        await tools.execute(
            "agent_resolve_worker_conflict",
            {"task_id": task["id"], "action": "retry"},
        )
    )
    assert retried["ok"] is True
    assert retried["graph"]["status"] == "waiting_for_integration"
    assert note.read_text() == "new root"

    integrated = json.loads(
        await tools.execute("agent_integrate_worker", {"task_id": task["id"]})
    )
    assert integrated["graph"]["status"] == "completed"
    assert note.read_text() == "retried version"


@pytest.mark.asyncio
async def test_task_graph_shared_budget_stops_new_execution_waves(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "create",
                    "create_file",
                    {"path": "result.txt", "content": "ready"},
                ),
            )
        return httpx.Response(200, content=content_stream("Implementation ready."))

    tools, store, run = worker_tools(tmp_path, handler)
    await tools.execute(
        "agent_create_task_graph",
        {
            "tasks": [
                {
                    "key": "implement",
                    "title": "Implement result",
                    "instruction": "Create result.txt.",
                    "role": "implementer",
                    "mode": "isolated_write",
                },
                {
                    "key": "review",
                    "title": "Review result",
                    "instruction": "Review result.txt.",
                    "role": "reviewer",
                    "depends_on": ["implement"],
                },
            ]
        },
    )
    task = store.worker_tasks(run["id"])[0]

    def consume_round_budget(record):
        graph = record["task_graph"]
        graph["budgets"]["max_model_rounds"] = graph["usage"]["model_rounds"]

    store.mutate(run["id"], consume_round_budget)
    integrated = json.loads(
        await tools.execute("agent_integrate_worker", {"task_id": task["id"]})
    )

    assert integrated["graph"]["status"] == "budget_exhausted"
    assert integrated["graph"]["nodes"]["review"]["status"] == "blocked"
    assert len(store.worker_tasks(run["id"])) == 1


@pytest.mark.asyncio
async def test_cancelled_task_graph_persists_failed_node_and_elapsed_usage(tmp_path):
    started = asyncio.Event()

    async def handler(request):
        started.set()
        await asyncio.sleep(10)
        return httpx.Response(200, content=content_stream("too late"))

    tools, store, run = worker_tools(tmp_path, handler)
    execution = asyncio.create_task(
        tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "slow",
                        "title": "Slow graph task",
                        "instruction": "Wait for unavailable context.",
                        "role": "researcher",
                    }
                ]
            },
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    execution.cancel()
    with suppress(asyncio.CancelledError):
        await execution

    graph = store.get(run["id"])["task_graph"]
    task = store.worker_tasks(run["id"])[0]
    assert task["status"] == "stopped"
    assert graph["nodes"]["slow"]["status"] == "failed"
    assert graph["usage"]["model_rounds"] == 1
    assert graph["usage"]["elapsed_seconds"] >= 1


@pytest.mark.asyncio
async def test_worker_review_returns_stable_bounded_unified_diff(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write",
                    "write_file",
                    {"path": "note.txt", "content": "first\nchanged\nthird\n"},
                ),
            )
        return httpx.Response(200, content=content_stream("Prepared review."))

    tools, store, run = worker_tools(tmp_path, handler)
    (tools.file_tools.root / "note.txt").write_text("first\nsecond\nthird\n")
    delegated = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Edit note",
                        "instruction": "Update the second line.",
                        "role": "implementer",
                        "mode": "isolated_write",
                    }
                ]
            },
        )
    )
    task_id = delegated["tasks"][0]["id"]

    first = json.loads(
        await tools.execute("agent_review_worker", {"task_id": task_id})
    )["review"]
    second = json.loads(
        await tools.execute("agent_review_worker", {"task_id": task_id})
    )["review"]

    assert first["files"][0]["diff"].startswith("--- a/note.txt\n+++ b/note.txt")
    assert "-second" in first["files"][0]["diff"]
    assert "+changed" in first["files"][0]["diff"]
    assert first["files"][0]["hunks"][0]["id"] == second["files"][0]["hunks"][0]["id"]
    assert first["conflicts"] == []


@pytest.mark.asyncio
async def test_review_can_accept_one_file_and_reject_another(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write-a",
                    "write_file",
                    {"path": "a.txt", "content": "worker a"},
                ),
            )
        if calls == 2:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write-b",
                    "write_file",
                    {"path": "b.txt", "content": "worker b"},
                ),
            )
        return httpx.Response(200, content=content_stream("Both files prepared."))

    tools, store, run = worker_tools(tmp_path, handler)
    first = tools.file_tools.root / "a.txt"
    second = tools.file_tools.root / "b.txt"
    first.write_text("root a")
    second.write_text("root b")
    delegated = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Edit files",
                        "instruction": "Update a.txt and b.txt.",
                        "role": "implementer",
                        "mode": "isolated_write",
                    }
                ]
            },
        )
    )
    task_id = delegated["tasks"][0]["id"]

    integrated = json.loads(
        await tools.execute(
            "agent_integrate_worker",
            {"task_id": task_id, "accepted_paths": ["a.txt"], "accepted_hunks": {}},
        )
    )
    task = store.worker_tasks(run["id"])[0]

    assert integrated["ok"] is True
    assert first.read_text() == "worker a"
    assert second.read_text() == "root b"
    assert task["review_decision"]["accepted_paths"] == ["a.txt"]
    assert task["review_decision"]["rejected_paths"] == ["b.txt"]
    assert [change["path"] for change in task["change_set"]] == ["a.txt"]


@pytest.mark.asyncio
async def test_review_can_integrate_and_rollback_one_selected_hunk(tmp_path):
    calls = 0
    original_lines = [f"line {index}\n" for index in range(1, 21)]
    changed_lines = list(original_lines)
    changed_lines[1] = "first selected change\n"
    changed_lines[17] = "second rejected change\n"

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write",
                    "write_file",
                    {"path": "note.txt", "content": "".join(changed_lines)},
                ),
            )
        return httpx.Response(200, content=content_stream("Two hunks prepared."))

    tools, store, run = worker_tools(tmp_path, handler)
    note = tools.file_tools.root / "note.txt"
    note.write_text("".join(original_lines))
    delegated = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Edit distant lines",
                        "instruction": "Update two distant lines.",
                        "role": "implementer",
                        "mode": "isolated_write",
                    }
                ]
            },
        )
    )
    task_id = delegated["tasks"][0]["id"]
    review = json.loads(
        await tools.execute("agent_review_worker", {"task_id": task_id})
    )["review"]
    hunks = review["files"][0]["hunks"]
    assert len(hunks) == 2

    integrated = json.loads(
        await tools.execute(
            "agent_integrate_worker",
            {
                "task_id": task_id,
                "accepted_paths": [],
                "accepted_hunks": {"note.txt": [hunks[0]["id"]]},
            },
        )
    )
    expected = list(original_lines)
    expected[1] = "first selected change\n"

    assert integrated["ok"] is True
    assert note.read_text() == "".join(expected)
    task = store.worker_tasks(run["id"])[0]
    assert task["review_decision"]["accepted_hunks"] == {
        "note.txt": [hunks[0]["id"]]
    }

    rolled_back = json.loads(
        await tools.execute("agent_rollback_worker", {"task_id": task_id})
    )
    assert rolled_back["ok"] is True
    assert note.read_text() == "".join(original_lines)


@pytest.mark.asyncio
async def test_review_rejects_stale_hunk_id_without_root_changes(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write",
                    "write_file",
                    {"path": "note.txt", "content": "worker"},
                ),
            )
        return httpx.Response(200, content=content_stream("Prepared."))

    tools, store, run = worker_tools(tmp_path, handler)
    note = tools.file_tools.root / "note.txt"
    note.write_text("root")
    delegated = json.loads(
        await tools.execute(
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Edit note",
                        "instruction": "Update note.txt.",
                        "role": "implementer",
                        "mode": "isolated_write",
                    }
                ]
            },
        )
    )
    task_id = delegated["tasks"][0]["id"]

    rejected = json.loads(
        await tools.execute(
            "agent_integrate_worker",
            {
                "task_id": task_id,
                "accepted_paths": [],
                "accepted_hunks": {"note.txt": ["stale-hunk"]},
            },
        )
    )

    assert rejected["ok"] is False
    assert "stale or unknown" in rejected["error"]
    assert note.read_text() == "root"
    assert store.worker_tasks(run["id"])[0]["integration_status"] == "pending"


@pytest.mark.asyncio
async def test_restart_recovers_only_unrecorded_selected_file_integration(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write-a", "write_file", {"path": "a.txt", "content": "worker a"}
                ),
            )
        if calls == 2:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write-b", "write_file", {"path": "b.txt", "content": "worker b"}
                ),
            )
        return httpx.Response(200, content=content_stream("Prepared."))

    tools, store, run = worker_tools(tmp_path, handler)
    first = tools.file_tools.root / "a.txt"
    second = tools.file_tools.root / "b.txt"
    first.write_text("root a")
    second.write_text("root b")
    await tools.execute(
        "agent_delegate_tasks",
        {
            "tasks": [
                {
                    "title": "Edit files",
                    "instruction": "Update both files.",
                    "role": "implementer",
                    "mode": "isolated_write",
                }
            ]
        },
    )
    task = store.worker_tasks(run["id"])[0]
    selected, staged, _ = tools.workspaces.integration_selection(
        run["id"], task, ["a.txt"], {}
    )
    tools.workspaces.integrate(
        run["id"], task, selected_changes=selected, staged_workspace=staged
    )
    assert first.read_text() == "worker a"
    assert second.read_text() == "root b"

    config = settings(tmp_path)
    AgentService(
        ModelGateway(config, httpx.MockTransport(handler)),
        FileTools(config.model_file_root),
        PdfTools(config.model_file_root),
        WebTools(),
        TerminalTools(config.model_file_root, mode="disabled"),
        StateTools(tmp_path / "memory.json"),
        ApprovalBroker(),
        AgentRunStore(config.agent_state_file),
        {
            "max_tool_rounds": 20,
            "max_tool_calls": 40,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
    )

    assert first.read_text() == "root a"
    assert second.read_text() == "root b"
    assert AgentRunStore(config.agent_state_file).worker_tasks(run["id"])[0][
        "integration_status"
    ] == "pending"


@pytest.mark.asyncio
async def test_rich_graph_conditional_branches_use_typed_artifacts(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=content_stream(
                    json.dumps(
                        {
                            "report": "Decision produced.",
                            "artifacts": [
                                {"name": "proceed", "type": "json", "content": True}
                            ],
                        }
                    )
                ),
            )
        return httpx.Response(200, content=content_stream("Matched branch ran."))

    tools, store, run = worker_tools(tmp_path, handler)
    result = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "decide",
                        "title": "Produce decision",
                        "instruction": "Return a proceed artifact.",
                        "role": "researcher",
                    },
                    {
                        "key": "yes",
                        "title": "Run selected branch",
                        "instruction": "Handle the selected branch.",
                        "role": "reviewer",
                        "depends_on": ["decide"],
                        "condition": {
                            "source": "decide",
                            "operator": "artifact_truthy",
                            "artifact": "proceed",
                        },
                    },
                    {
                        "key": "no",
                        "title": "Run rejected branch",
                        "instruction": "Handle the rejected branch.",
                        "role": "reviewer",
                        "depends_on": ["decide"],
                        "condition": {
                            "source": "decide",
                            "operator": "artifact_equals",
                            "artifact": "proceed",
                            "value": False,
                        },
                    },
                ]
            },
        )
    )

    graph = result["graph"]
    assert graph["status"] == "completed"
    assert graph["nodes"]["yes"]["status"] == "completed"
    assert graph["nodes"]["no"]["status"] == "skipped"
    assert graph["nodes"]["decide"]["artifacts"][0]["name"] == "proceed"
    artifact = json.loads(
        await tools.execute(
            "agent_read_graph_artifact",
            {"key": "decide", "name": "proceed"},
        )
    )
    assert artifact["artifact"]["content"] is True
    assert "untrusted" in artifact["artifact"]["security"]
    public = AgentRunStore.public(store.get(run["id"]))
    assert "content" not in public["task_graph"]["nodes"]["decide"]["artifacts"][0]
    assert "content" not in public["worker_tasks"][0]["artifacts"][0]


@pytest.mark.asyncio
async def test_rich_graph_map_fans_out_and_adds_reduce_join(tmp_path):
    instructions = []

    def handler(request):
        payload = json.loads(request.content)
        instruction = payload["messages"][1]["content"]
        instructions.append(instruction)
        if "Discover modules" in instruction:
            report = {
                "report": "Two modules discovered.",
                "artifacts": [
                    {"name": "modules", "type": "json", "content": ["alpha", "beta"]}
                ],
            }
            return httpx.Response(200, content=content_stream(json.dumps(report)))
        if "Synthesize module reports" in instruction:
            return httpx.Response(200, content=content_stream("Reduced both reports."))
        return httpx.Response(200, content=content_stream("Mapped item inspected."))

    tools, store, run = worker_tools(tmp_path, handler)
    result = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "modules",
                        "kind": "map",
                        "title": "Discover modules",
                        "instruction": "Discover modules that need inspection.",
                        "role": "researcher",
                        "map": {
                            "source_artifact": "modules",
                            "max_items": 4,
                            "item_template": {
                                "title": "Inspect {item}",
                                "instruction": "Inspect module {item} at index {index}.",
                                "role": "reviewer",
                            },
                            "reduce": {
                                "title": "Synthesize modules",
                                "instruction": "Synthesize module reports.",
                                "role": "reviewer",
                            },
                        },
                    }
                ]
            },
        )
    )

    graph = result["graph"]
    assert graph["status"] == "completed"
    assert graph["nodes"]["modules"]["map_children"] == [
        "modules_item_1",
        "modules_item_2",
        "modules_reduce",
    ]
    assert graph["nodes"]["modules_reduce"]["kind"] == "join"
    assert graph["nodes"]["modules_reduce"]["status"] == "completed"
    assert any('"alpha"' in instruction for instruction in instructions)
    assert any('"beta"' in instruction for instruction in instructions)


@pytest.mark.asyncio
async def test_first_success_join_cancels_queued_sibling(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=content_stream(
                "First candidate succeeded." if calls == 1 else "Joined winner."
            ),
        )

    tools, store, run = worker_tools(tmp_path, handler)
    result = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "max_concurrency": 1,
                "tasks": [
                    {
                        "key": "candidate_a",
                        "title": "Try candidate A",
                        "instruction": "Attempt candidate A.",
                        "role": "researcher",
                    },
                    {
                        "key": "candidate_b",
                        "title": "Try candidate B",
                        "instruction": "Attempt candidate B.",
                        "role": "researcher",
                    },
                    {
                        "key": "winner",
                        "kind": "join",
                        "title": "Use first success",
                        "instruction": "Summarize the first successful candidate.",
                        "role": "reviewer",
                        "depends_on": ["candidate_a", "candidate_b"],
                        "join": {
                            "strategy": "first_success",
                            "cancel_remaining": True,
                        },
                    },
                ],
            },
        )
    )

    graph = result["graph"]
    assert graph["status"] == "completed"
    assert graph["nodes"]["candidate_a"]["status"] == "completed"
    assert graph["nodes"]["candidate_b"]["status"] == "cancelled"
    assert graph["nodes"]["winner"]["status"] == "completed"
    assert calls == 2


@pytest.mark.asyncio
async def test_bounded_loop_retries_until_artifact_condition_matches(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        result = {
            "report": f"Iteration {calls}.",
            "artifacts": [
                {"name": "done", "type": "json", "content": calls >= 2}
            ],
        }
        return httpx.Response(200, content=content_stream(json.dumps(result)))

    tools, store, run = worker_tools(tmp_path, handler)
    result = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "poll",
                        "kind": "loop",
                        "title": "Poll bounded state",
                        "instruction": "Inspect whether the state is done.",
                        "role": "researcher",
                        "loop": {
                            "max_iterations": 3,
                            "until": {
                                "operator": "artifact_truthy",
                                "artifact": "done",
                            },
                        },
                    }
                ]
            },
        )
    )

    node = result["graph"]["nodes"]["poll"]
    assert result["graph"]["status"] == "completed"
    assert node["loop_satisfied"] is True
    assert [item["matched"] for item in node["loop_iterations"]] == [False, True]
    assert len(store.worker_tasks(run["id"])) == 2


@pytest.mark.asyncio
async def test_node_model_budgets_rerun_replace_and_templates(tmp_path):
    models = []
    instructions = []

    def handler(request):
        payload = json.loads(request.content)
        models.append(payload["model"])
        instructions.append(payload["messages"][1]["content"])
        return httpx.Response(200, content=content_stream(f"Result {len(models)}."))

    tools, store, run = worker_tools(tmp_path, handler)
    created = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "custom",
                        "title": "Custom execution",
                        "instruction": "Use the custom execution policy.",
                        "role": "researcher",
                        "model": "special-model",
                        "budgets": {
                            "max_model_rounds": 2,
                            "max_tool_calls": 3,
                            "max_seconds": 30,
                        },
                    }
                ]
            },
        )
    )
    rerun = json.loads(
        await tools.execute(
            "agent_control_graph_node",
            {
                "key": "custom",
                "action": "rerun",
                "reason": "Confirm the result independently.",
            },
        )
    )
    replaced = json.loads(
        await tools.execute(
            "agent_replace_graph_node",
            {
                "key": "custom",
                "reason": "Use a reviewer for the final attempt.",
                "replacement": {
                    "key": "custom",
                    "title": "Replacement review",
                    "instruction": "Review the replacement path.",
                    "role": "reviewer",
                },
            },
        )
    )
    templates = json.loads(await tools.execute("agent_list_graph_templates", {}))

    assert created["graph"]["nodes"]["custom"]["node_budgets"]["max_tool_calls"] == 3
    assert rerun["graph"]["nodes"]["custom"]["attempt_count"] == 2
    assert replaced["graph"]["nodes"]["custom"]["title"] == "Replacement review"
    assert replaced["graph"]["nodes"]["custom"]["attempt_count"] == 3
    assert replaced["graph"]["usage"]["model_rounds"] == 3
    assert models[:2] == ["special-model", "special-model"]
    assert models[2] == "model-test"
    assert {item["name"] for item in templates["templates"]} >= {
        "parallel_analysis",
        "research_implement_review",
        "test_fix_verify",
    }
    assert all(item["version"] == 1 for item in templates["templates"])
    assert all("focus" in item["parameters"] for item in templates["templates"])


@pytest.mark.asyncio
async def test_rerun_and_replace_without_cascade_reject_completed_descendants(tmp_path):
    def handler(request):
        return httpx.Response(200, content=content_stream("Complete."))

    tools, store, run = worker_tools(tmp_path, handler)
    created = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "source",
                        "title": "Source",
                        "instruction": "Produce the source result.",
                        "role": "researcher",
                    },
                    {
                        "key": "consumer",
                        "title": "Consumer",
                        "instruction": "Consume the source result.",
                        "role": "reviewer",
                        "depends_on": ["source"],
                    },
                ]
            },
        )
    )
    rerun = json.loads(
        await tools.execute(
            "agent_control_graph_node",
            {
                "key": "source",
                "action": "rerun",
                "reason": "Try without invalidating the consumer.",
                "cascade": False,
            },
        )
    )
    replacement = json.loads(
        await tools.execute(
            "agent_replace_graph_node",
            {
                "key": "source",
                "reason": "Try replacement without invalidating the consumer.",
                "cascade": False,
                "replacement": {
                    "key": "source",
                    "title": "Replacement source",
                    "instruction": "Produce a replacement source result.",
                    "role": "researcher",
                },
            },
        )
    )

    assert created["graph"]["status"] == "completed"
    assert rerun["ok"] is False
    assert "requires cascade" in rerun["error"]
    assert replacement["ok"] is False
    assert "requires cascade" in replacement["error"]


@pytest.mark.asyncio
async def test_graph_node_can_pause_across_integration_barrier_and_resume(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "write",
                    "create_file",
                    {"path": "result.txt", "content": "ready"},
                ),
            )
        return httpx.Response(
            200,
            content=content_stream(
                "Implementation ready." if calls == 2 else "Review complete."
            ),
        )

    tools, store, run = worker_tools(tmp_path, handler)
    created = json.loads(
        await tools.execute(
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "implement",
                        "title": "Create result",
                        "instruction": "Create result.txt.",
                        "role": "implementer",
                        "mode": "isolated_write",
                    },
                    {
                        "key": "review",
                        "title": "Review result",
                        "instruction": "Review the integrated result.",
                        "role": "reviewer",
                        "depends_on": ["implement"],
                    },
                ]
            },
        )
    )
    assert created["graph"]["status"] == "waiting_for_integration"

    paused = json.loads(
        await tools.execute(
            "agent_control_graph_node",
            {"key": "review", "action": "pause", "reason": "Wait for human review."},
        )
    )
    task_id = created["graph"]["nodes"]["implement"]["task_id"]
    integrated = json.loads(
        await tools.execute("agent_integrate_worker", {"task_id": task_id})
    )
    resumed = json.loads(
        await tools.execute(
            "agent_control_graph_node",
            {"key": "review", "action": "resume", "reason": "Human review finished."},
        )
    )

    assert paused["graph"]["nodes"]["review"]["status"] == "paused"
    assert integrated["graph"]["status"] == "paused"
    assert resumed["graph"]["status"] == "completed"
    assert resumed["graph"]["nodes"]["review"]["status"] == "completed"


@pytest.mark.asyncio
async def test_graph_can_be_created_from_builtin_template(tmp_path):
    def handler(request):
        return httpx.Response(200, content=content_stream("Template worker complete."))

    tools, store, run = worker_tools(tmp_path, handler)
    result = json.loads(
        await tools.execute(
            "agent_create_graph_from_template",
            {
                "template": "parallel_analysis",
                "template_version": 1,
                "parameters": {"focus": "operator controls"},
                "max_concurrency": 2,
            },
        )
    )

    assert result["graph"]["status"] == "completed"
    assert result["graph"]["template"] == "parallel_analysis"
    assert result["graph"]["template_version"] == 1
    assert result["graph"]["template_parameters"] == {"focus": "operator controls"}
    assert '"operator controls"' in result["graph"]["nodes"]["research"]["instruction"]
    assert result["graph"]["nodes"]["synthesis"]["kind"] == "join"
