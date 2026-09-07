import asyncio
import json
import time

import httpx
import pytest

from app.agent_loop import AgentService
from app.agent_runs import AgentRunStore
from app.approvals import ApprovalBroker
from app.config import Settings
from app.file_tools import FileTools
from app.main import create_app
from app.model_gateway import ModelGateway
from app.pdf_tools import PdfTools
from app.schemas import AgentCloneRequest, AgentStartRequest, AgentSteerRequest
from app.state_tools import StateTools
from app.terminal_tools import TerminalTools
from app.web_tools import WebTools


def tool_stream(call_id, name, arguments):
    first = {
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
    finished = {
        "choices": [{"delta": {}, "finish_reason": "tool_calls"}]
    }
    return (
        f"data: {json.dumps(first)}\n\n"
        f"data: {json.dumps(finished)}\n\n"
        "data: [DONE]\n\n"
    ).encode()


def lifecycle_responses():
    return [
        ("plan", "agent_set_plan", {"steps": ["Implement and inspect"]}),
        (
            "step",
            "agent_update_step",
            {"index": 0, "status": "completed", "note": "Implemented"},
        ),
        ("verify", "agent_begin_verification", {}),
        ("inspect", "list_files", {"path": "."}),
        (
            "evidence",
            "agent_record_verification",
            {"evidence": "Inspection confirmed the expected result."},
        ),
        ("complete", "agent_complete", {"summary": "Goal completed."}),
    ]


def tool_stream_with_usage(call_id, name, arguments, *, prompt_tokens=10, completion_tokens=5):
    stream = tool_stream(call_id, name, arguments).decode()
    usage = json.dumps(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )
    return stream.replace(
        "data: [DONE]\n\n",
        f"data: {usage}\n\ndata: [DONE]\n\n",
    ).encode()


def agent_settings(tmp_path, **overrides):
    values = {
        "model_base_url": "http://model.test",
        "model_name": "model-test",
        "model_api_key": None,
        "model_file_root": tmp_path / "workspace",
        "agent_state_file": tmp_path / "agent.json",
        "terminal_mode": "disabled",
    }
    values.update(overrides)
    return Settings(**values)


def operator_service(tmp_path):
    settings = agent_settings(tmp_path)
    workspace = settings.model_file_root
    workspace.mkdir(exist_ok=True)
    return AgentService(
        ModelGateway(settings, httpx.MockTransport(lambda request: httpx.Response(500))),
        FileTools(workspace),
        PdfTools(workspace),
        WebTools(),
        TerminalTools(workspace, mode="disabled"),
        StateTools(tmp_path / "memory.json"),
        ApprovalBroker(),
        AgentRunStore(tmp_path / "agent.json"),
        {
            "max_tool_rounds": 10,
            "max_tool_calls": 20,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
    )


@pytest.mark.asyncio
async def test_agent_api_runs_lifecycle_and_persists_public_state(tmp_path):
    requests = []
    responses = lifecycle_responses()
    responses[3] = ("inspect", "read_file", {"path": "note.txt"})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("expected workspace evidence")

    def model_handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        call_id, name, arguments = responses[len(requests) - 1]
        return httpx.Response(200, content=tool_stream(call_id, name, arguments))

    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/agent/runs",
            json={"goal": "Implement the feature", "model": "model-test"},
        )
        current = await client.get("/api/agent/runs/current")
        history = await client.get("/api/agent/runs")
        events = await client.get(
            f"/api/agent/runs/{current.json()['id']}/events",
            params={"after": 1},
        )
        remembered = await client.get(
            "/api/memory/search",
            params={"q": "Implement the feature", "kinds": "run"},
        )

    assert response.status_code == 200
    assert '"harness_event":"agent_run"' in response.text
    assert '"harness_event":"agent_finished"' in response.text
    assert '"status":"completed"' in response.text
    public = current.json()
    assert public["status"] == "completed"
    assert public["summary"] == "Goal completed."
    assert public["verification_evidence"][0]["text"] == (
        "Inspection confirmed the expected result."
    )
    assert public["verification_evidence"][0]["tool"] == "read_file"
    assert public["verification_evidence"][0]["receipt_id"]
    assert public["event_count"] > 0
    assert len(public["tool_receipts"]) == 1
    assert history.json()[0]["id"] == public["id"]
    assert events.json()[0]["seq"] == 2
    assert remembered.json()["results"][0]["provenance"]["run_id"] == public["id"]
    assert "messages" not in public
    offered_tools = {
        tool["function"]["name"] for tool in requests[0]["tools"]
    }
    assert {
        "agent_set_plan",
        "agent_update_step",
        "agent_begin_verification",
        "agent_record_verification",
        "agent_complete",
        "agent_block",
        "search_memory",
        "read_memory",
        "save_lesson",
    } <= offered_tools
    assert any(
        message.get("role") == "tool"
        and "expected workspace evidence" in message.get("content", "")
        for message in requests[4]["messages"]
    )


@pytest.mark.asyncio
async def test_observability_api_accounts_usage_and_replays_regression_scenario(tmp_path):
    requests = []
    responses = lifecycle_responses() + lifecycle_responses()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def model_handler(request):
        requests.append(json.loads(request.content))
        call_id, name, arguments = responses[len(requests) - 1]
        return httpx.Response(
            200,
            content=tool_stream_with_usage(call_id, name, arguments),
        )

    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(
            tmp_path,
            model_input_cost_per_million=2.0,
            model_output_cost_per_million=4.0,
        ),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        source_response = await client.post(
            "/api/agent/runs",
            json={"goal": "Replay this regression", "model": "model-test"},
        )
        source = (await client.get("/api/agent/runs/current")).json()
        trace = (await client.get(f"/api/agent/runs/{source['id']}/trace")).json()
        evaluation = (
            await client.get(f"/api/agent/runs/{source['id']}/evaluation")
        ).json()
        created = await client.post(
            f"/api/evals/scenarios/from-run/{source['id']}",
            json={"name": "Lifecycle regression"},
        )
        scenario = created.json()
        replay_response = await client.post(
            f"/api/evals/scenarios/{scenario['id']}/replay"
        )
        reloaded = (
            await client.get(f"/api/evals/scenarios/{scenario['id']}")
        ).json()
        replay_id = reloaded["replays"][-1]["run_id"]
        await client.get(f"/api/agent/runs/{replay_id}/trace")
        await client.get(f"/api/agent/runs/{replay_id}/trace")
        after_trace_reads = (
            await client.get(f"/api/evals/scenarios/{scenario['id']}")
        ).json()

    assert source_response.status_code == 200
    assert trace["usage"]["input_tokens"] == 60
    assert trace["usage"]["output_tokens"] == 30
    assert trace["usage"]["provider_token_calls"] == 6
    assert trace["usage"]["estimated_token_calls"] == 0
    assert trace["usage"]["cost_usd"] == 0.00024
    assert len(trace["spans"]) == 13
    assert evaluation["quality_score"] == 100
    assert evaluation["safety_score"] == 100
    assert created.status_code == 200
    assert replay_response.status_code == 200
    assert reloaded["replays"][-1]["assertions"]["passed"] is True
    assert len(after_trace_reads["replays"]) == 1
    assert requests[0]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["Still thinking", ""])
async def test_agent_exhausts_round_budget_when_model_does_not_use_lifecycle(tmp_path, content):
    requests = []

    def model_handler(request):
        messages = json.loads(request.content)["messages"]
        requests.append(messages)
        assert messages[0]["role"] == "system"
        if len(requests) == 2:
            assert "The run is still active" in messages[0]["content"]
            assert [message["role"] for message in messages[1:]] == ["user", "assistant"]
            assert messages[-1]["content"] == content
        payload = {"choices": [{"delta": {"content": content}}]}
        stream = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()
        return httpx.Response(200, content=stream)

    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(tmp_path, agent_max_tool_rounds=2),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/agent/runs",
            json={"goal": "Keep thinking", "model": "model-test"},
        )
        current = (await client.get("/api/agent/runs/current")).json()

    assert response.status_code == 200
    assert current["status"] == "budget_exhausted"
    assert current["usage"]["tool_rounds"] == 2
    assert len(requests) == 2
    assert "2 model/tool rounds" in current["summary"]
    saved = AgentRunStore(tmp_path / "agent.json").get(current["id"])
    assert all(message["content"] == content for message in saved["messages"] if message["role"] == "assistant")


@pytest.mark.asyncio
async def test_failed_agent_run_can_be_manually_resumed(tmp_path):
    calls = 0
    responses = lifecycle_responses()

    def model_handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporarily offline", request=request)
        messages = json.loads(request.content)["messages"]
        assert messages[0]["role"] == "system"
        assert "resumed manually" in messages[0]["content"]
        assert all(message["role"] not in {"system", "developer"} for message in messages[1:])
        call_id, name, arguments = responses[calls - 2]
        return httpx.Response(200, content=tool_stream(call_id, name, arguments))

    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        failed = await client.post(
            "/api/agent/runs",
            json={"goal": "Recoverable task", "model": "model-test"},
        )
        stopped = (await client.get("/api/agent/runs/current")).json()
        resumed = await client.post(
            f"/api/agent/runs/{stopped['id']}/resume",
            json={"instruction": "The model is available; continue."},
        )
        completed = (await client.get("/api/agent/runs/current")).json()

    assert '"status":"failed"' in failed.text
    assert stopped["resumable"] is True
    assert resumed.status_code == 200
    assert completed["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_can_be_stopped_while_waiting_for_approval(tmp_path):
    calls = 0

    def model_handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=tool_stream(
                    "plan",
                    "agent_set_plan",
                    {"steps": ["Create the file"]},
                ),
            )
        return httpx.Response(
            200,
            content=tool_stream(
                "write",
                "create_file",
                {"path": "note.txt", "content": "agent output"},
            ),
        )

    workspace = tmp_path / "workspace"
    settings = agent_settings(tmp_path)
    approvals = ApprovalBroker()
    service = AgentService(
        ModelGateway(settings, httpx.MockTransport(model_handler)),
        FileTools(workspace),
        PdfTools(workspace),
        WebTools(),
        TerminalTools(workspace, mode="disabled"),
        StateTools(tmp_path / "memory.json"),
        approvals,
        AgentRunStore(tmp_path / "agent.json"),
        {
            "max_tool_rounds": 10,
            "max_tool_calls": 20,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
    )
    execution = await service.create(
        AgentStartRequest(goal="Create a note", model="model-test")
    )
    events = execution.events()
    approval_seen = False
    async for raw_event in events:
        event = raw_event.decode()
        approval_seen = '"harness_event":"tool_approval"' in event
        if approval_seen:
            break
    assert approval_seen, service.current()["last_error"]

    await service.stop(execution.run_id)
    remaining = [event.decode() async for event in events]

    run = service.current()
    assert run["status"] == "stopped"
    assert any('"harness_event":"agent_finished"' in event for event in remaining)
    assert not (workspace / "note.txt").exists()


@pytest.mark.asyncio
async def test_elapsed_budget_interrupts_a_waiting_model_request(tmp_path):
    async def model_handler(request):
        await asyncio.sleep(5)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(tmp_path, agent_max_seconds=1),
    )
    transport = httpx.ASGITransport(app=app)
    started = time.monotonic()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/agent/runs",
            json={"goal": "Wait forever", "model": "model-test"},
        )
        current = (await client.get("/api/agent/runs/current")).json()

    assert response.status_code == 200
    assert current["status"] == "budget_exhausted"
    assert time.monotonic() - started < 3


@pytest.mark.asyncio
async def test_workspace_grant_bypasses_only_mutation_approval(tmp_path):
    requests = []
    responses = [
        ("plan", "agent_set_plan", {"steps": ["Create and verify the note"]}),
        (
            "create",
            "create_file",
            {"path": "note.txt", "content": "created by agent"},
        ),
        (
            "step",
            "agent_update_step",
            {"index": 0, "status": "completed"},
        ),
        ("verify", "agent_begin_verification", {}),
        ("read", "read_file", {"path": "note.txt"}),
        ("evidence", "agent_record_verification", {"evidence": "Note content verified."}),
        ("complete", "agent_complete", {"summary": "Created and verified."}),
    ]

    def model_handler(request):
        requests.append(json.loads(request.content))
        call_id, name, arguments = responses[len(requests) - 1]
        return httpx.Response(200, content=tool_stream(call_id, name, arguments))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/agent/runs",
            json={
                "goal": "Create a note",
                "model": "model-test",
                "grants": ["workspace_mutations"],
            },
        )
        run = (await client.get("/api/agent/runs/current")).json()
        events = (
            await client.get(f"/api/agent/runs/{run['id']}/events")
        ).json()

    assert response.status_code == 200
    assert run["status"] == "completed"
    assert run["grants"] == ["workspace_mutations"]
    assert (workspace / "note.txt").read_text() == "created by agent"
    event_types = [event["type"] for event in events]
    assert "capability_grant_used" in event_types
    assert "approval_requested" not in event_types


@pytest.mark.asyncio
async def test_terminal_grant_is_rejected_when_terminal_is_disabled(tmp_path):
    app = create_app(
        model_transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        settings=agent_settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/agent/runs",
            json={"goal": "Run a command", "grants": ["terminal"]},
        )

    assert response.status_code == 409
    assert "disabled" in response.json()["detail"]


@pytest.mark.asyncio
async def test_operator_clone_compare_and_debug_are_bounded_and_public(tmp_path):
    service = operator_service(tmp_path)
    execution = await service.create(
        AgentStartRequest(goal="Inspect operator behavior", model="model-test")
    )

    def stop(record):
        record["status"] = "stopped"
        record["summary"] = "Stopped for operator inspection."

    service.store.mutate(execution.run_id, stop, event_type="run_stopped_for_test")
    await service.release(execution.run_id)
    clone = await service.clone(
        execution.run_id,
        AgentCloneRequest(goal="Inspect cloned operator behavior"),
    )
    comparison = service.compare(execution.run_id, clone["id"])
    debug = service.debug(clone["id"])

    assert clone["status"] == "stopped"
    assert clone["resumable"] is True
    assert clone["parent_run_id"] == execution.run_id
    assert clone["worker_tasks"] == []
    assert "goal" in {item["field"] for item in comparison["differences"]}
    assert comparison["node_differences"] == []
    assert debug["diagnostics"]["state_validated"] is True
    assert debug["diagnostics"]["open_receipts"] == []
    assert debug["recent_events"][-1]["type"] == "run_cloned"
    assert all("result" not in item for item in debug["run"]["tool_receipts"])


@pytest.mark.asyncio
async def test_active_run_accepts_steering_at_the_next_model_boundary(tmp_path):
    requests = []
    responses = [
        ("plan", "agent_set_plan", {"steps": ["Investigate"]}),
        (
            "blocked",
            "agent_block",
            {"reason": "Paused after applying guidance", "requires_user_input": True},
        ),
    ]

    def model_handler(request):
        requests.append(json.loads(request.content))
        call_id, name, arguments = responses[len(requests) - 1]
        return httpx.Response(200, content=tool_stream(call_id, name, arguments))

    settings = agent_settings(tmp_path)
    service = AgentService(
        ModelGateway(settings, httpx.MockTransport(model_handler)),
        FileTools(settings.model_file_root),
        PdfTools(settings.model_file_root),
        WebTools(),
        TerminalTools(settings.model_file_root, mode="disabled"),
        StateTools(tmp_path / "memory.json"),
        ApprovalBroker(),
        AgentRunStore(tmp_path / "agent.json"),
        {
            "max_tool_rounds": 10,
            "max_tool_calls": 20,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
    )
    execution = await service.create(
        AgentStartRequest(goal="Investigate the issue", model="model-test")
    )
    stream = execution.events()
    async for event in stream:
        if '"control":"agent_set_plan"' in event.decode():
            await service.steer(
                execution.run_id,
                AgentSteerRequest(instruction="Focus on the parser first."),
            )
            break
    remaining = [event async for event in stream]
    events = service.events(execution.run_id)

    assert remaining
    assert any(
        message.get("role") == "user"
        and "Focus on the parser first." in message.get("content", "")
        for message in requests[1]["messages"]
    )
    assert [event["type"] for event in events].count("steering_queued") == 1
    assert [event["type"] for event in events].count("steering_applied") == 1


@pytest.mark.asyncio
async def test_uncertain_side_effect_is_not_reexecuted_after_interruption(tmp_path):
    settings = agent_settings(tmp_path)
    workspace = settings.model_file_root
    workspace.mkdir()
    service = AgentService(
        ModelGateway(settings, httpx.MockTransport(lambda request: httpx.Response(500))),
        FileTools(workspace),
        PdfTools(workspace),
        WebTools(),
        TerminalTools(workspace, mode="disabled"),
        StateTools(tmp_path / "memory.json"),
        ApprovalBroker(),
        AgentRunStore(tmp_path / "agent.json"),
        {
            "max_tool_rounds": 10,
            "max_tool_calls": 20,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
    )
    execution = await service.create(
        AgentStartRequest(goal="Create a note", model="model-test")
    )
    arguments = {"path": "note.txt", "content": "do not duplicate"}
    receipt, created = service.store.begin_tool_receipt(
        execution.run_id,
        "interrupted-call",
        "create_file",
        arguments,
        arguments,
        True,
    )

    result = json.loads(
        await execution._execute_idempotent(
            {
                "id": "retried-call",
                "function": {"name": "create_file"},
            },
            "create_file",
            arguments,
            arguments,
        )
    )

    assert created is True
    assert result["outcome_uncertain"] is True
    assert result["receipt_id"] == receipt["id"]
    assert service.current()["status"] == "waiting_for_user"
    assert not (workspace / "note.txt").exists()


@pytest.mark.asyncio
async def test_root_agent_delegates_parallel_workers_and_integrates_reports(tmp_path):
    root_requests = []
    worker_requests = []
    root_responses = [
        ("plan", "agent_set_plan", {"steps": ["Inspect with specialists"]}),
        (
            "delegate",
            "agent_delegate_tasks",
            {
                "tasks": [
                    {
                        "title": "Inspect implementation",
                        "instruction": "Find the relevant code paths.",
                        "role": "researcher",
                        "scope": ".",
                    },
                    {
                        "title": "Review risks",
                        "instruction": "Identify likely regression risks.",
                        "role": "reviewer",
                        "scope": ".",
                    },
                ]
            },
        ),
        (
            "step",
            "agent_update_step",
            {"index": 0, "status": "completed", "note": "Reports integrated"},
        ),
        ("verify", "agent_begin_verification", {}),
        ("read", "read_file", {"path": "note.txt"}),
        (
            "evidence",
            "agent_record_verification",
            {"evidence": "Final artifact inspected."},
        ),
        ("complete", "agent_complete", {"summary": "Delegated goal completed."}),
    ]

    async def model_handler(request):
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        if system.startswith("You are a read-only specialist worker"):
            worker_requests.append(payload)
            await asyncio.sleep(0.01)
            return httpx.Response(200, content=(
                b'data: {"choices":[{"delta":{"content":"Worker report."}}]}\n\n'
                b"data: [DONE]\n\n"
            ))
        root_requests.append(payload)
        call_id, name, arguments = root_responses[len(root_requests) - 1]
        return httpx.Response(200, content=tool_stream(call_id, name, arguments))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("final artifact")
    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/agent/runs",
            json={"goal": "Use specialists", "model": "model-test"},
        )
        run = (await client.get("/api/agent/runs/current")).json()
        tasks = (
            await client.get(f"/api/agent/runs/{run['id']}/tasks")
        ).json()
        events = (
            await client.get(f"/api/agent/runs/{run['id']}/events")
        ).json()

    assert response.status_code == 200
    assert run["status"] == "completed"
    assert run["worker_task_count"] == 2
    assert len(tasks) == 2
    assert {task["status"] for task in tasks} == {"completed"}
    assert len(worker_requests) == 2
    assert "agent_delegate_tasks" in {
        tool["function"]["name"] for tool in root_requests[0]["tools"]
    }
    assert any(
        message.get("role") == "tool"
        and "Worker report." in message.get("content", "")
        for message in root_requests[2]["messages"]
    )
    assert [event["type"] for event in events].count("worker_task_completed") == 2


@pytest.mark.asyncio
async def test_root_agent_runs_v4_task_graph_and_graph_api_reports_completion(tmp_path):
    root_requests = []
    worker_requests = []
    root_responses = [
        ("plan", "agent_set_plan", {"steps": ["Run dependency graph"]}),
        (
            "graph",
            "agent_create_task_graph",
            {
                "tasks": [
                    {
                        "key": "inspect",
                        "title": "Inspect workspace",
                        "instruction": "Inspect the workspace.",
                        "role": "researcher",
                    },
                    {
                        "key": "review",
                        "title": "Review findings",
                        "instruction": "Review the inspection report.",
                        "role": "reviewer",
                        "depends_on": ["inspect"],
                    },
                ]
            },
        ),
        ("step", "agent_update_step", {"index": 0, "status": "completed"}),
        ("verify", "agent_begin_verification", {}),
        ("read", "read_file", {"path": "note.txt"}),
        (
            "evidence",
            "agent_record_verification",
            {"evidence": "Combined root workspace inspected."},
        ),
        ("complete", "agent_complete", {"summary": "Graph goal completed."}),
    ]

    def model_handler(request):
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        if system.startswith("You are a read-only specialist worker"):
            worker_requests.append(payload)
            instruction = payload["messages"][1]["content"]
            if "Task: Review findings" in instruction:
                assert "[inspect] Workspace inspected." in instruction
                report = "Findings reviewed."
            else:
                report = "Workspace inspected."
            return httpx.Response(
                200,
                content=(
                    f'data: {{"choices":[{{"delta":{{"content":{json.dumps(report)}}}}}]}}\n\n'
                    "data: [DONE]\n\n"
                ).encode(),
            )
        root_requests.append(payload)
        call_id, name, arguments = root_responses[len(root_requests) - 1]
        return httpx.Response(200, content=tool_stream(call_id, name, arguments))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("combined root")
    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/agent/runs",
            json={"goal": "Use task orchestration", "model": "model-test"},
        )
        run = (await client.get("/api/agent/runs/current")).json()
        graph_response = await client.get(f"/api/agent/runs/{run['id']}/graph")

    assert response.status_code == 200
    assert run["status"] == "completed"
    assert graph_response.status_code == 200
    assert graph_response.json()["status"] == "completed"
    assert len(worker_requests) == 2
    assert "agent_create_task_graph" in {
        tool["function"]["name"] for tool in root_requests[0]["tools"]
    }


@pytest.mark.asyncio
async def test_worker_integration_requires_approval_before_root_workspace_changes(tmp_path):
    root_calls = 0
    worker_calls = 0

    async def model_handler(request):
        nonlocal root_calls, worker_calls
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        if system.startswith("You are an implementation worker"):
            worker_calls += 1
            if worker_calls == 1:
                return httpx.Response(
                    200,
                    content=tool_stream(
                        "worker-write",
                        "write_file",
                        {"path": "note.txt", "content": "integrated version"},
                    ),
                )
            if worker_calls == 2:
                return httpx.Response(
                    200,
                    content=tool_stream(
                        "worker-extra",
                        "create_file",
                        {"path": "extra.txt", "content": "reject this file"},
                    ),
                )
            return httpx.Response(
                200,
                content=(
                    b'data: {"choices":[{"delta":{"content":"Implementation ready."}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )

        root_calls += 1
        if root_calls == 1:
            response = ("plan", "agent_set_plan", {"steps": ["Implement safely"]})
        elif root_calls == 2:
            response = (
                "delegate",
                "agent_delegate_tasks",
                {
                    "tasks": [
                        {
                            "title": "Update the note",
                            "instruction": "Change note.txt to the integrated version.",
                            "role": "implementer",
                            "mode": "isolated_write",
                            "scope": ".",
                        }
                    ]
                },
            )
        elif root_calls == 3:
            delegated = next(
                json.loads(message["content"])
                for message in reversed(payload["messages"])
                if message.get("role") == "tool" and "tasks" in message["content"]
            )
            response = (
                "integrate",
                "agent_integrate_worker",
                {"task_id": delegated["tasks"][0]["id"]},
            )
        elif root_calls == 4:
            response = (
                "step",
                "agent_update_step",
                {"index": 0, "status": "completed"},
            )
        elif root_calls == 5:
            response = ("verify", "agent_begin_verification", {})
        elif root_calls == 6:
            response = ("read", "read_file", {"path": "note.txt"})
        elif root_calls == 7:
            response = (
                "evidence",
                "agent_record_verification",
                {"evidence": "Integrated note content verified."},
            )
        else:
            response = (
                "complete",
                "agent_complete",
                {"summary": "Implementation integrated and verified."},
            )
        return httpx.Response(200, content=tool_stream(*response))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    note = workspace / "note.txt"
    note.write_text("root version")
    settings = agent_settings(tmp_path)
    approvals = ApprovalBroker()
    service = AgentService(
        ModelGateway(settings, httpx.MockTransport(model_handler)),
        FileTools(workspace),
        PdfTools(workspace),
        WebTools(),
        TerminalTools(workspace, mode="disabled"),
        StateTools(tmp_path / "memory.json"),
        approvals,
        AgentRunStore(tmp_path / "agent.json"),
        {
            "max_tool_rounds": 20,
            "max_tool_calls": 40,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
    )
    execution = await service.create(
        AgentStartRequest(goal="Safely update the note", model="model-test")
    )
    approval_seen = False
    async for raw_event in execution.events():
        event = raw_event.decode()
        if '"harness_event":"tool_approval"' not in event:
            continue
        payload = json.loads(event.removeprefix("data: ").strip())
        assert payload["name"] == "agent_integrate_worker"
        assert note.read_text() == "root version"
        assert payload["arguments"]["review"]["change_count"] == 2
        assert approvals.decide(
            payload["approval_id"],
            True,
            {"accepted_paths": ["note.txt"], "accepted_hunks": {}},
        ) is True
        approval_seen = True

    run = service.current()
    task = run["worker_tasks"][0]

    assert approval_seen is True
    assert run["status"] == "completed"
    assert task["integration_status"] == "integrated"
    assert task["review_decision"]["accepted_paths"] == ["note.txt"]
    assert note.read_text() == "integrated version"
    assert not (workspace / "extra.txt").exists()
