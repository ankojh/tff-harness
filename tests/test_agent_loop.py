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
from app.schemas import AgentStartRequest
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
        (
            "evidence",
            "agent_record_verification",
            {"evidence": "Inspection confirmed the expected result."},
        ),
        ("complete", "agent_complete", {"summary": "Goal completed."}),
    ]


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


@pytest.mark.asyncio
async def test_agent_api_runs_lifecycle_and_persists_public_state(tmp_path):
    requests = []
    responses = lifecycle_responses()
    responses.insert(1, ("inspect", "read_file", {"path": "note.txt"}))
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

    assert response.status_code == 200
    assert '"harness_event":"agent_run"' in response.text
    assert '"harness_event":"agent_finished"' in response.text
    assert '"status":"completed"' in response.text
    public = current.json()
    assert public["status"] == "completed"
    assert public["summary"] == "Goal completed."
    assert public["verification_evidence"] == [
        "Inspection confirmed the expected result."
    ]
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
    } <= offered_tools
    assert any(
        message.get("role") == "tool"
        and "expected workspace evidence" in message.get("content", "")
        for message in requests[2]["messages"]
    )


@pytest.mark.asyncio
async def test_agent_exhausts_round_budget_when_model_does_not_use_lifecycle(tmp_path):
    def model_handler(request):
        stream = (
            b'data: {"choices":[{"delta":{"content":"Still thinking"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=stream)

    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=agent_settings(tmp_path, agent_max_tool_rounds=1),
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
    assert current["usage"]["tool_rounds"] == 1
    assert "1 model/tool rounds" in current["summary"]


@pytest.mark.asyncio
async def test_failed_agent_run_can_be_manually_resumed(tmp_path):
    calls = 0
    responses = lifecycle_responses()

    def model_handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporarily offline", request=request)
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
