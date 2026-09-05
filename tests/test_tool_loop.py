import asyncio
import json

import httpx
import pytest

from app.approvals import ApprovalBroker
from app.config import Settings
from app.file_tools import FileTools
from app.main import create_app
from app.model_gateway import ModelGateway
from app.pdf_tools import PdfTools
from app.schemas import ChatRequest
from app.state_tools import StateTools
from app.terminal_tools import TerminalTools
from app.tool_loop import ToolLoop
from app.web_tools import WebTools


@pytest.mark.asyncio
async def test_read_tool_runs_and_model_continues(tmp_path):
    FileTools(tmp_path).create_file("note.txt", "hello from the workspace")
    requests = []

    def model_handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            stream = (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"id":"call_read","type":"function","function":{"name":'
                b'"read_file","arguments":"{\\"path\\":\\"note.txt\\"}"}}]},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
                b'data: [DONE]\n\n'
            )
        else:
            stream = (
                b'data: {"choices":[{"delta":{"content":"Read complete."},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                b'data: [DONE]\n\n'
            )
        return httpx.Response(200, content=stream)

    settings = Settings(
        model_base_url="http://model.test",
        model_name="gemma-test",
        model_api_key=None,
        model_file_root=tmp_path,
    )
    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=settings,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Read note.txt"}]},
        )

    assert response.status_code == 200
    assert '"harness_event":"tool_started"' in response.text
    assert '"harness_event":"tool_result"' in response.text
    assert '"content":"Read complete."' in response.text
    tool_names = {tool["function"]["name"] for tool in requests[0]["tools"]}
    assert "read_file" in tool_names
    assert "read_pdf" in tool_names
    assert "run_command" in tool_names
    assert {"list_state", "read_state", "write_state", "delete_state"} <= tool_names
    assert requests[0]["messages"][0]["role"] == "system"
    assert "persistent state" in requests[0]["messages"][0]["content"]
    tool_result = requests[1]["messages"][-1]
    assert tool_result["role"] == "tool"
    assert "hello from the workspace" in tool_result["content"]


@pytest.mark.asyncio
async def test_fetch_url_tool_runs_and_model_receives_page(tmp_path):
    model_requests = []

    def model_handler(request):
        payload = json.loads(request.content)
        model_requests.append(payload)
        if len(model_requests) == 1:
            stream = (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"id":"call_fetch","type":"function","function":{"name":'
                b'"fetch_url","arguments":"{\\"url\\":\\"https://example.com\\"}"}}]},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
                b'data: [DONE]\n\n'
            )
        else:
            stream = (
                b'data: {"choices":[{"delta":{"content":"Fetch complete."},'
                b'"finish_reason":null}]}\n\n'
                b'data: [DONE]\n\n'
            )
        return httpx.Response(200, content=stream)

    def web_handler(request):
        return httpx.Response(
            200,
            text="<html><title>Example</title><body>outside information</body></html>",
            headers={"content-type": "text/html"},
        )

    settings = Settings(
        model_base_url="http://model.test",
        model_name="gemma-test",
        model_api_key=None,
        model_file_root=tmp_path,
    )
    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        web_transport=httpx.MockTransport(web_handler),
        settings=settings,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Fetch example.com"}]},
        )

    assert response.status_code == 200
    assert '"name":"fetch_url"' in response.text
    assert '"content":"Fetch complete."' in response.text
    tool_names = {tool["function"]["name"] for tool in model_requests[0]["tools"]}
    assert {"search_web", "fetch_url"} <= tool_names
    tool_result = model_requests[1]["messages"][-1]
    assert "outside information" in tool_result["content"]


@pytest.mark.asyncio
async def test_command_requires_approval_and_model_receives_output(tmp_path):
    model_requests = []

    def model_handler(request):
        payload = json.loads(request.content)
        model_requests.append(payload)
        if len(model_requests) == 1:
            stream = (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"id":"call_command","type":"function","function":{"name":'
                b'"run_command","arguments":"{\\"command\\":\\"printf terminal-output\\"}"}}]},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
                b'data: [DONE]\n\n'
            )
        else:
            stream = (
                b'data: {"choices":[{"delta":{"content":"Command complete."},'
                b'"finish_reason":null}]}\n\n'
                b'data: [DONE]\n\n'
            )
        return httpx.Response(200, content=stream)

    settings = Settings(
        model_base_url="http://model.test",
        model_name="gemma-test",
        model_api_key=None,
        model_file_root=tmp_path,
    )
    gateway = ModelGateway(settings, httpx.MockTransport(model_handler))
    approvals = ApprovalBroker()
    loop = ToolLoop(
        gateway,
        FileTools(tmp_path),
        PdfTools(tmp_path),
        WebTools(),
        TerminalTools(tmp_path, mode="host"),
        StateTools(tmp_path / ".state.json"),
        approvals,
    )
    request = ChatRequest(
        messages=[{"role": "user", "content": "Run a command"}]
    )
    run = await loop.start(request)
    events = []
    event_stream = run.events()
    while True:
        raw_event = await event_stream.__anext__()
        text = raw_event.decode()
        events.append(text)
        if '"harness_event":"tool_approval"' in text:
            approval_event = json.loads(text.removeprefix("data: "))
            assert approvals.decide(approval_event["approval_id"], True)
            break
    async for event in event_stream:
        events.append(event.decode())

    assert any('"harness_event":"tool_approval"' in event for event in events)
    assert not any('"harness_event":"tool_started"' in event for event in events)
    assert any('"harness_event":"tool_result"' in event for event in events)
    assert any('"stdout":"terminal-output"' in event for event in events)
    tool_result = model_requests[1]["messages"][-1]
    assert "terminal-output" in tool_result["content"]


@pytest.mark.asyncio
async def test_approval_resolution_can_carry_review_selection():
    approvals = ApprovalBroker()
    approval_id = approvals.register("agent_integrate_worker", ".")
    waiting = asyncio.create_task(approvals.wait_resolution(approval_id))
    selection = {
        "accepted_paths": ["note.txt"],
        "accepted_hunks": {"other.txt": ["hunk-id"]},
    }

    assert approvals.decide(approval_id, True, selection) is True
    resolution = await waiting

    assert resolution.approved is True
    assert resolution.selection == selection


@pytest.mark.asyncio
async def test_model_can_update_persistent_state_without_approval(tmp_path):
    model_requests = []

    def model_handler(request):
        payload = json.loads(request.content)
        model_requests.append(payload)
        if len(model_requests) == 1:
            stream = (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"id":"call_state","type":"function","function":{"name":'
                b'"write_state","arguments":"{\\"key\\":\\"project/decision\\",'
                b'\\"value\\":\\"Use the bounded store.\\"}"}}]},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
                b'data: [DONE]\n\n'
            )
        else:
            stream = (
                b'data: {"choices":[{"delta":{"content":"State updated."},'
                b'"finish_reason":null}]}\n\n'
                b'data: [DONE]\n\n'
            )
        return httpx.Response(200, content=stream)

    settings = Settings(
        model_base_url="http://model.test",
        model_name="gemma-test",
        model_api_key=None,
        model_file_root=tmp_path,
    )
    gateway = ModelGateway(settings, httpx.MockTransport(model_handler))
    approvals = ApprovalBroker()
    state_tools = StateTools(tmp_path / ".state.json")
    loop = ToolLoop(
        gateway,
        FileTools(tmp_path),
        PdfTools(tmp_path),
        WebTools(),
        TerminalTools(tmp_path),
        state_tools,
        approvals,
    )
    run = await loop.start(
        ChatRequest(messages=[{"role": "user", "content": "Remember the decision"}])
    )

    events = [event.decode() async for event in run.events()]

    assert not any('"harness_event":"tool_approval"' in event for event in events)
    assert state_tools.read_state("project/decision")["value"] == "Use the bounded store."
    assert '"created": true' in model_requests[1]["messages"][-1]["content"]
