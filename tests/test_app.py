import json

import httpx
import pytest

from app.main import create_app
from app.agent_runs import AgentRunStore
from app.config import Settings


@pytest.mark.asyncio
async def test_index_serves_chat_ui():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "<title>Chat</title>" in response.text
    assert '"/api/chat"' in response.text
    assert '"/api/agent/runs"' in response.text
    assert 'id="agentPanel"' in response.text
    assert 'id="agentGraph"' in response.text
    assert 'id="agentGraphCanvas"' in response.text
    assert 'id="agentGraphTimeline"' in response.text
    assert 'id="operatorDialog"' in response.text
    assert 'id="compareAgent"' in response.text
    assert 'id="cloneAgent"' in response.text
    assert 'id="debugAgent"' in response.text
    assert 'id="observeAgent"' in response.text
    assert 'id="evalsAgent"' in response.text
    assert 'id="memoryStatus"' in response.text
    assert "agent_create_task_graph" in response.text
    assert '<script src="/static/markdown.js"></script>' in response.text
    assert 'id="terminalMode"' in response.text
    assert "Turbo Fair Field" not in response.text
    assert "data-chat-panel" not in response.text


@pytest.mark.asyncio
async def test_markdown_renderer_is_served():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/markdown.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert "function renderMarkdown" in response.text
    assert "escapeHtml" in response.text


@pytest.mark.asyncio
async def test_graph_templates_are_exposed_without_model_access():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/agent/graph/templates")

    assert response.status_code == 200
    assert {item["name"] for item in response.json()} >= {
        "parallel_analysis",
        "research_implement_review",
        "test_fix_verify",
    }
    assert all(item["version"] == 1 for item in response.json())
    assert all("focus" in item["parameters"] for item in response.json())


@pytest.mark.asyncio
async def test_operator_run_routes_compare_clone_and_debug(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = Settings(
        model_base_url="http://model.test",
        model_name="model-test",
        model_api_key=None,
        model_file_root=workspace,
        agent_state_file=tmp_path / "agent.json",
        terminal_mode="disabled",
    )
    store = AgentRunStore(settings.agent_state_file)
    budgets = {
        "max_tool_rounds": 10,
        "max_tool_calls": 20,
        "max_seconds": 300,
        "max_consecutive_failures": 3,
    }

    def create_stopped(goal):
        run = store.create(
            goal,
            model="model-test",
            temperature=0.4,
            max_tokens=4096,
            budgets=budgets,
            messages=[
                {"role": "system", "content": "System"},
                {"role": "user", "content": goal},
            ],
        )
        return store.mutate(
            run["id"],
            lambda record: record.update({"status": "stopped"}),
            event_type="run_stopped_for_test",
        )

    left = create_stopped("Left run")
    right = create_stopped("Right run")
    app = create_app(
        model_transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        settings=settings,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        loaded = await client.get(f"/api/agent/runs/{left['id']}")
        compared = await client.get(
            "/api/agent/runs/compare",
            params={"left_id": left["id"], "right_id": right["id"]},
        )
        cloned = await client.post(f"/api/agent/runs/{left['id']}/clone", json={})
        clone_id = cloned.json()["id"]
        debugged = await client.get(f"/api/agent/runs/{clone_id}/debug")

    assert loaded.status_code == 200
    assert compared.status_code == 200
    assert compared.json()["left"]["id"] == left["id"]
    assert cloned.status_code == 200
    assert cloned.json()["parent_run_id"] == left["id"]
    assert debugged.status_code == 200
    assert debugged.json()["diagnostics"]["state_validated"] is True


@pytest.mark.asyncio
async def test_status_reports_offline():
    def fail(request):
        raise httpx.ConnectError("offline", request=request)

    app = create_app(model_transport=httpx.MockTransport(fail))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["connected"] is False
    assert response.json()["agent_version"] == "9.0"
    assert response.json()["terminal_mode"] == "sandbox"
    assert response.json()["sandbox_image"] == "tff-harness-sandbox:latest"


@pytest.mark.asyncio
async def test_chat_relays_stream(monkeypatch):
    def model_handler(request):
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gemma-test"}]})
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        return httpx.Response(200, content=b"".join(chunks))

    monkeypatch.delenv("MODEL_NAME", raising=False)
    app = create_app(model_transport=httpx.MockTransport(model_handler))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        ) as response:
            body = await response.aread()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b'"content":"Hello"' in body
    assert body.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_model_gateway_falls_back_when_stream_usage_is_unsupported(tmp_path):
    posts = []

    def model_handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "model-test"}]})
        payload = json.loads(request.content)
        posts.append(payload)
        if len(posts) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "unknown field stream_options"}},
            )
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    settings = Settings(
        model_base_url="http://model.test",
        model_name="model-test",
        model_api_key=None,
        model_file_root=tmp_path / "workspace",
        agent_state_file=tmp_path / "agent.json",
        terminal_mode="disabled",
    )
    app = create_app(
        model_transport=httpx.MockTransport(model_handler),
        settings=settings,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == 200
    assert len(posts) == 2
    assert posts[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in posts[1]


@pytest.mark.asyncio
async def test_chat_rejects_empty_messages():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat", json={"messages": []})

    assert response.status_code == 422
