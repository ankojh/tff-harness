import httpx
import pytest

from app.main import create_app


@pytest.mark.asyncio
async def test_index_serves_chat_ui():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "<title>Chat</title>" in response.text
    assert 'fetch("/api/chat"' in response.text
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
async def test_status_reports_offline():
    def fail(request):
        raise httpx.ConnectError("offline", request=request)

    app = create_app(model_transport=httpx.MockTransport(fail))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["connected"] is False
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
async def test_chat_rejects_empty_messages():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat", json={"messages": []})

    assert response.status_code == 422
