from copy import deepcopy
import json

import httpx
import pytest

from app.config import Settings
from app.context_compaction import ContextCompactor
from app.memory_tools import MemoryStore
from app.model_gateway import ModelGateway


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_turn", [
    {"role": "assistant", "content": None},
    {"role": "assistant"},
    {"role": "assistant", "content": ""},
    {"role": "assistant", "content": None, "tool_calls": []},
])
async def test_completion_normalizes_empty_assistant_turns(tmp_path, empty_turn):
    messages = [
        {"role": "user", "content": "Continue"},
        empty_turn,
        {"role": "system", "content": "The run is still active."},
    ]
    original = deepcopy(messages)

    def handler(request):
        sent = json.loads(request.content)["messages"]
        assert sent == [messages[2], messages[0], {**empty_turn, "content": ""}]
        return httpx.Response(200, content=b'data: [DONE]\n\n')

    gateway = ModelGateway(Settings.from_env(), httpx.MockTransport(handler))
    stream = await gateway.open_completion(messages, "model-test", 0.0, 1)
    assert [payload async for payload in stream.payloads()] == []
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_usage", [False, True])
async def test_completion_orders_guidance_without_mutating_history(tmp_path, retry_usage):
    messages = [
        {"role": "system", "content": "Initial instructions"},
        {"role": "user", "content": "Inspect the workspace"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "read", "type": "function", "function": {"name": "list_files", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "read", "content": '{"ok":true}'},
        {"role": "system", "content": "Resume this run"},
        {"role": "developer", "content": "Preserve the original scope"},
        {"role": "user", "content": "Continue"},
    ]
    original = deepcopy(messages)
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["messages"][0] == {
            "role": "system",
            "content": "System guidance:\nInitial instructions\n\nSystem guidance:\nResume this run\n\nDeveloper guidance:\nPreserve the original scope",
        }
        assert payload["messages"][1:] == [original[i] for i in [1, 2, 3, 6]]
        if retry_usage and len(requests) == 1:
            return httpx.Response(400, json={"error": {"message": "unknown field stream_options"}})
        return httpx.Response(200, content=b'data: {"choices":[]}\n\ndata: [DONE]\n\n')

    gateway = ModelGateway(Settings.from_env(), httpx.MockTransport(handler))
    stream = await gateway.open_completion(messages, "model-test", 0.0, 1)
    assert [payload async for payload in stream.payloads()] == [{"choices": []}]
    assert messages == original
    assert len(requests) == (2 if retry_usage else 1)


@pytest.mark.asyncio
async def test_compacted_context_guidance_precedes_conversation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    compactor = ContextCompactor(MemoryStore(tmp_path / "memory.json", workspace))
    messages = [
        {"role": "system", "content": "Initial instructions"},
        {"role": "user", "content": "Inspect"},
        {"role": "assistant", "content": "Old context " * 100},
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "Recent findings"},
    ]
    compacted, record = compactor.compact(
        {"id": "a" * 32, "goal": "Inspect", "status": "working", "plan": []},
        messages, threshold=500, target_recent=100,
    )
    assert record is not None

    def handler(request):
        sent = json.loads(request.content)["messages"]
        assert sent[0] == {"role": "system", "content": compacted[0]["content"] + "\n\n" + compacted[2]["content"]}
        assert sent[1:] == [compacted[1], *compacted[3:]]
        return httpx.Response(200, content=b'data: [DONE]\n\n')

    gateway = ModelGateway(Settings.from_env(), httpx.MockTransport(handler))
    stream = await gateway.open_completion(compacted, "model-test", 0.0, 1)
    assert [payload async for payload in stream.payloads()] == []
