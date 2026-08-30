from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.approvals import ApprovalBroker
from app.file_tools import FileTools
from app.model_gateway import ChatStream, ModelGateway, ModelGatewayError
from app.pdf_tools import PdfTools
from app.schemas import ChatRequest
from app.state_tools import STATE_SYSTEM_PROMPT, StateTools
from app.terminal_tools import TerminalTools
from app.web_tools import WebTools


MAX_TOOL_ROUNDS = 8


def sse(payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


@dataclass
class ToolRun:
    gateway: ModelGateway
    file_tools: FileTools
    pdf_tools: PdfTools
    web_tools: WebTools
    terminal_tools: TerminalTools
    state_tools: StateTools
    approvals: ApprovalBroker
    request: ChatRequest
    first_stream: ChatStream
    initial_messages: list[dict[str, Any]]

    async def events(self):
        history = [dict(message) for message in self.initial_messages]
        stream = self.first_stream

        for _ in range(MAX_TOOL_ROUNDS):
            content = ""
            calls: dict[int, dict[str, Any]] = {}
            try:
                async for payload in stream.payloads():
                    self._accumulate(payload, calls)
                    content += self._content(payload)
                    yield sse(payload)
            except ModelGatewayError as exc:
                yield sse({"harness_event": "error", "message": str(exc)})
                yield b"data: [DONE]\n\n"
                return

            if not calls:
                yield b"data: [DONE]\n\n"
                return

            normalized_calls = [calls[index] for index in sorted(calls)]
            history.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": normalized_calls,
                }
            )

            for call in normalized_calls:
                name = call["function"]["name"]
                arguments, argument_error = self._arguments(call)
                if argument_error:
                    result = json.dumps({"ok": False, "error": argument_error})
                elif self._requires_approval(name):
                    display = self._display_arguments(name, arguments)
                    approval_id = self.approvals.register(
                        name,
                        str(display.get("path") or display.get("command") or ""),
                    )
                    yield sse(
                        {
                            "harness_event": "tool_approval",
                            "approval_id": approval_id,
                            "call_id": call["id"],
                            "name": name,
                            "arguments": display,
                        }
                    )
                    approved = await self.approvals.wait(approval_id)
                    if approved:
                        result = await self._execute(name, arguments)
                    else:
                        result = json.dumps(
                            {"ok": False, "error": "The user denied this operation."}
                        )
                else:
                    display = self._display_arguments(name, arguments)
                    yield sse(
                        {
                            "harness_event": "tool_started",
                            "call_id": call["id"],
                            "name": name,
                            "arguments": display,
                        }
                    )
                    result = await self._execute(name, arguments)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )
                parsed_result = json.loads(result)
                result_event = {
                    "harness_event": "tool_result",
                    "call_id": call["id"],
                    "name": name,
                    "ok": parsed_result.get("ok", False),
                    "error": parsed_result.get("error"),
                }
                if name in self.terminal_tools.names:
                    result_event["output"] = self.terminal_tools.display_result(
                        parsed_result
                    )
                yield sse(result_event)

            try:
                stream = await self.gateway.open_completion(
                    messages=history,
                    requested_model=self.first_stream.model,
                    temperature=self.request.temperature,
                    max_tokens=self.request.max_tokens,
                    tools=self.definitions,
                )
            except ModelGatewayError as exc:
                yield sse({"harness_event": "error", "message": str(exc)})
                yield b"data: [DONE]\n\n"
                return

        yield sse(
            {
                "harness_event": "error",
                "message": f"Stopped after {MAX_TOOL_ROUNDS} tool rounds.",
            }
        )
        yield b"data: [DONE]\n\n"

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            *self.file_tools.definitions,
            *self.pdf_tools.definitions,
            *self.web_tools.definitions,
            *self.terminal_tools.definitions,
            *self.state_tools.definitions,
        ]

    def _requires_approval(self, name: str) -> bool:
        return (
            self.file_tools.requires_approval(name)
            or self.terminal_tools.requires_approval(name)
        )

    def _display_arguments(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if name in self.web_tools.names:
            return self.web_tools.display_arguments(name, arguments)
        if name in self.pdf_tools.names:
            return self.pdf_tools.display_arguments(name, arguments)
        if name in self.terminal_tools.names:
            return self.terminal_tools.display_arguments(name, arguments)
        if name in self.state_tools.names:
            return self.state_tools.display_arguments(name, arguments)
        return self.file_tools.display_arguments(name, arguments)

    async def _execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name in self.web_tools.names:
            return await self.web_tools.execute(name, arguments)
        if name in self.pdf_tools.names:
            return await self.pdf_tools.execute(name, arguments)
        if name in self.terminal_tools.names:
            return await self.terminal_tools.execute(name, arguments)
        if name in self.state_tools.names:
            return self.state_tools.execute(name, arguments)
        return self.file_tools.execute(name, arguments)

    @staticmethod
    def _arguments(call: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            return {}, "Tool arguments were not valid JSON."
        if not isinstance(arguments, dict):
            return {}, "Tool arguments must be an object."
        return arguments, None

    @staticmethod
    def _content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("delta") or {}).get("content")
        return content if isinstance(content, str) else ""

    @staticmethod
    def _accumulate(
        payload: dict[str, Any], calls: dict[int, dict[str, Any]]
    ) -> None:
        choices = payload.get("choices") or []
        if not choices:
            return
        fragments = (choices[0].get("delta") or {}).get("tool_calls") or []
        for fragment in fragments:
            index = fragment.get("index")
            if not isinstance(index, int):
                continue
            call = calls.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if fragment.get("id"):
                call["id"] = fragment["id"]
            if fragment.get("type"):
                call["type"] = fragment["type"]
            function = fragment.get("function") or {}
            if function.get("name"):
                call["function"]["name"] = function["name"]
            if isinstance(function.get("arguments"), str):
                call["function"]["arguments"] += function["arguments"]


class ToolLoop:
    def __init__(
        self,
        gateway: ModelGateway,
        file_tools: FileTools,
        pdf_tools: PdfTools,
        web_tools: WebTools,
        terminal_tools: TerminalTools,
        state_tools: StateTools,
        approvals: ApprovalBroker,
    ) -> None:
        self.gateway = gateway
        self.file_tools = file_tools
        self.pdf_tools = pdf_tools
        self.web_tools = web_tools
        self.terminal_tools = terminal_tools
        self.state_tools = state_tools
        self.approvals = approvals

    async def start(self, request: ChatRequest) -> ToolRun:
        initial_messages = [
            {"role": "system", "content": STATE_SYSTEM_PROMPT},
            *[message.model_dump() for message in request.messages],
        ]
        definitions = [
            *self.file_tools.definitions,
            *self.pdf_tools.definitions,
            *self.web_tools.definitions,
            *self.terminal_tools.definitions,
            *self.state_tools.definitions,
        ]
        first_stream = await self.gateway.open_completion(
            messages=initial_messages,
            requested_model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=definitions,
        )
        return ToolRun(
            gateway=self.gateway,
            file_tools=self.file_tools,
            pdf_tools=self.pdf_tools,
            web_tools=self.web_tools,
            terminal_tools=self.terminal_tools,
            state_tools=self.state_tools,
            approvals=self.approvals,
            request=request,
            first_stream=first_stream,
            initial_messages=initial_messages,
        )
