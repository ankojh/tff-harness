from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
import json
import time
from typing import Any

from app.agent_runs import (
    ACTIVE_AGENT_STATUSES,
    AGENT_SYSTEM_PROMPT,
    RESUMABLE_AGENT_STATUSES,
    TERMINAL_AGENT_STATUSES,
    AgentControlTools,
    AgentRunError,
    AgentRunStore,
)
from app.approvals import ApprovalBroker
from app.file_tools import FileTools
from app.model_gateway import ModelGateway, ModelGatewayError
from app.pdf_tools import PdfTools
from app.schemas import AgentResumeRequest, AgentStartRequest
from app.state_tools import STATE_SYSTEM_PROMPT, StateTools
from app.terminal_tools import TerminalTools
from app.tool_loop import ToolRun, sse
from app.web_tools import WebTools


MAX_PERSISTED_TOOL_CHARS = 32_000
MAX_PERSISTED_CONTENT_CHARS = 16_000


class AgentService:
    def __init__(
        self,
        gateway: ModelGateway,
        file_tools: FileTools,
        pdf_tools: PdfTools,
        web_tools: WebTools,
        terminal_tools: TerminalTools,
        state_tools: StateTools,
        approvals: ApprovalBroker,
        store: AgentRunStore,
        budgets: dict[str, int],
    ) -> None:
        self.gateway = gateway
        self.file_tools = file_tools
        self.pdf_tools = pdf_tools
        self.web_tools = web_tools
        self.terminal_tools = terminal_tools
        self.state_tools = state_tools
        self.approvals = approvals
        self.store = store
        self.budgets = dict(budgets)
        self.store.mark_interrupted()
        self._lock = asyncio.Lock()
        self._active_id: str | None = None
        self._stop_events: dict[str, asyncio.Event] = {}

    async def create(self, request: AgentStartRequest) -> "AgentExecution":
        async with self._lock:
            if self._active_id is not None:
                raise AgentRunError("Another agent run is already active.")
            goal = request.goal.strip()
            if not goal:
                raise AgentRunError("Agent goal must not be blank.")
            messages = [
                {
                    "role": "system",
                    "content": f"{STATE_SYSTEM_PROMPT}\n\n{AGENT_SYSTEM_PROMPT}",
                },
                {"role": "user", "content": goal},
            ]
            run = self.store.create(
                goal,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                budgets=self.budgets,
                messages=messages,
            )
            stop_event = asyncio.Event()
            self._active_id = run["id"]
            self._stop_events[run["id"]] = stop_event
            return AgentExecution(self, run["id"], stop_event)

    async def resume(
        self,
        run_id: str,
        request: AgentResumeRequest,
    ) -> "AgentExecution":
        async with self._lock:
            if self._active_id is not None:
                raise AgentRunError("Another agent run is already active.")
            run = self.store.get(run_id)
            if run["status"] not in RESUMABLE_AGENT_STATUSES:
                raise AgentRunError(f"A {run['status']} run cannot be resumed.")
            messages = self._repair_history(run["messages"])
            instruction = request.instruction.strip() if request.instruction else ""
            resume_context = {
                "status": run["status"],
                "plan": run["plan"],
                "verification_evidence": run["verification_evidence"],
                "last_error": run["last_error"],
            }
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This persisted agent run is being resumed manually. "
                        f"Current run state: {json.dumps(resume_context, ensure_ascii=False)}"
                    ),
                }
            )
            if instruction:
                messages.append({"role": "user", "content": instruction})

            def update(record: dict[str, Any]) -> None:
                record["messages"] = self._compact_messages(messages)
                record["status"] = "working" if record["plan"] else "planning"
                record["stop_requested"] = False
                record["last_error"] = None
                record["summary"] = None

            self.store.mutate(run_id, update)
            stop_event = asyncio.Event()
            self._active_id = run_id
            self._stop_events[run_id] = stop_event
            return AgentExecution(self, run_id, stop_event, messages=messages)

    async def stop(self, run_id: str) -> dict[str, Any]:
        async with self._lock:
            run = self.store.get(run_id)
            if (
                run["status"] in TERMINAL_AGENT_STATUSES
                and run["status"] != "waiting_for_user"
            ):
                return AgentRunStore.public(run)

            def request_stop(record: dict[str, Any]) -> None:
                record["stop_requested"] = True
                if self._active_id != run_id:
                    record["status"] = "stopped"
                    record["summary"] = "Stopped by the user."

            run = self.store.mutate(run_id, request_stop)
            stop_event = self._stop_events.get(run_id)
            if stop_event is not None:
                stop_event.set()
            return AgentRunStore.public(run)

    def current(self) -> dict[str, Any] | None:
        run = self.store.current()
        return AgentRunStore.public(run) if run else None

    async def release(self, run_id: str) -> None:
        async with self._lock:
            if self._active_id == run_id:
                self._active_id = None
            self._stop_events.pop(run_id, None)

    @staticmethod
    def _repair_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        repaired = deepcopy(messages)
        last_assistant = None
        for index in range(len(repaired) - 1, -1, -1):
            if repaired[index].get("role") == "assistant" and repaired[index].get("tool_calls"):
                last_assistant = index
                break
        if last_assistant is None:
            return repaired
        required = {
            call.get("id")
            for call in repaired[last_assistant].get("tool_calls", [])
            if call.get("id")
        }
        present = {
            message.get("tool_call_id")
            for message in repaired[last_assistant + 1 :]
            if message.get("role") == "tool"
        }
        for call_id in sorted(required - present):
            repaired.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {"ok": False, "error": "The previous run was interrupted."}
                    ),
                }
            )
        return repaired

    @staticmethod
    def _compact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = deepcopy(messages)
        for message in compact:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            limit = (
                MAX_PERSISTED_TOOL_CHARS
                if message.get("role") == "tool"
                else MAX_PERSISTED_CONTENT_CHARS
            )
            if len(content) > limit:
                message["content"] = content[:limit] + "\n[truncated in persisted run state]"
        return compact


class AgentExecution:
    def __init__(
        self,
        service: AgentService,
        run_id: str,
        stop_event: asyncio.Event,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.service = service
        self.run_id = run_id
        self.stop_event = stop_event
        self._messages = deepcopy(messages) if messages is not None else None
        self._started = 0.0
        self._base_elapsed = 0
        self._budget_expired = False

    async def events(self):
        run = self.service.store.get(self.run_id)
        messages = self._messages or deepcopy(run["messages"])
        self._base_elapsed = run["usage"]["elapsed_seconds"]
        self._started = time.monotonic()
        yield self._run_event()
        watchdog = asyncio.create_task(self._watch_elapsed_budget())
        try:
            while True:
                reason = self._budget_reason()
                if reason:
                    self._finish_with_status("budget_exhausted", reason)
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return
                if self._stop_requested():
                    self._finish_after_interrupt()
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return

                self._increment_round()
                try:
                    stream = await self._open_completion(messages, run)
                except ModelGatewayError as exc:
                    self._finish_with_status("failed", str(exc), error=str(exc))
                    yield sse({"harness_event": "error", "message": str(exc)})
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return
                if stream is None:
                    self._finish_after_interrupt()
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return

                content = ""
                calls: dict[int, dict[str, Any]] = {}
                try:
                    async for payload in self._payloads_until_stopped(stream):
                        ToolRun._accumulate(payload, calls)
                        content += ToolRun._content(payload)
                        yield sse(payload)
                except ModelGatewayError as exc:
                    self._finish_with_status("failed", str(exc), error=str(exc))
                    yield sse({"harness_event": "error", "message": str(exc)})
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return

                if self._stop_requested():
                    self._finish_after_interrupt()
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return

                if not calls:
                    messages.append({"role": "assistant", "content": content or None})
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The run is still active. Continue working and use the "
                                "agent lifecycle tools; ordinary prose cannot complete a run."
                            ),
                        }
                    )
                    self._persist_messages(messages)
                    continue

                normalized_calls = [calls[index] for index in sorted(calls)]
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": normalized_calls,
                    }
                )
                self._persist_messages(messages)

                for index, call in enumerate(normalized_calls):
                    reason = self._budget_reason(include_tool_call=True)
                    if reason:
                        self._append_skipped_results(messages, normalized_calls[index:], reason)
                        self._persist_messages(messages)
                        self._finish_with_status("budget_exhausted", reason)
                        yield self._finished_event()
                        yield b"data: [DONE]\n\n"
                        return
                    if self._stop_requested():
                        self._append_skipped_results(
                            messages,
                            normalized_calls[index:],
                            "Stopped by the user.",
                        )
                        self._persist_messages(messages)
                        self._finish_after_interrupt()
                        yield self._finished_event()
                        yield b"data: [DONE]\n\n"
                        return

                    name = call["function"]["name"]
                    arguments, argument_error = ToolRun._arguments(call)
                    self._increment_tool_call()
                    if argument_error:
                        result = json.dumps({"ok": False, "error": argument_error})
                    elif name in self.control_tools.names:
                        result = self.control_tools.execute(name, arguments)
                    else:
                        async for event, tool_result in self._run_external_tool(
                            call, name, arguments
                        ):
                            if event is not None:
                                yield event
                            if tool_result is not None:
                                result = tool_result

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )
                    parsed = json.loads(result)
                    self._record_tool_result(bool(parsed.get("ok")))
                    self._persist_messages(messages)

                    if name in self.control_tools.names:
                        yield self._run_event(control=name)
                    else:
                        result_event: dict[str, Any] = {
                            "harness_event": "tool_result",
                            "call_id": call["id"],
                            "name": name,
                            "ok": parsed.get("ok", False),
                            "error": parsed.get("error"),
                        }
                        if name in self.service.terminal_tools.names:
                            result_event["output"] = self.service.terminal_tools.display_result(
                                parsed
                            )
                        yield sse(result_event)

                    current = self.service.store.get(self.run_id)
                    if current["status"] in TERMINAL_AGENT_STATUSES:
                        self._append_skipped_results(
                            messages,
                            normalized_calls[index + 1 :],
                            f"Run entered {current['status']} status.",
                        )
                        self._persist_messages(messages)
                        self._persist_elapsed()
                        yield self._finished_event()
                        yield b"data: [DONE]\n\n"
                        return
                    reason = self._budget_reason()
                    if reason:
                        self._append_skipped_results(
                            messages,
                            normalized_calls[index + 1 :],
                            reason,
                        )
                        self._persist_messages(messages)
                        self._finish_with_status("budget_exhausted", reason)
                        yield self._finished_event()
                        yield b"data: [DONE]\n\n"
                        return
        except asyncio.CancelledError:
            self._stop_if_active("The agent stream was interrupted. Resume it manually.")
            raise
        except Exception as exc:
            self._finish_with_status("failed", str(exc), error=str(exc))
            yield sse({"harness_event": "error", "message": str(exc)})
            yield self._finished_event()
            yield b"data: [DONE]\n\n"
        finally:
            watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog
            repaired = AgentService._repair_history(messages)
            with suppress(AgentRunError):
                self._persist_messages(repaired)
                self._persist_elapsed()
                self._stop_if_active(
                    "The agent stream ended before the run completed. Resume it manually."
                )
            await self.service.release(self.run_id)

    @property
    def control_tools(self) -> AgentControlTools:
        return AgentControlTools(self.service.store, self.run_id)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            *self.service.file_tools.definitions,
            *self.service.pdf_tools.definitions,
            *self.service.web_tools.definitions,
            *self.service.terminal_tools.definitions,
            *self.service.state_tools.definitions,
            *self.control_tools.definitions,
        ]

    async def _run_external_tool(
        self,
        call: dict[str, Any],
        name: str,
        arguments: dict[str, Any],
    ):
        if name not in self.external_names:
            yield None, json.dumps({"ok": False, "error": f"Unknown tool: {name}"})
            return
        display = self._display_arguments(name, arguments)
        if self._requires_approval(name):
            previous_status = self.service.store.get(self.run_id)["status"]

            def waiting(run: dict[str, Any]) -> None:
                run["status"] = "waiting_for_approval"

            self.service.store.mutate(self.run_id, waiting)
            approval_id = self.service.approvals.register(
                name,
                str(display.get("path") or display.get("command") or ""),
            )
            yield self._run_event(), None
            yield sse(
                {
                    "harness_event": "tool_approval",
                    "approval_id": approval_id,
                    "call_id": call["id"],
                    "name": name,
                    "arguments": display,
                }
            ), None
            approved = await self._wait_for_approval(approval_id)

            def restore(run: dict[str, Any]) -> None:
                if run["status"] == "waiting_for_approval":
                    run["status"] = previous_status

            self.service.store.mutate(self.run_id, restore)
            yield self._run_event(), None
            if self._stop_requested():
                yield None, json.dumps({"ok": False, "error": "Stopped by the user."})
            elif approved:
                yield None, await self._execute_external_until_stopped(name, arguments)
            else:
                yield None, json.dumps(
                    {"ok": False, "error": "The user denied this operation."}
                )
            return

        yield sse(
            {
                "harness_event": "tool_started",
                "call_id": call["id"],
                "name": name,
                "arguments": display,
            }
        ), None
        yield None, await self._execute_external_until_stopped(name, arguments)

    async def _wait_for_approval(self, approval_id: str) -> bool:
        approval_task = asyncio.create_task(self.service.approvals.wait(approval_id))
        stop_task = asyncio.create_task(self.stop_event.wait())
        done, pending = await asyncio.wait(
            {approval_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if stop_task in done and self.stop_event.is_set():
            self.service.approvals.cancel(approval_id)
            approval_task.cancel()
            with suppress(asyncio.CancelledError):
                await approval_task
            return False
        with suppress(asyncio.CancelledError):
            await stop_task
        return approval_task.result()

    @property
    def external_names(self) -> set[str]:
        return (
            self.service.file_tools.names
            | self.service.pdf_tools.names
            | self.service.web_tools.names
            | self.service.terminal_tools.names
            | self.service.state_tools.names
        )

    def _requires_approval(self, name: str) -> bool:
        return (
            self.service.file_tools.requires_approval(name)
            or self.service.terminal_tools.requires_approval(name)
        )

    def _display_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in self.service.web_tools.names:
            return self.service.web_tools.display_arguments(name, arguments)
        if name in self.service.pdf_tools.names:
            return self.service.pdf_tools.display_arguments(name, arguments)
        if name in self.service.terminal_tools.names:
            return self.service.terminal_tools.display_arguments(name, arguments)
        if name in self.service.state_tools.names:
            return self.service.state_tools.display_arguments(name, arguments)
        return self.service.file_tools.display_arguments(name, arguments)

    async def _execute_external(self, name: str, arguments: dict[str, Any]) -> str:
        if name in self.service.web_tools.names:
            return await self.service.web_tools.execute(name, arguments)
        if name in self.service.pdf_tools.names:
            return await self.service.pdf_tools.execute(name, arguments)
        if name in self.service.terminal_tools.names:
            return await self.service.terminal_tools.execute(name, arguments)
        if name in self.service.state_tools.names:
            return self.service.state_tools.execute(name, arguments)
        return self.service.file_tools.execute(name, arguments)

    async def _open_completion(
        self,
        messages: list[dict[str, Any]],
        run: dict[str, Any],
    ):
        completion_task = asyncio.create_task(
            self.service.gateway.open_completion(
                messages=messages,
                requested_model=run["model"],
                temperature=run["temperature"],
                max_tokens=run["max_tokens"],
                tools=self.definitions,
            )
        )
        stop_task = asyncio.create_task(self.stop_event.wait())
        done, pending = await asyncio.wait(
            {completion_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if stop_task in done and self.stop_event.is_set():
            completion_task.cancel()
            with suppress(asyncio.CancelledError):
                await completion_task
            return None
        with suppress(asyncio.CancelledError):
            await stop_task
        return completion_task.result()

    async def _payloads_until_stopped(self, stream):
        payloads = stream.payloads()
        try:
            while True:
                next_task = asyncio.create_task(payloads.__anext__())
                stop_task = asyncio.create_task(self.stop_event.wait())
                done, pending = await asyncio.wait(
                    {next_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if stop_task in done and self.stop_event.is_set():
                    next_task.cancel()
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        await next_task
                    return
                with suppress(asyncio.CancelledError):
                    await stop_task
                try:
                    yield next_task.result()
                except StopAsyncIteration:
                    return
        finally:
            await payloads.aclose()

    async def _execute_external_until_stopped(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        execution_task = asyncio.create_task(self._execute_external(name, arguments))
        stop_task = asyncio.create_task(self.stop_event.wait())
        done, pending = await asyncio.wait(
            {execution_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if stop_task in done and self.stop_event.is_set():
            execution_task.cancel()
            with suppress(asyncio.CancelledError):
                await execution_task
            reason = self._budget_reason() or "Stopped by the user."
            return json.dumps({"ok": False, "error": reason})
        with suppress(asyncio.CancelledError):
            await stop_task
        return execution_task.result()

    async def _watch_elapsed_budget(self) -> None:
        run = self.service.store.get(self.run_id)
        remaining = max(
            0,
            run["budgets"]["max_seconds"] - self._base_elapsed,
        )
        await asyncio.sleep(remaining)
        self._budget_expired = True
        self.stop_event.set()

    def _run_event(self, control: str | None = None) -> bytes:
        run = AgentRunStore.public(self.service.store.get(self.run_id))
        payload: dict[str, Any] = {"harness_event": "agent_run", "run": run}
        if control:
            payload["control"] = control
        return sse(payload)

    def _finished_event(self) -> bytes:
        run = AgentRunStore.public(self.service.store.get(self.run_id))
        return sse(
            {
                "harness_event": "agent_finished",
                "run": run,
                "status": run["status"],
                "summary": run["summary"],
                "message": run["last_error"],
            }
        )

    def _increment_round(self) -> None:
        self.service.store.mutate(
            self.run_id,
            lambda run: run["usage"].__setitem__(
                "tool_rounds", run["usage"]["tool_rounds"] + 1
            ),
        )

    def _increment_tool_call(self) -> None:
        self.service.store.mutate(
            self.run_id,
            lambda run: run["usage"].__setitem__(
                "tool_calls", run["usage"]["tool_calls"] + 1
            ),
        )

    def _record_tool_result(self, ok: bool) -> None:
        def update(run: dict[str, Any]) -> None:
            usage = run["usage"]
            key = "successful_tool_calls" if ok else "failed_tool_calls"
            usage[key] += 1
            usage["consecutive_failures"] = 0 if ok else usage["consecutive_failures"] + 1

        self.service.store.mutate(self.run_id, update)

    def _persist_messages(self, messages: list[dict[str, Any]]) -> None:
        compact = AgentService._compact_messages(messages)
        self.service.store.mutate(
            self.run_id,
            lambda run: run.__setitem__("messages", compact),
        )

    def _persist_elapsed(self) -> None:
        elapsed = self._elapsed_seconds()
        self.service.store.mutate(
            self.run_id,
            lambda run: run["usage"].__setitem__("elapsed_seconds", elapsed),
        )

    def _elapsed_seconds(self) -> int:
        if not self._started:
            return self._base_elapsed
        return self._base_elapsed + int(time.monotonic() - self._started)

    def _budget_reason(self, include_tool_call: bool = False) -> str | None:
        run = self.service.store.get(self.run_id)
        usage = run["usage"]
        budgets = run["budgets"]
        if usage["tool_rounds"] >= budgets["max_tool_rounds"]:
            return f"Stopped after {budgets['max_tool_rounds']} model/tool rounds."
        if usage["tool_calls"] + int(include_tool_call) > budgets["max_tool_calls"]:
            return f"Stopped after {budgets['max_tool_calls']} tool calls."
        if self._elapsed_seconds() >= budgets["max_seconds"]:
            return f"Stopped after {budgets['max_seconds']} active seconds."
        if usage["consecutive_failures"] >= budgets["max_consecutive_failures"]:
            return (
                "Stopped after "
                f"{budgets['max_consecutive_failures']} consecutive tool failures."
            )
        return None

    def _stop_requested(self) -> bool:
        return self.stop_event.is_set() or self.service.store.get(self.run_id)[
            "stop_requested"
        ]

    def _finish_with_status(
        self,
        status: str,
        summary: str,
        *,
        error: str | None = None,
    ) -> None:
        def update(run: dict[str, Any]) -> None:
            if run["status"] == "completed":
                return
            run["status"] = status
            run["summary"] = summary
            run["last_error"] = error
            run["stop_requested"] = False
            run["usage"]["elapsed_seconds"] = self._elapsed_seconds()

        self.service.store.mutate(self.run_id, update)

    def _finish_after_interrupt(self) -> None:
        reason = self._budget_reason()
        if self._budget_expired or reason:
            self._finish_with_status(
                "budget_exhausted",
                reason or "The elapsed-time budget was exhausted.",
            )
        else:
            self._finish_with_status("stopped", "Stopped by the user.")

    def _stop_if_active(self, summary: str) -> None:
        def update(run: dict[str, Any]) -> None:
            if run["status"] in ACTIVE_AGENT_STATUSES:
                run["status"] = "stopped"
                run["summary"] = summary
                run["stop_requested"] = False

        self.service.store.mutate(self.run_id, update)

    @staticmethod
    def _append_skipped_results(
        messages: list[dict[str, Any]],
        calls: list[dict[str, Any]],
        reason: str,
    ) -> None:
        for call in calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"ok": False, "error": reason}),
                }
            )
