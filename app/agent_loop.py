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
    MAX_PENDING_STEERING,
    RESUMABLE_AGENT_STATUSES,
    TERMINAL_AGENT_STATUSES,
    AgentControlTools,
    AgentRunError,
    AgentRunStore,
)
from app.agent_workers import AgentTaskGraphTools, AgentWorkerTools
from app.approvals import ApprovalBroker, ApprovalResolution
from app.context_compaction import ContextCompactor
from app.evals import EvalStore
from app.file_tools import FileTools
from app.memory_tools import (
    MEMORY_SYSTEM_PROMPT,
    MemoryStore,
    MemoryToolError,
    MemoryTools,
)
from app.model_gateway import ModelGateway, ModelGatewayError
from app.observability import (
    calculate_cost,
    classify_failure,
    normalize_usage,
    public_trace,
    sanitize_text,
)
from app.pdf_tools import PdfTools
from app.schemas import (
    AgentCloneRequest,
    AgentResumeRequest,
    AgentStartRequest,
    AgentSteerRequest,
)
from app.state_tools import STATE_SYSTEM_PROMPT, StateTools
from app.terminal_tools import TerminalTools
from app.tool_loop import ToolRun, sse
from app.web_tools import WebTools
from app.worker_workspaces import WorkerWorkspaceManager


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
        memory_store: MemoryStore | None = None,
        eval_store: EvalStore | None = None,
        *,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        self.gateway = gateway
        self.file_tools = file_tools
        self.pdf_tools = pdf_tools
        self.web_tools = web_tools
        self.terminal_tools = terminal_tools
        self.state_tools = state_tools
        self.approvals = approvals
        self.store = store
        self.memory_store = memory_store or MemoryStore(
            store.path.parent / f".{store.path.stem}.memory.json",
            file_tools.root,
        )
        self.context_compactor = ContextCompactor(self.memory_store)
        self.eval_store = eval_store or EvalStore(
            store.path.parent / f".{store.path.stem}.evals.json"
        )
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.budgets = dict(budgets)
        self.store.mark_interrupted()
        self._recover_worker_integrations()
        self._lock = asyncio.Lock()
        self._active_id: str | None = None
        self._stop_events: dict[str, asyncio.Event] = {}

    def _recover_worker_integrations(self) -> None:
        manager = WorkerWorkspaceManager(self.store, self.file_tools.root)
        for run in self.store.list_runs():
            for task in run["worker_tasks"]:
                if task["integration_status"] != "pending":
                    continue
                error = None
                try:
                    outcome = manager.recover_pending_integration(run["id"], task)
                except AgentRunError as exc:
                    outcome = "conflict"
                    error = str(exc)
                if outcome is None:
                    continue

                def update(record: dict[str, Any]) -> None:
                    if outcome == "conflict":
                        record["integration_status"] = "conflict"
                        record["error"] = error or (
                            "An interrupted integration could not be safely restored."
                        )

                self.store.mutate_worker_task(
                    run["id"],
                    task["id"],
                    update,
                    event_type=(
                        "worker_integration_recovery_conflict"
                        if outcome == "conflict"
                        else "worker_integration_reverted_after_restart"
                    ),
                    event_data={"outcome": outcome, "error": error},
                )

    async def create(self, request: AgentStartRequest) -> "AgentExecution":
        async with self._lock:
            if self._active_id is not None:
                raise AgentRunError("Another agent run is already active.")
            goal = request.goal.strip()
            if not goal:
                raise AgentRunError("Agent goal must not be blank.")
            if "terminal" in request.grants:
                if self.terminal_tools.mode == "disabled":
                    raise AgentRunError(
                        "Terminal capability cannot be granted while terminal tools are disabled."
                    )
                ready, error = await self.terminal_tools.readiness()
                if ready is False:
                    raise AgentRunError(
                        "Terminal capability cannot be granted because the sandbox "
                        f"is not ready: {error}"
                    )
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{STATE_SYSTEM_PROMPT}\n\n{MEMORY_SYSTEM_PROMPT}\n\n"
                        f"{AGENT_SYSTEM_PROMPT}"
                    ),
                },
                {"role": "user", "content": goal},
            ]
            try:
                await asyncio.to_thread(self.memory_store.refresh_repository)
            except (MemoryToolError, OSError) as exc:
                raise AgentRunError(f"Could not initialize workspace memory: {exc}") from exc
            run = self.store.create(
                goal,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                budgets=self.budgets,
                messages=messages,
                grants=request.grants,
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
            messages, compaction = self.context_compactor.compact(run, messages)

            def update(record: dict[str, Any]) -> None:
                record["messages"] = self._compact_messages(messages)
                if compaction is not None:
                    record["context_compactions"].append(compaction)
                record["status"] = "working" if record["plan"] else "planning"
                record["stop_requested"] = False
                record["last_error"] = None
                record["summary"] = None

            self.store.mutate(
                run_id,
                update,
                event_type="run_resumed",
                event_data={"instruction": instruction or None},
                make_current=True,
            )
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

            run = self.store.mutate(
                run_id,
                request_stop,
                event_type="stop_requested",
            )
            stop_event = self._stop_events.get(run_id)
            if stop_event is not None:
                stop_event.set()
            return AgentRunStore.public(run)

    async def steer(
        self,
        run_id: str,
        request: AgentSteerRequest,
    ) -> dict[str, Any]:
        instruction = request.instruction.strip()
        if not instruction:
            raise AgentRunError("Steering instruction must not be blank.")
        async with self._lock:
            run = self.store.get(run_id)
            if self._active_id != run_id or run["status"] not in ACTIVE_AGENT_STATUSES:
                raise AgentRunError("Only an active agent run can be steered.")
            def queue_steering(record: dict[str, Any]) -> None:
                if len(record["pending_steering"]) >= MAX_PENDING_STEERING:
                    raise AgentRunError("The pending steering limit has been reached.")
                record["pending_steering"].append(instruction)

            run = self.store.mutate(
                run_id,
                queue_steering,
                event_type="steering_queued",
                event_data={"instruction": instruction},
            )
            return AgentRunStore.public(run)

    def current(self) -> dict[str, Any] | None:
        run = self.store.current()
        return AgentRunStore.public(run) if run else None

    def history(self) -> list[dict[str, Any]]:
        return self.store.list_runs()

    def run(self, run_id: str) -> dict[str, Any]:
        return AgentRunStore.public(self.store.get(run_id))

    async def clone(
        self,
        run_id: str,
        request: AgentCloneRequest,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._active_id is not None:
                raise AgentRunError("Stop the active agent run before cloning a run.")
            source = self.store.get(run_id)
            goal = request.goal.strip() if request.goal else source["goal"]
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{STATE_SYSTEM_PROMPT}\n\n{MEMORY_SYSTEM_PROMPT}\n\n"
                        f"{AGENT_SYSTEM_PROMPT}"
                    ),
                },
                {"role": "user", "content": goal},
            ]
            clone = self.store.create(
                goal,
                model=source["model"],
                temperature=source["temperature"],
                max_tokens=source["max_tokens"],
                budgets=source["budgets"],
                messages=messages,
                grants=source["grants"],
            )

            def mark_cloned(record: dict[str, Any]) -> None:
                record["parent_run_id"] = source["id"]
                record["status"] = "stopped"
                record["summary"] = "Fresh clone ready to resume."

            clone = self.store.mutate(
                clone["id"],
                mark_cloned,
                event_type="run_cloned",
                event_data={"source_run_id": source["id"]},
                make_current=True,
            )
            return AgentRunStore.public(clone)

    def compare(self, left_id: str, right_id: str) -> dict[str, Any]:
        if left_id == right_id:
            raise AgentRunError("Choose two different runs to compare.")
        left = self.store.get(left_id)
        right = self.store.get(right_id)
        left_public = AgentRunStore.public(left)
        right_public = AgentRunStore.public(right)

        def summary(run: dict[str, Any]) -> dict[str, Any]:
            graph = run.get("task_graph")
            workers = run.get("worker_tasks", [])
            return {
                "id": run["id"],
                "goal": run["goal"],
                "status": run["status"],
                "parent_run_id": run.get("parent_run_id"),
                "created_at": run["created_at"],
                "updated_at": run["updated_at"],
                "usage": deepcopy(run["usage"]),
                "plan": [
                    {"text": item["text"], "status": item["status"]}
                    for item in run.get("plan", [])
                ],
                "verification_count": len(run.get("verification_evidence", [])),
                "event_count": run.get("event_count", 0),
                "worker_count": len(workers),
                "worker_statuses": self._count_by(workers, "status"),
                "graph_status": graph.get("status") if graph else None,
                "graph_revision": graph.get("revision") if graph else None,
                "graph_template": (
                    f"{graph.get('template')}@{graph.get('template_version')}"
                    if graph and graph.get("template")
                    else None
                ),
                "evaluation": deepcopy(run.get("evaluation")),
                "failure_count": run.get("failure_count", 0),
            }

        left_summary = summary(left_public)
        right_summary = summary(right_public)
        fields = (
            "goal",
            "status",
            "usage",
            "plan",
            "verification_count",
            "event_count",
            "worker_count",
            "worker_statuses",
            "graph_status",
            "graph_revision",
            "graph_template",
            "evaluation",
            "failure_count",
        )
        differences = [
            {
                "field": field,
                "left": deepcopy(left_summary[field]),
                "right": deepcopy(right_summary[field]),
            }
            for field in fields
            if left_summary[field] != right_summary[field]
        ]
        left_nodes = (left_public.get("task_graph") or {}).get("nodes", {})
        right_nodes = (right_public.get("task_graph") or {}).get("nodes", {})
        node_differences = []
        for key in sorted(set(left_nodes) | set(right_nodes)):
            left_node = left_nodes.get(key)
            right_node = right_nodes.get(key)
            left_state = self._node_comparison(left_node)
            right_state = self._node_comparison(right_node)
            if left_state != right_state:
                node_differences.append(
                    {"key": key, "left": left_state, "right": right_state}
                )
        return {
            "left": left_summary,
            "right": right_summary,
            "differences": differences,
            "node_differences": node_differences,
        }

    def debug(self, run_id: str) -> dict[str, Any]:
        run = self.store.get(run_id)
        public = AgentRunStore.public(run)
        graph = public.get("task_graph")
        nodes = list((graph or {}).get("nodes", {}).values())
        workers = public["worker_tasks"]
        receipts = public["tool_receipts"]
        return {
            "run": public,
            "diagnostics": {
                "state_validated": True,
                "open_receipts": [
                    item["id"] for item in receipts if item["status"] == "started"
                ],
                "failed_receipts": [
                    item["id"]
                    for item in receipts
                    if item["status"] == "completed" and item.get("ok") is False
                ],
                "pending_integrations": [
                    item["id"]
                    for item in workers
                    if item["integration_status"] in {"pending", "conflict"}
                ],
                "failed_nodes": [
                    item["key"]
                    for item in nodes
                    if item["status"]
                    in {"failed", "blocked", "conflict", "discarded", "rolled_back"}
                ],
                "paused_nodes": [
                    item["key"] for item in nodes if item["status"] == "paused"
                ],
                "root_budget_percent": {
                    key.removeprefix("max_"): round(
                        100 * run["usage"].get(key.removeprefix("max_"), 0) / maximum,
                        1,
                    )
                    for key, maximum in run["budgets"].items()
                    if maximum > 0
                },
                "graph_budget_percent": {
                    key.removeprefix("max_"): round(
                        100
                        * graph["usage"].get(key.removeprefix("max_"), 0)
                        / maximum,
                        1,
                    )
                    for key, maximum in (graph or {}).get("budgets", {}).items()
                    if maximum > 0
                },
                "failure_summary": deepcopy((run.get("evaluation") or {}).get("failure_summary", {})),
                "trace_span_count": len(run["trace"]["spans"]),
                "token_accounting": (
                    "mixed"
                    if run["usage"].get("provider_token_calls", 0)
                    and run["usage"].get("estimated_token_calls", 0)
                    else "provider"
                    if run["usage"].get("provider_token_calls", 0)
                    else "estimated"
                ),
            },
            "recent_events": deepcopy(run["events"][-100:]),
        }

    def trace(self, run_id: str) -> dict[str, Any]:
        run = self.store.get(run_id)
        if run.get("evaluation") is None and run["status"] in TERMINAL_AGENT_STATUSES:
            run = self._finalize_run(run_id)
        return public_trace(run)

    def evaluation(self, run_id: str) -> dict[str, Any]:
        run = self.store.get(run_id)
        if run.get("evaluation") is None and run["status"] in TERMINAL_AGENT_STATUSES:
            run = self._finalize_run(run_id)
        if run.get("evaluation") is None:
            raise AgentRunError("The run is still active and has not been evaluated.")
        return deepcopy(run["evaluation"])

    def scenarios(self) -> list[dict[str, Any]]:
        return self.eval_store.list()

    def scenario(self, scenario_id: str) -> dict[str, Any]:
        return self.eval_store.get(scenario_id)

    def create_scenario(
        self,
        run_id: str,
        *,
        name: str | None,
        assertions: dict[str, Any],
    ) -> dict[str, Any]:
        run = self.store.get(run_id)
        if run.get("evaluation") is None and run["status"] in TERMINAL_AGENT_STATUSES:
            run = self._finalize_run(run_id)
        return self.eval_store.create_from_run(run, name=name, assertions=assertions)

    async def replay_scenario(self, scenario_id: str) -> "AgentExecution":
        scenario = self.eval_store.get(scenario_id)
        request = scenario["request"]
        execution = await self.create(
            AgentStartRequest(
                goal=request["goal"],
                model=request.get("model"),
                temperature=request.get("temperature", 0.4),
                max_tokens=request.get("max_tokens", 4096),
                grants=request.get("grants", []),
            )
        )

        def mark(record: dict[str, Any]) -> None:
            record["regression"] = {
                "scenario_id": scenario_id,
                "replay_of_run_id": scenario["source_run_id"],
            }

        self.store.mutate(
            execution.run_id,
            mark,
            event_type="regression_replay_started",
            event_data={"scenario_id": scenario_id, "source_run_id": scenario["source_run_id"]},
        )
        return execution

    def _finalize_run(self, run_id: str) -> dict[str, Any]:
        run = self.store.finalize_observability(run_id)
        regression = run.get("regression")
        replay_evaluated = any(
            event["type"] == "regression_replay_evaluated" for event in run["events"]
        )
        if regression is not None and not replay_evaluated:
            result = self.eval_store.record_replay(regression["scenario_id"], run)
            run = self.store.record_event(
                run_id,
                "regression_replay_evaluated",
                {
                    "scenario_id": regression["scenario_id"],
                    "passed": result["assertions"]["passed"],
                },
            )
        return run

    @staticmethod
    def _count_by(items: list[dict[str, Any]], field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            value = str(item.get(field))
            counts[value] = counts.get(value, 0) + 1
        return counts

    @staticmethod
    def _node_comparison(node: dict[str, Any] | None) -> dict[str, Any] | None:
        if node is None:
            return None
        return {
            "title": node["title"],
            "kind": node["kind"],
            "status": node["status"],
            "attempt_count": node.get("attempt_count", 0),
            "artifact_count": len(node.get("artifacts", [])),
            "error": node.get("error"),
        }

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        if after < 0:
            raise AgentRunError("after must be zero or greater.")
        return self.store.events(run_id, after)

    def tasks(self, run_id: str) -> list[dict[str, Any]]:
        return [
            AgentRunStore.public_worker_task(task)
            for task in self.store.worker_tasks(run_id)
        ]

    def graph(self, run_id: str) -> dict[str, Any] | None:
        return AgentRunStore.public_task_graph(self.store.get(run_id))

    def graph_templates(self) -> list[dict[str, Any]]:
        return AgentTaskGraphTools.templates()

    def graph_artifact(
        self,
        run_id: str,
        key: str,
        name: str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        return self._worker_tools(run_id).graphs._read_artifact(
            {"key": key, "name": name, "task_id": task_id}
        )

    async def control_graph_node(
        self,
        run_id: str,
        key: str,
        *,
        action: str,
        reason: str,
        cascade: bool,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._active_id is not None:
                raise AgentRunError(
                    "Stop the active agent run before using direct graph controls."
                )
            run = self.store.get(run_id)
            if run["status"] not in RESUMABLE_AGENT_STATUSES:
                raise AgentRunError("Direct graph controls require a stopped or resumable run.")
            return await self._worker_tools(run_id).graphs._control_node(
                {
                    "key": key,
                    "action": action,
                    "reason": reason,
                    "cascade": cascade,
                    "_defer_advance": True,
                }
            )

    async def replace_graph_node(
        self,
        run_id: str,
        key: str,
        *,
        replacement: dict[str, Any],
        reason: str,
        cascade: bool,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._active_id is not None:
                raise AgentRunError(
                    "Stop the active agent run before replacing a graph node directly."
                )
            run = self.store.get(run_id)
            if run["status"] not in RESUMABLE_AGENT_STATUSES:
                raise AgentRunError("Direct graph controls require a stopped or resumable run.")
            return await self._worker_tools(run_id).graphs._replace_node(
                {
                    "key": key,
                    "replacement": replacement,
                    "reason": reason,
                    "cascade": cascade,
                    "_defer_advance": True,
                }
            )

    def _worker_tools(self, run_id: str) -> AgentWorkerTools:
        self.store.get(run_id)
        return AgentWorkerTools(
            self.gateway,
            self.file_tools,
            self.pdf_tools,
            self.web_tools,
            self.state_tools,
            self.store,
            run_id,
            self.terminal_tools,
            self.memory_store,
            self.input_cost_per_million,
            self.output_cost_per_million,
        )

    def review(self, run_id: str, task_id: str) -> dict[str, Any]:
        task = next(
            (
                item
                for item in self.store.worker_tasks(run_id)
                if item["id"] == task_id
            ),
            None,
        )
        if task is None:
            raise AgentRunError("Worker task does not exist.")
        return WorkerWorkspaceManager(self.store, self.file_tools.root).review(
            run_id,
            task,
        )

    def memory_status(self) -> dict[str, Any]:
        return self.memory_store.status()

    def search_memory(
        self,
        query: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 5,
        include_stale: bool = False,
    ) -> dict[str, Any]:
        return self.memory_store.search(
            query,
            kinds=kinds,
            limit=limit,
            include_stale=include_stale,
        )

    def read_memory(self, memory_id: str, *, allow_stale: bool = False) -> dict[str, Any]:
        return self.memory_store.read(memory_id, allow_stale=allow_stale)

    def rate_memory_lesson(
        self,
        memory_id: str,
        *,
        rating: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self.memory_store.rate_lesson(
            memory_id,
            rating=rating,
            reason=reason,
        )

    def consolidate_memory_lessons(self) -> dict[str, Any]:
        return self.memory_store.consolidate_lessons()

    def refresh_memory_index(self) -> dict[str, Any]:
        return self.memory_store.refresh_repository()

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

                steering = self._drain_steering()
                if steering:
                    for instruction in steering:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"User steering for this active run: {instruction}",
                            }
                        )
                    self._persist_messages(messages)
                    yield self._run_event()

                self._increment_round()
                model_started = time.monotonic()
                model_span = self.service.store.start_span(
                    self.run_id,
                    "model",
                    "chat_completion",
                    attributes={
                        "model": run.get("model"),
                        "round": self.service.store.get(self.run_id)["usage"]["tool_rounds"],
                    },
                )
                try:
                    stream = await self._open_completion(messages, run)
                except ModelGatewayError as exc:
                    self._complete_model_span(
                        model_span,
                        model_started,
                        messages,
                        "",
                        {},
                        None,
                        status="error",
                        error=str(exc),
                    )
                    self._finish_with_status("failed", str(exc), error=str(exc))
                    yield sse({"harness_event": "error", "message": str(exc)})
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return
                if stream is None:
                    self._complete_model_span(
                        model_span,
                        model_started,
                        messages,
                        "",
                        {},
                        None,
                        status="cancelled",
                        error="Model request was interrupted.",
                    )
                    self._finish_after_interrupt()
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return

                content = ""
                calls: dict[int, dict[str, Any]] = {}
                provider_usage: dict[str, Any] | None = None
                first_response_ms: int | None = None
                try:
                    async for payload in self._payloads_until_stopped(stream):
                        if first_response_ms is None and (
                            payload.get("choices") or payload.get("usage")
                        ):
                            first_response_ms = round((time.monotonic() - model_started) * 1000)
                        if isinstance(payload.get("usage"), dict):
                            provider_usage = payload["usage"]
                        ToolRun._accumulate(payload, calls)
                        content += ToolRun._content(payload)
                        yield sse(payload)
                except ModelGatewayError as exc:
                    self._complete_model_span(
                        model_span,
                        model_started,
                        messages,
                        content,
                        calls,
                        provider_usage,
                        status="error",
                        error=str(exc),
                        first_response_ms=first_response_ms,
                    )
                    self._finish_with_status("failed", str(exc), error=str(exc))
                    yield sse({"harness_event": "error", "message": str(exc)})
                    yield self._finished_event()
                    yield b"data: [DONE]\n\n"
                    return

                self._complete_model_span(
                    model_span,
                    model_started,
                    messages,
                    content,
                    calls,
                    provider_usage,
                    status="cancelled" if self._stop_requested() else "ok",
                    error="Model stream was interrupted." if self._stop_requested() else None,
                    first_response_ms=first_response_ms,
                )

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
                    tool_started = time.monotonic()
                    tool_span = self.service.store.start_span(
                        self.run_id,
                        "tool",
                        name,
                        attributes={"call_id": call["id"]},
                    )
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
                    tool_latency_ms = round((time.monotonic() - tool_started) * 1000)
                    tool_ok = bool(parsed.get("ok"))
                    tool_error = parsed.get("error") if not tool_ok else None
                    self.service.store.finish_span(
                        self.run_id,
                        tool_span["id"],
                        status="ok" if tool_ok else "error",
                        duration_ms=tool_latency_ms,
                        error=str(tool_error) if tool_error else None,
                        attributes={"receipt_id": parsed.get("receipt_id")},
                    )
                    self.service.store.add_tool_latency(self.run_id, tool_latency_ms)
                    if not tool_ok:
                        self.service.store.record_failure(
                            self.run_id,
                            classify_failure(
                                str(tool_error or "Tool execution failed."),
                                source="tool",
                                tool=name,
                                span_id=tool_span["id"],
                            ),
                        )
                    self._record_tool_result(tool_ok)
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
                        if current["status"] == "completed":
                            self._capture_completed_run()
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
    def worker_tools(self) -> AgentWorkerTools:
        return self.service._worker_tools(self.run_id)

    @property
    def memory_tools(self) -> MemoryTools:
        return MemoryTools(
            self.service.memory_store,
            self.service.store,
            self.run_id,
        )

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            *self.service.file_tools.definitions,
            *self.service.pdf_tools.definitions,
            *self.service.web_tools.definitions,
            *self.service.terminal_tools.definitions,
            *self.service.state_tools.definitions,
            *self.memory_tools.definitions,
            *self.worker_tools.definitions,
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

            self.service.store.mutate(
                self.run_id,
                waiting,
                event_type="approval_requested",
                event_data={"tool": name, "call_id": call["id"]},
            )
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
            resolution = await self._wait_for_approval(approval_id)
            approved = resolution.approved
            if (
                approved
                and name == "agent_integrate_worker"
                and resolution.selection is not None
            ):
                arguments = {**arguments, **resolution.selection}
                display = self._display_arguments(name, arguments)

            def restore(run: dict[str, Any]) -> None:
                if run["status"] == "waiting_for_approval":
                    run["status"] = previous_status

            self.service.store.mutate(
                self.run_id,
                restore,
                event_type="approval_decided",
                event_data={
                    "tool": name,
                    "call_id": call["id"],
                    "approved": approved,
                    "selection": resolution.selection,
                },
            )
            yield self._run_event(), None
            if self._stop_requested():
                yield None, json.dumps({"ok": False, "error": "Stopped by the user."})
            elif approved:
                yield None, await self._execute_idempotent(
                    call,
                    name,
                    arguments,
                    display,
                )
            else:
                yield None, json.dumps(
                    {"ok": False, "error": "The user denied this operation."}
                )
            return

        grant = self._grant_for_tool(name)
        if grant:
            self.service.store.record_event(
                self.run_id,
                "capability_grant_used",
                {"grant": grant, "tool": name, "call_id": call["id"]},
            )

        yield sse(
            {
                "harness_event": "tool_started",
                "call_id": call["id"],
                "name": name,
                "arguments": display,
            }
        ), None
        yield None, await self._execute_idempotent(
            call,
            name,
            arguments,
            display,
        )

    async def _wait_for_approval(self, approval_id: str) -> ApprovalResolution:
        approval_task = asyncio.create_task(
            self.service.approvals.wait_resolution(approval_id)
        )
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
            return ApprovalResolution(False)
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
            | self.memory_tools.names
            | self.worker_tools.names
        )

    def _requires_approval(self, name: str) -> bool:
        return self._grant_for_tool(name) is None and (
            self.service.file_tools.requires_approval(name)
            or self.service.terminal_tools.requires_approval(name)
            or self.worker_tools.requires_approval(name)
        )

    def _grant_for_tool(self, name: str) -> str | None:
        grants = set(self.service.store.get(self.run_id)["grants"])
        if (
            name in self.service.terminal_tools.names
            and "terminal" in grants
        ):
            return "terminal"
        if (
            self.service.file_tools.requires_approval(name)
            and "workspace_mutations" in grants
        ):
            return "workspace_mutations"
        if (
            self.worker_tools.requires_approval(name)
            and "workspace_mutations" in grants
        ):
            return "workspace_mutations"
        return None

    def _display_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in self.service.web_tools.names:
            return self.service.web_tools.display_arguments(name, arguments)
        if name in self.service.pdf_tools.names:
            return self.service.pdf_tools.display_arguments(name, arguments)
        if name in self.service.terminal_tools.names:
            return self.service.terminal_tools.display_arguments(name, arguments)
        if name in self.service.state_tools.names:
            return self.service.state_tools.display_arguments(name, arguments)
        if name in self.memory_tools.names:
            return self.memory_tools.display_arguments(name, arguments)
        if name in self.worker_tools.names:
            return self.worker_tools.display_arguments(name, arguments)
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
        if name in self.memory_tools.names:
            return await asyncio.to_thread(self.memory_tools.execute, name, arguments)
        if name in self.worker_tools.names:
            return await self.worker_tools.execute(name, arguments)
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

    async def _execute_idempotent(
        self,
        call: dict[str, Any],
        name: str,
        arguments: dict[str, Any],
        display: dict[str, Any],
    ) -> str:
        side_effecting = (
            self.service.file_tools.requires_approval(name)
            or name in self.service.terminal_tools.names
            or name in {"write_state", "delete_state"}
            or name in {
                "agent_delegate_tasks",
                "agent_integrate_worker",
                "agent_rollback_worker",
                "agent_discard_worker",
            }
        )
        receipt, created = self.service.store.begin_tool_receipt(
            self.run_id,
            call["id"],
            name,
            arguments,
            display,
            side_effecting,
        )
        if not created:
            if receipt["status"] == "completed" and receipt.get("result") is not None:
                self.service.store.record_event(
                    self.run_id,
                    "tool_result_reused",
                    {"receipt_id": receipt["id"], "tool": name},
                )
                return receipt["result"]

            reason = (
                "A previous identical side-effecting tool execution was interrupted "
                "after it started. Its outcome is uncertain, so it was not executed again."
            )

            def wait_for_user(run: dict[str, Any]) -> None:
                run["status"] = "waiting_for_user"
                run["summary"] = reason

            self.service.store.mutate(
                self.run_id,
                wait_for_user,
                event_type="tool_outcome_uncertain",
                event_data={"receipt_id": receipt["id"], "tool": name},
            )
            return json.dumps(
                {
                    "ok": False,
                    "error": reason,
                    "receipt_id": receipt["id"],
                    "outcome_uncertain": True,
                }
            )

        result = await self._execute_external_until_stopped(name, arguments)
        if self._stop_requested():
            return json.dumps(
                {
                    "ok": False,
                    "error": self._budget_reason() or "Stopped by the user.",
                    "receipt_id": receipt["id"],
                    "outcome_uncertain": True,
                }
            )
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = {"ok": False, "error": "Tool returned invalid JSON."}
        payload["receipt_id"] = receipt["id"]
        decorated = json.dumps(payload, ensure_ascii=False)
        self.service.store.complete_tool_receipt(
            self.run_id,
            call["id"],
            decorated,
            bool(payload.get("ok")),
        )
        return decorated

    async def _watch_elapsed_budget(self) -> None:
        run = self.service.store.get(self.run_id)
        remaining = max(
            0,
            run["budgets"]["max_seconds"] - self._base_elapsed,
        )
        await asyncio.sleep(remaining)
        self._budget_expired = True
        self.stop_event.set()

    def _drain_steering(self) -> list[str]:
        return self.service.store.take_steering(self.run_id)

    def _run_event(self, control: str | None = None) -> bytes:
        run = AgentRunStore.public(self.service.store.get(self.run_id))
        payload: dict[str, Any] = {"harness_event": "agent_run", "run": run}
        if control:
            payload["control"] = control
        return sse(payload)

    def _finished_event(self) -> bytes:
        stored = self.service.store.get(self.run_id)
        if stored["status"] in TERMINAL_AGENT_STATUSES:
            stored = self.service._finalize_run(self.run_id)
        run = AgentRunStore.public(stored)
        return sse(
            {
                "harness_event": "agent_finished",
                "run": run,
                "status": run["status"],
                "summary": run["summary"],
                "message": run["last_error"],
            }
        )

    def _capture_completed_run(self) -> None:
        run = self.service.store.get(self.run_id)
        if run["status"] != "completed":
            return
        with suppress(MemoryToolError, OSError, AgentRunError):
            result = self.service.memory_store.save_run(run)
            if result is not None:
                self.service.store.record_event(
                    self.run_id,
                    "run_memory_captured",
                    {
                        "memory_id": result["id"],
                        "artifacts_indexed": result.get("artifacts_indexed", 0),
                        "artifact_memory_ids": result.get("artifact_memory_ids", []),
                    },
                )

    def _complete_model_span(
        self,
        span: dict[str, Any],
        started: float,
        messages: list[dict[str, Any]],
        content: str,
        calls: dict[int, dict[str, Any]],
        provider_usage: dict[str, Any] | None,
        *,
        status: str,
        error: str | None,
        first_response_ms: int | None = None,
    ) -> None:
        latency_ms = round((time.monotonic() - started) * 1000)
        ordered_calls = [calls[index] for index in sorted(calls)]
        usage = normalize_usage(
            provider_usage,
            input_value=messages,
            output_value={"content": content, "tool_calls": ordered_calls},
        )
        cost = calculate_cost(
            usage,
            input_cost_per_million=self.service.input_cost_per_million,
            output_cost_per_million=self.service.output_cost_per_million,
        )
        self.service.store.finish_span(
            self.run_id,
            span["id"],
            status=status,
            duration_ms=latency_ms,
            usage=usage,
            cost=cost,
            error=error,
            attributes={"first_response_ms": first_response_ms},
        )
        self.service.store.record_model_usage(self.run_id, usage, cost, latency_ms)
        if error:
            self.service.store.record_failure(
                self.run_id,
                classify_failure(error, source="model", span_id=span["id"]),
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
        run = self.service.store.get(self.run_id)
        compacted, record = self.service.context_compactor.compact(run, messages)
        compact = AgentService._compact_messages(compacted)
        messages[:] = deepcopy(compact)

        def update(stored: dict[str, Any]) -> None:
            stored["messages"] = compact
            if record is not None:
                stored["context_compactions"].append(record)

        self.service.store.mutate(
            self.run_id,
            update,
            event_type="context_compacted" if record is not None else None,
            event_data=record,
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

        current = self.service.store.get(self.run_id)
        if current["status"] == "completed":
            return
        self.service.store.mutate(
            self.run_id,
            update,
            event_type="run_status_changed",
            event_data={"status": status, "summary": summary, "error": error},
        )
        if status == "budget_exhausted":
            self.service.store.record_failure(
                self.run_id,
                classify_failure(summary, source="budget"),
            )
        elif status == "failed" and error:
            sanitized = sanitize_text(error)
            current = self.service.store.get(self.run_id)
            if not any(item["message"] == sanitized for item in current["failures"]):
                self.service.store.record_failure(
                    self.run_id,
                    classify_failure(sanitized, source="harness"),
                )

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

        current = self.service.store.get(self.run_id)
        if current["status"] in ACTIVE_AGENT_STATUSES:
            self.service.store.mutate(
                self.run_id,
                update,
                event_type="run_status_changed",
                event_data={"status": "stopped", "summary": summary},
            )

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
