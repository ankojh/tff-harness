from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable
import uuid


AGENT_STATUSES = {
    "planning",
    "working",
    "waiting_for_approval",
    "waiting_for_user",
    "verifying",
    "completed",
    "blocked",
    "stopped",
    "failed",
    "budget_exhausted",
}
ACTIVE_AGENT_STATUSES = {
    "planning",
    "working",
    "waiting_for_approval",
    "verifying",
}
RESUMABLE_AGENT_STATUSES = {"waiting_for_user", "blocked", "stopped", "failed"}
TERMINAL_AGENT_STATUSES = {
    "completed",
    "blocked",
    "stopped",
    "failed",
    "budget_exhausted",
    "waiting_for_user",
}
MAX_AGENT_FILE_BYTES = 4 * 1024 * 1024
MAX_AGENT_MESSAGES = 600
MAX_PLAN_STEPS = 20
MAX_PLAN_STEP_CHARS = 500
MAX_EVIDENCE_ITEMS = 30
MAX_EVIDENCE_CHARS = 1000


AGENT_SYSTEM_PROMPT = """You are operating as a supervised workspace agent responsible for completing one delegated goal.

Required lifecycle:
1. Understand the goal and inspect only relevant context. Use persistent memory selectively.
2. Call agent_set_plan before doing substantive work. Keep the plan concise and observable.
3. Execute the plan with the available workspace, web, PDF, terminal, and state tools. Mutating tools and terminal commands require user approval.
4. Call agent_update_step whenever progress changes. Adapt the plan when evidence requires it.
5. After all plan steps are complete, call agent_begin_verification. Verify the outcome with appropriate tests, builds, reads, or inspection.
6. Call agent_record_verification for each concrete piece of evidence.
7. Call agent_complete only after verification. If the goal cannot be completed without user input or authority, call agent_block.

You own accomplishing and verifying the goal. The harness owns persistence, isolation, approvals, and budgets. Never invent a broader goal, weaken safeguards, claim unverified success, or finish with ordinary prose instead of the appropriate agent lifecycle tool."""


class AgentRunError(Exception):
    """An agent run request or state transition is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentRunStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._lock = threading.Lock()

    def create(
        self,
        goal: str,
        *,
        model: str | None,
        temperature: float,
        max_tokens: int,
        budgets: dict[str, int],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            current = self._load_document().get("run")
            if current and current["status"] in ACTIVE_AGENT_STATUSES:
                raise AgentRunError("Another agent run is already active.")
            now = utc_now()
            run = {
                "id": uuid.uuid4().hex,
                "goal": goal,
                "status": "planning",
                "plan": [],
                "current_step": None,
                "verification_evidence": [],
                "summary": None,
                "last_error": None,
                "stop_requested": False,
                "created_at": now,
                "updated_at": now,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "budgets": dict(budgets),
                "usage": {
                    "tool_rounds": 0,
                    "tool_calls": 0,
                    "successful_tool_calls": 0,
                    "failed_tool_calls": 0,
                    "consecutive_failures": 0,
                    "elapsed_seconds": 0,
                },
                "messages": deepcopy(messages),
            }
            self._save_document({"version": 1, "run": run})
            return deepcopy(run)

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            run = self._load_document().get("run")
            return deepcopy(run) if run else None

    def get(self, run_id: str) -> dict[str, Any]:
        run = self.current()
        if run is None or run["id"] != run_id:
            raise AgentRunError("Agent run does not exist.")
        return run

    def mutate(
        self,
        run_id: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        with self._lock:
            document = self._load_document()
            run = document.get("run")
            if run is None or run["id"] != run_id:
                raise AgentRunError("Agent run does not exist.")
            callback(run)
            run["updated_at"] = utc_now()
            self._validate_run(run)
            self._save_document(document)
            return deepcopy(run)

    def mark_interrupted(self) -> dict[str, Any] | None:
        with self._lock:
            document = self._load_document()
            run = document.get("run")
            if run is None or run["status"] not in ACTIVE_AGENT_STATUSES:
                return deepcopy(run) if run else None
            run["status"] = "stopped"
            run["stop_requested"] = False
            run["last_error"] = (
                "The harness restarted while this run was active. Resume it manually."
            )
            run["updated_at"] = utc_now()
            self._save_document(document)
            return deepcopy(run)

    @staticmethod
    def public(run: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: deepcopy(value) for key, value in run.items() if key != "messages"
        }
        result["resumable"] = run["status"] in RESUMABLE_AGENT_STATUSES
        return result

    def _load_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "run": None}
        if self.path.stat().st_size > MAX_AGENT_FILE_BYTES:
            raise AgentRunError("The agent run state file is too large and was not loaded.")
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AgentRunError("The agent run state file is invalid.") from exc
        if not isinstance(document, dict) or document.get("version") != 1:
            raise AgentRunError("The agent run state file has an unsupported format.")
        run = document.get("run")
        if run is not None:
            self._validate_run(run)
        return document

    def _save_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if temporary.stat().st_size > MAX_AGENT_FILE_BYTES:
                raise AgentRunError("The agent run state exceeds its storage limit.")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _validate_run(run: Any) -> None:
        if not isinstance(run, dict):
            raise AgentRunError("The agent run state is invalid.")
        required_strings = {"id", "goal", "status", "created_at", "updated_at"}
        if any(not isinstance(run.get(key), str) for key in required_strings):
            raise AgentRunError("The agent run state is invalid.")
        if not run["goal"].strip() or len(run["goal"]) > 20_000:
            raise AgentRunError("The agent run has an invalid goal.")
        if run["status"] not in AGENT_STATUSES:
            raise AgentRunError("The agent run has an invalid status.")
        if (
            not isinstance(run.get("messages"), list)
            or len(run["messages"]) > MAX_AGENT_MESSAGES
        ):
            raise AgentRunError("The agent run has too many stored messages.")
        if (
            not isinstance(run.get("plan"), list)
            or len(run["plan"]) > MAX_PLAN_STEPS
        ):
            raise AgentRunError("The agent run has an invalid plan.")
        for step in run["plan"]:
            if (
                not isinstance(step, dict)
                or not isinstance(step.get("text"), str)
                or not step["text"]
                or len(step["text"]) > MAX_PLAN_STEP_CHARS
                or step.get("status") not in {"pending", "in_progress", "completed"}
            ):
                raise AgentRunError("The agent run has an invalid plan step.")
        evidence = run.get("verification_evidence")
        if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE_ITEMS:
            raise AgentRunError("The agent run has invalid verification evidence.")
        if any(
            not isinstance(item, str) or not item or len(item) > MAX_EVIDENCE_CHARS
            for item in evidence
        ):
            raise AgentRunError("The agent run has invalid verification evidence.")
        budgets = run.get("budgets")
        required_budgets = {
            "max_tool_rounds",
            "max_tool_calls",
            "max_seconds",
            "max_consecutive_failures",
        }
        if not isinstance(budgets, dict) or any(
            isinstance(budgets.get(key), bool)
            or not isinstance(budgets.get(key), int)
            or budgets[key] < 1
            for key in required_budgets
        ):
            raise AgentRunError("The agent run has invalid budgets.")
        usage = run.get("usage")
        required_usage = {
            "tool_rounds",
            "tool_calls",
            "successful_tool_calls",
            "failed_tool_calls",
            "consecutive_failures",
            "elapsed_seconds",
        }
        if not isinstance(usage, dict) or any(
            isinstance(usage.get(key), bool)
            or not isinstance(usage.get(key), int)
            or usage[key] < 0
            for key in required_usage
        ):
            raise AgentRunError("The agent run has invalid usage counters.")


class AgentControlTools:
    def __init__(self, store: AgentRunStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id

    @property
    def names(self) -> set[str]:
        return {
            "agent_set_plan",
            "agent_update_step",
            "agent_begin_verification",
            "agent_record_verification",
            "agent_complete",
            "agent_block",
        }

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            self._definition(
                "agent_set_plan",
                "Create or replace the concise execution plan for this run.",
                {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": MAX_PLAN_STEP_CHARS},
                        "minItems": 1,
                        "maxItems": MAX_PLAN_STEPS,
                    }
                },
                ["steps"],
            ),
            self._definition(
                "agent_update_step",
                "Update one plan step after its progress changes.",
                {
                    "index": {"type": "integer", "minimum": 0},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                    "note": {"type": "string", "maxLength": MAX_EVIDENCE_CHARS},
                },
                ["index", "status"],
            ),
            self._definition(
                "agent_begin_verification",
                "Enter verification after every plan step is completed.",
                {},
                [],
            ),
            self._definition(
                "agent_record_verification",
                "Record concrete verification evidence from tests, builds, reads, or inspection.",
                {"evidence": {"type": "string", "maxLength": MAX_EVIDENCE_CHARS}},
                ["evidence"],
            ),
            self._definition(
                "agent_complete",
                "Complete the run after verification evidence has been recorded.",
                {"summary": {"type": "string", "maxLength": 4000}},
                ["summary"],
            ),
            self._definition(
                "agent_block",
                "Stop because user input, authority, or an external change is required.",
                {
                    "reason": {"type": "string", "maxLength": 4000},
                    "requires_user_input": {
                        "type": "boolean",
                        "default": True,
                    },
                },
                ["reason"],
            ),
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "agent_set_plan":
                result = self._set_plan(arguments)
            elif name == "agent_update_step":
                result = self._update_step(arguments)
            elif name == "agent_begin_verification":
                result = self._begin_verification()
            elif name == "agent_record_verification":
                result = self._record_verification(arguments)
            elif name == "agent_complete":
                result = self._complete(arguments)
            elif name == "agent_block":
                result = self._block(arguments)
            else:
                raise AgentRunError(f"Unknown agent control tool: {name}")
            return json.dumps({"ok": True, **result}, ensure_ascii=False)
        except AgentRunError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    def _set_plan(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_steps = arguments.get("steps")
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_PLAN_STEPS:
            raise AgentRunError("steps must contain 1-20 plan steps.")
        steps = []
        for index, text in enumerate(raw_steps):
            if not isinstance(text, str) or not text.strip():
                raise AgentRunError("Every plan step must be a non-empty string.")
            if len(text) > MAX_PLAN_STEP_CHARS:
                raise AgentRunError("A plan step exceeds the 500-character limit.")
            steps.append(
                {
                    "text": text.strip(),
                    "status": "in_progress" if index == 0 else "pending",
                    "note": None,
                }
            )

        def update(run: dict[str, Any]) -> None:
            if run["status"] in TERMINAL_AGENT_STATUSES:
                raise AgentRunError("A terminal run cannot replace its plan.")
            run["plan"] = steps
            run["current_step"] = 0
            run["status"] = "working"

        run = self.store.mutate(self.run_id, update)
        return {"plan": run["plan"], "status": run["status"]}

    def _update_step(self, arguments: dict[str, Any]) -> dict[str, Any]:
        index = arguments.get("index")
        status = arguments.get("status")
        note = arguments.get("note")
        if isinstance(index, bool) or not isinstance(index, int):
            raise AgentRunError("index must be an integer.")
        if status not in {"pending", "in_progress", "completed"}:
            raise AgentRunError("status must be pending, in_progress, or completed.")
        if note is not None and (not isinstance(note, str) or len(note) > MAX_EVIDENCE_CHARS):
            raise AgentRunError("note must be a string of at most 1000 characters.")

        def update(run: dict[str, Any]) -> None:
            if run["status"] not in {"working", "planning"}:
                raise AgentRunError("Plan steps can only be updated while working.")
            if not 0 <= index < len(run["plan"]):
                raise AgentRunError("Plan step index is out of range.")
            run["plan"][index]["status"] = status
            run["plan"][index]["note"] = note.strip() if isinstance(note, str) else None
            in_progress = [
                position
                for position, step in enumerate(run["plan"])
                if step["status"] == "in_progress"
            ]
            pending = [
                position
                for position, step in enumerate(run["plan"])
                if step["status"] == "pending"
            ]
            run["current_step"] = (in_progress or pending or [None])[0]
            run["status"] = "working"

        run = self.store.mutate(self.run_id, update)
        return {
            "step": run["plan"][index],
            "current_step": run["current_step"],
        }

    def _begin_verification(self) -> dict[str, Any]:
        def update(run: dict[str, Any]) -> None:
            if not run["plan"] or any(
                step["status"] != "completed" for step in run["plan"]
            ):
                raise AgentRunError("Every plan step must be completed before verification.")
            run["status"] = "verifying"
            run["current_step"] = None

        run = self.store.mutate(self.run_id, update)
        return {"status": run["status"]}

    def _record_verification(self, arguments: dict[str, Any]) -> dict[str, Any]:
        evidence = arguments.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise AgentRunError("evidence must be a non-empty string.")
        if len(evidence) > MAX_EVIDENCE_CHARS:
            raise AgentRunError("evidence exceeds the 1000-character limit.")

        def update(run: dict[str, Any]) -> None:
            if run["status"] != "verifying":
                raise AgentRunError("Verification evidence requires verifying status.")
            if len(run["verification_evidence"]) >= MAX_EVIDENCE_ITEMS:
                raise AgentRunError("The verification evidence limit has been reached.")
            run["verification_evidence"].append(evidence.strip())

        run = self.store.mutate(self.run_id, update)
        return {"verification_evidence": run["verification_evidence"]}

    def _complete(self, arguments: dict[str, Any]) -> dict[str, Any]:
        summary = arguments.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise AgentRunError("summary must be a non-empty string.")
        if len(summary) > 4000:
            raise AgentRunError("summary exceeds the 4000-character limit.")

        def update(run: dict[str, Any]) -> None:
            if run["status"] != "verifying":
                raise AgentRunError("The run must be verifying before completion.")
            if not run["verification_evidence"]:
                raise AgentRunError("Record verification evidence before completion.")
            run["status"] = "completed"
            run["summary"] = summary.strip()
            run["stop_requested"] = False

        run = self.store.mutate(self.run_id, update)
        return {"status": run["status"], "summary": run["summary"]}

    def _block(self, arguments: dict[str, Any]) -> dict[str, Any]:
        reason = arguments.get("reason")
        requires_user_input = arguments.get("requires_user_input", True)
        if not isinstance(reason, str) or not reason.strip():
            raise AgentRunError("reason must be a non-empty string.")
        if len(reason) > 4000:
            raise AgentRunError("reason exceeds the 4000-character limit.")
        if not isinstance(requires_user_input, bool):
            raise AgentRunError("requires_user_input must be a boolean.")

        def update(run: dict[str, Any]) -> None:
            if run["status"] not in ACTIVE_AGENT_STATUSES:
                raise AgentRunError("Only an active run can become blocked.")
            run["status"] = "waiting_for_user" if requires_user_input else "blocked"
            run["summary"] = reason.strip()

        run = self.store.mutate(self.run_id, update)
        return {"status": run["status"], "reason": run["summary"]}

    @staticmethod
    def _definition(
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
