from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import threading
from typing import Any
import uuid

from app.agent_runs import AgentRunError, utc_now
from app.observability import evaluate_assertions


EVAL_STORE_VERSION = 1
MAX_SCENARIOS = 100
MAX_REPLAYS_PER_SCENARIO = 50
MAX_EVAL_FILE_BYTES = 4 * 1024 * 1024


class EvalStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._lock = threading.Lock()

    def create_from_run(
        self,
        run: dict[str, Any],
        *,
        name: str | None = None,
        assertions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if run.get("status") not in {"completed", "failed", "blocked", "budget_exhausted", "stopped"}:
            raise AgentRunError("Create regression scenarios only from terminal runs.")
        scenario_name = (name or run["goal"][:120]).strip()
        if not scenario_name or len(scenario_name) > 200:
            raise AgentRunError("Regression scenario name must be 1-200 characters.")
        evaluation = run.get("evaluation") or {}
        usage = run.get("usage", {})
        defaults = {
            "expected_status": "completed",
            "min_quality_score": min(70, evaluation.get("quality_score", 70)),
            "min_safety_score": min(80, evaluation.get("safety_score", 80)),
            "max_cost_usd": (
                round(max(0.001, usage.get("cost_usd", 0.0) * 2), 8)
                if usage.get("cost_source") != "unpriced"
                else None
            ),
            "max_latency_ms": max(1000, usage.get("latency_ms", 0) * 2),
        }
        if assertions:
            defaults.update({key: value for key, value in assertions.items() if value is not None})
        now = utc_now()
        scenario = {
            "id": uuid.uuid4().hex,
            "version": 1,
            "name": scenario_name,
            "source_run_id": run["id"],
            "request": {
                "goal": run["goal"],
                "model": run.get("model"),
                "temperature": run.get("temperature", 0.4),
                "max_tokens": run.get("max_tokens", 4096),
                "grants": deepcopy(run.get("grants", [])),
            },
            "assertions": defaults,
            "replays": [],
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            document = self._load()
            document["scenarios"][scenario["id"]] = scenario
            if len(document["scenarios"]) > MAX_SCENARIOS:
                oldest = min(document["scenarios"].values(), key=lambda item: item["created_at"])
                del document["scenarios"][oldest["id"]]
            self._save(document)
        return deepcopy(scenario)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            scenarios = list(self._load()["scenarios"].values())
        scenarios.sort(key=lambda item: item["created_at"], reverse=True)
        return deepcopy(scenarios)

    def get(self, scenario_id: str) -> dict[str, Any]:
        with self._lock:
            scenario = self._load()["scenarios"].get(scenario_id)
        if scenario is None:
            raise AgentRunError("Regression scenario does not exist.")
        return deepcopy(scenario)

    def record_replay(self, scenario_id: str, run: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            document = self._load()
            scenario = document["scenarios"].get(scenario_id)
            if scenario is None:
                raise AgentRunError("Regression scenario does not exist.")
            result = {
                "run_id": run["id"],
                "status": run["status"],
                "usage": deepcopy(run.get("usage", {})),
                "evaluation": deepcopy(run.get("evaluation")),
                "assertions": evaluate_assertions(run, scenario["assertions"]),
                "completed_at": utc_now(),
            }
            existing = next(
                (item for item in scenario["replays"] if item["run_id"] == run["id"]),
                None,
            )
            if existing is None:
                scenario["replays"].append(result)
            else:
                existing.clear()
                existing.update(result)
            scenario["replays"] = scenario["replays"][-MAX_REPLAYS_PER_SCENARIO:]
            scenario["updated_at"] = utc_now()
            self._save(document)
            return deepcopy(result)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": EVAL_STORE_VERSION, "scenarios": {}}
        if self.path.stat().st_size > MAX_EVAL_FILE_BYTES:
            raise AgentRunError("The eval scenario store is too large.")
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AgentRunError("The eval scenario store is invalid.") from exc
        if document.get("version") != EVAL_STORE_VERSION or not isinstance(document.get("scenarios"), dict):
            raise AgentRunError("The eval scenario store has an unsupported format.")
        if len(document["scenarios"]) > MAX_SCENARIOS:
            raise AgentRunError("The eval scenario store exceeds its scenario limit.")
        for scenario_id, scenario in document["scenarios"].items():
            self._validate_scenario(scenario_id, scenario)
        return document

    @staticmethod
    def _validate_scenario(scenario_id: str, scenario: Any) -> None:
        request = scenario.get("request") if isinstance(scenario, dict) else None
        assertions = scenario.get("assertions") if isinstance(scenario, dict) else None
        replays = scenario.get("replays") if isinstance(scenario, dict) else None
        if (
            not isinstance(scenario_id, str)
            or not isinstance(scenario, dict)
            or scenario.get("id") != scenario_id
            or scenario.get("version") != 1
            or not isinstance(scenario.get("name"), str)
            or not 1 <= len(scenario["name"]) <= 200
            or not isinstance(scenario.get("source_run_id"), str)
            or not scenario["source_run_id"]
            or not isinstance(request, dict)
            or not isinstance(request.get("goal"), str)
            or not 1 <= len(request["goal"]) <= 20_000
            or not isinstance(request.get("model"), (str, type(None)))
            or isinstance(request.get("temperature"), bool)
            or not isinstance(request.get("temperature"), (int, float))
            or not 0 <= request["temperature"] <= 2
            or isinstance(request.get("max_tokens"), bool)
            or not isinstance(request.get("max_tokens"), int)
            or not 1 <= request["max_tokens"] <= 32_768
            or not isinstance(request.get("grants"), list)
            or any(grant not in {"workspace_mutations", "terminal"} for grant in request["grants"])
            or not isinstance(assertions, dict)
            or assertions.get("expected_status") not in {
                "completed", "blocked", "failed", "budget_exhausted", "stopped"
            }
            or any(
                isinstance(assertions.get(key), bool)
                or not isinstance(assertions.get(key), int)
                or not 0 <= assertions[key] <= 100
                for key in ("min_quality_score", "min_safety_score")
            )
            or (
                assertions.get("max_cost_usd") is not None
                and (
                    isinstance(assertions["max_cost_usd"], bool)
                    or not isinstance(assertions["max_cost_usd"], (int, float))
                    or assertions["max_cost_usd"] < 0
                )
            )
            or (
                assertions.get("max_latency_ms") is not None
                and (
                    isinstance(assertions["max_latency_ms"], bool)
                    or not isinstance(assertions["max_latency_ms"], int)
                    or assertions["max_latency_ms"] < 0
                )
            )
            or not isinstance(replays, list)
            or len(replays) > MAX_REPLAYS_PER_SCENARIO
            or not isinstance(scenario.get("created_at"), str)
            or not isinstance(scenario.get("updated_at"), str)
        ):
            raise AgentRunError("The eval scenario store contains an invalid scenario.")
        for replay in replays:
            if (
                not isinstance(replay, dict)
                or not isinstance(replay.get("run_id"), str)
                or not isinstance(replay.get("status"), str)
                or not isinstance(replay.get("usage"), dict)
                or not isinstance(replay.get("evaluation"), dict)
                or not isinstance(replay.get("assertions"), dict)
                or not isinstance(replay["assertions"].get("passed"), bool)
                or not isinstance(replay.get("completed_at"), str)
            ):
                raise AgentRunError("The eval scenario store contains an invalid replay.")

    def _save(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if temporary.stat().st_size > MAX_EVAL_FILE_BYTES:
                raise AgentRunError("The eval scenario store exceeds its size limit.")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
