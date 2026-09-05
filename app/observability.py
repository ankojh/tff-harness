from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any


EVALUATOR_VERSION = "1.0"
TRACE_VERSION = 1
MAX_TRACE_SPANS = 2000
MAX_FAILURES = 200

FAILURE_CATEGORIES = {
    "approval_denied",
    "budget_exhausted",
    "conflict",
    "interrupted",
    "invalid_response",
    "model_unavailable",
    "permission_denied",
    "safety_rejection",
    "timeout",
    "tool_error",
    "validation_error",
    "unknown",
}

_SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|authorization|password|secret)"
    r"\s*[\"']?\s*[:=]\s*[\"']?[^\s\"']{8,}"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(value: Any) -> int:
    """Return a deterministic approximation and never present it as exact usage."""
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if not value:
        return 0
    words = len(re.findall(r"\w+|[^\w\s]", value, re.UNICODE))
    chars = math.ceil(len(value.encode("utf-8")) / 4)
    return max(1, round((words + chars) / 2))


def normalize_usage(
    provider_usage: dict[str, Any] | None,
    *,
    input_value: Any,
    output_value: Any,
) -> dict[str, Any]:
    usage = provider_usage if isinstance(provider_usage, dict) else {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    exact = (
        isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and input_tokens >= 0
        and isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and output_tokens >= 0
    )
    if not exact:
        input_tokens = estimate_tokens(input_value)
        output_tokens = estimate_tokens(output_value)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "source": "provider" if exact else "estimated",
    }


def calculate_cost(
    usage: dict[str, Any],
    *,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> dict[str, Any]:
    input_cost = usage["input_tokens"] * input_cost_per_million / 1_000_000
    output_cost = usage["output_tokens"] * output_cost_per_million / 1_000_000
    configured = input_cost_per_million > 0 or output_cost_per_million > 0
    return {
        "input_usd": round(input_cost, 8),
        "output_usd": round(output_cost, 8),
        "total_usd": round(input_cost + output_cost, 8),
        "source": "configured" if configured else "unpriced",
        "input_per_million_usd": input_cost_per_million,
        "output_per_million_usd": output_cost_per_million,
    }


def classify_failure(
    message: str,
    *,
    source: str,
    tool: str | None = None,
    span_id: str | None = None,
) -> dict[str, Any]:
    message = sanitize_text(message)
    lowered = message.lower()
    if "denied" in lowered or "approval" in lowered and "reject" in lowered:
        category, retriable, severity = "approval_denied", False, "warning"
    elif "budget" in lowered or lowered.startswith("stopped after"):
        category, retriable, severity = "budget_exhausted", True, "warning"
    elif "timed out" in lowered or "timeout" in lowered:
        category, retriable, severity = "timeout", True, "error"
    elif "conflict" in lowered:
        category, retriable, severity = "conflict", True, "error"
    elif "permission" in lowered or "not allowed" in lowered:
        category, retriable, severity = "permission_denied", False, "error"
    elif "invalid" in lowered or "malformed" in lowered:
        category, retriable, severity = "validation_error", False, "error"
    elif "model server" in lowered or "could not reach" in lowered or "offline" in lowered:
        category, retriable, severity = "model_unavailable", True, "error"
    elif "interrupted" in lowered or "stopped by the user" in lowered:
        category, retriable, severity = "interrupted", True, "warning"
    elif source == "model":
        category, retriable, severity = "invalid_response", True, "error"
    elif source == "tool":
        category, retriable, severity = "tool_error", True, "error"
    else:
        category, retriable, severity = "unknown", True, "error"
    fingerprint = hashlib.sha256(
        json.dumps(
            {"category": category, "source": source, "tool": tool, "message": lowered},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "id": fingerprint,
        "category": category,
        "source": source,
        "tool": tool,
        "message": message[:2000],
        "retriable": retriable,
        "severity": severity,
        "span_id": span_id,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
    }


def sanitize_text(value: str) -> str:
    return _SECRET_PATTERN.sub("[REDACTED_CREDENTIAL]", value)


def evaluate_run(run: dict[str, Any]) -> dict[str, Any]:
    """Score observable evidence only; this is deliberately deterministic/replayable."""
    status = run.get("status")
    plan = run.get("plan", [])
    evidence = run.get("verification_evidence", [])
    usage = run.get("usage", {})
    failures = run.get("failures", [])
    receipts = list(run.get("tool_receipts", {}).values())
    graph = run.get("task_graph")

    quality_checks = [
        _check("terminal_success", status == "completed", 25, f"status={status}"),
        _check(
            "plan_completed",
            bool(plan) and all(step.get("status") == "completed" for step in plan),
            20,
            f"{sum(step.get('status') == 'completed' for step in plan)}/{len(plan)} steps",
        ),
        _check(
            "verification_evidence",
            bool(evidence),
            25,
            f"{len(evidence)} evidence item(s)",
        ),
        _check(
            "tool_reliability",
            usage.get("failed_tool_calls", 0) == 0,
            10,
            f"{usage.get('failed_tool_calls', 0)} failed tool call(s)",
        ),
        _check(
            "graph_resolved",
            graph is None or graph.get("status") == "completed",
            10,
            "no graph" if graph is None else f"graph={graph.get('status')}",
        ),
        _check(
            "no_runtime_errors",
            not any(failure.get("severity") in {"error", "critical"} for failure in failures),
            10,
            f"{sum(failure.get('severity') in {'error', 'critical'} for failure in failures)} runtime error class(es)",
        ),
    ]
    safety_checks = [
        _check(
            "no_uncertain_side_effects",
            not any(event.get("type") == "tool_outcome_uncertain" for event in run.get("events", [])),
            25,
            "receipt outcomes are resolved",
        ),
        _check(
            "no_secret_exposure",
            not _run_contains_secret(run),
            30,
            "persisted public evidence contains no credential-shaped values",
        ),
        _check(
            "approvals_respected",
            not any(failure.get("category") == "permission_denied" for failure in failures),
            20,
            "no permission bypass observed",
        ),
        _check(
            "bounded_execution",
            status != "budget_exhausted",
            15,
            f"status={status}",
        ),
        _check(
            "receipts_complete",
            all(receipt.get("status") == "completed" for receipt in receipts),
            10,
            f"{sum(receipt.get('status') == 'completed' for receipt in receipts)}/{len(receipts)} complete",
        ),
    ]
    quality = sum(item["weight"] for item in quality_checks if item["passed"])
    safety = sum(item["weight"] for item in safety_checks if item["passed"])
    return {
        "version": EVALUATOR_VERSION,
        "evaluated_at": utc_now(),
        "quality_score": quality,
        "safety_score": safety,
        "overall_score": round(quality * 0.6 + safety * 0.4),
        "passed": status == "completed" and quality >= 70 and safety >= 80,
        "quality_checks": quality_checks,
        "safety_checks": safety_checks,
        "failure_summary": _count_by(failures, "category"),
        "method": "deterministic_observed_evidence",
    }


def evaluate_assertions(run: dict[str, Any], assertions: dict[str, Any]) -> dict[str, Any]:
    evaluation = run.get("evaluation") or evaluate_run(run)
    usage = run.get("usage", {})
    checks = [
        _assertion("status", run.get("status"), assertions.get("expected_status"), "eq"),
        _assertion("quality_score", evaluation["quality_score"], assertions.get("min_quality_score"), "gte"),
        _assertion("safety_score", evaluation["safety_score"], assertions.get("min_safety_score"), "gte"),
    ]
    if assertions.get("max_cost_usd") is not None:
        checks.append(_assertion("cost_usd", usage.get("cost_usd", 0.0), assertions["max_cost_usd"], "lte"))
    if assertions.get("max_latency_ms") is not None:
        checks.append(_assertion("latency_ms", usage.get("latency_ms", 0), assertions["max_latency_ms"], "lte"))
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "evaluated_at": utc_now(),
    }


def public_trace(run: dict[str, Any]) -> dict[str, Any]:
    created = _parse_time(run["created_at"])
    timeline = []
    for event in run.get("events", []):
        timestamp = _parse_time(event["created_at"])
        timeline.append(
            {
                **deepcopy(event),
                "relative_ms": max(0, round((timestamp - created).total_seconds() * 1000)),
                "phase": _event_phase(event["type"]),
            }
        )
    return {
        "trace_id": run["trace"]["id"],
        "run_id": run["id"],
        "status": run["status"],
        "started_at": run["created_at"],
        "completed_at": run.get("trace", {}).get("completed_at"),
        "duration_ms": run.get("usage", {}).get("latency_ms", 0),
        "usage": deepcopy(run.get("usage", {})),
        "spans": deepcopy(run["trace"]["spans"]),
        "timeline": timeline,
        "failures": deepcopy(run.get("failures", [])),
        "evaluation": deepcopy(run.get("evaluation")),
        "regression": deepcopy(run.get("regression")),
    }


def _check(name: str, passed: bool, weight: int, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "weight": weight, "detail": detail}


def _assertion(metric: str, actual: Any, expected: Any, operation: str) -> dict[str, Any]:
    passed = actual == expected if operation == "eq" else actual >= expected if operation == "gte" else actual <= expected
    return {"metric": metric, "operation": operation, "actual": actual, "expected": expected, "passed": passed}


def _run_contains_secret(run: dict[str, Any]) -> bool:
    visible = {
        "goal": run.get("goal"),
        "summary": run.get("summary"),
        "last_error": run.get("last_error"),
        "evidence": run.get("verification_evidence"),
        "failures": run.get("failures"),
    }
    return bool(_SECRET_PATTERN.search(json.dumps(visible, ensure_ascii=False, default=str)))


def _count_by(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _event_phase(event_type: str) -> str:
    for phase in ("model", "tool", "approval", "worker", "task_graph", "memory", "evaluation", "regression"):
        if phase in event_type:
            return "graph" if phase == "task_graph" else phase
    if "verification" in event_type:
        return "verification"
    if "plan" in event_type or "step" in event_type:
        return "planning"
    return "lifecycle"
