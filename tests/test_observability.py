import json

from app.agent_runs import AgentControlTools, AgentRunStore
from app.evals import EvalStore
from app.observability import (
    calculate_cost,
    classify_failure,
    normalize_usage,
    public_trace,
)


def make_terminal_run(tmp_path):
    store = AgentRunStore(tmp_path / "agent.json")
    run = store.create(
        "Verify observability",
        model="model-test",
        temperature=0.4,
        max_tokens=2048,
        budgets={
            "max_tool_rounds": 10,
            "max_tool_calls": 20,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
        messages=[{"role": "user", "content": "Verify observability"}],
    )
    controls = AgentControlTools(store, run["id"])
    controls.execute("agent_set_plan", {"steps": ["Inspect the trace"]})
    controls.execute("agent_update_step", {"index": 0, "status": "completed"})
    controls.execute("agent_begin_verification", {})
    receipt, _ = store.begin_tool_receipt(
        run["id"],
        "inspect-trace",
        "read_file",
        {"path": "trace.json"},
        {"path": "trace.json"},
        False,
    )
    store.complete_tool_receipt(run["id"], "inspect-trace", '{"ok": true}', True)
    controls.execute(
        "agent_record_verification",
        {
            "evidence": "The persisted trace was inspected.",
            "receipt_id": receipt["id"],
        },
    )
    controls.execute("agent_complete", {"summary": "Observability verified."})
    return store, run["id"]


def test_trace_usage_failure_and_automated_evaluation_are_durable(tmp_path):
    store, run_id = make_terminal_run(tmp_path)
    span = store.start_span(run_id, "model", "chat_completion")
    usage = normalize_usage(
        {"prompt_tokens": 120, "completion_tokens": 30},
        input_value="ignored",
        output_value="ignored",
    )
    cost = calculate_cost(
        usage,
        input_cost_per_million=2.0,
        output_cost_per_million=4.0,
    )
    store.finish_span(
        run_id,
        span["id"],
        status="error",
        duration_ms=25,
        usage=usage,
        cost=cost,
        error="Tool returned invalid data.",
    )
    store.record_model_usage(run_id, usage, cost, 25)
    store.record_failure(
        run_id,
        classify_failure(
            "Tool returned invalid data.",
            source="tool",
            tool="read_file",
            span_id=span["id"],
        ),
    )
    finalized = store.finalize_observability(run_id)
    reloaded = AgentRunStore(store.path).get(run_id)
    trace = public_trace(reloaded)

    assert finalized["evaluation"]["quality_score"] == 90
    assert finalized["evaluation"]["safety_score"] == 100
    assert reloaded["usage"]["total_tokens"] == 150
    assert reloaded["usage"]["cost_usd"] == 0.00036
    assert reloaded["failures"][0]["category"] == "validation_error"
    assert trace["spans"][1]["duration_ms"] == 25
    assert trace["timeline"][-1]["type"] == "evaluation_completed"


def test_estimated_tokens_are_explicit_and_regression_results_are_replayable(tmp_path):
    estimated = normalize_usage(
        None,
        input_value=[{"role": "user", "content": "Inspect the repository"}],
        output_value="Done",
    )
    assert estimated["source"] == "estimated"
    assert estimated["total_tokens"] > 0

    store, run_id = make_terminal_run(tmp_path)
    run = store.finalize_observability(run_id)
    evals = EvalStore(tmp_path / "evals.json")
    scenario = evals.create_from_run(run, name="observability regression")
    result = evals.record_replay(scenario["id"], run)
    reloaded = EvalStore(evals.path).get(scenario["id"])

    assert result["assertions"]["passed"] is True
    assert reloaded["request"]["goal"] == run["goal"]
    assert reloaded["replays"][0]["run_id"] == run_id


def test_v9_run_state_migrates_to_v10_observability_schema(tmp_path):
    store, run_id = make_terminal_run(tmp_path)
    document = json.loads(store.path.read_text())
    document["version"] = 9
    legacy = document["runs"][run_id]
    for key in ("trace", "failures", "evaluation", "regression"):
        legacy.pop(key)
    for key in (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "provider_token_calls",
        "estimated_token_calls",
        "latency_ms",
        "model_latency_ms",
        "tool_latency_ms",
        "cost_usd",
        "cost_source",
    ):
        legacy["usage"].pop(key)
    store.path.write_text(json.dumps(document))

    migrated = AgentRunStore(store.path).get(run_id)

    assert migrated["trace"]["version"] == 1
    assert migrated["failures"] == []
    assert migrated["usage"]["total_tokens"] == 0
    assert migrated["events"][-1]["data"] == {"from_version": 9, "to_version": 10}
