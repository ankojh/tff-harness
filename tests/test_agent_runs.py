import json

from app.agent_runs import AgentControlTools, AgentRunStore


def make_run(store):
    return store.create(
        "Implement the feature",
        model="model-test",
        temperature=0.4,
        max_tokens=2048,
        budgets={
            "max_tool_rounds": 10,
            "max_tool_calls": 20,
            "max_seconds": 300,
            "max_consecutive_failures": 3,
        },
        messages=[{"role": "user", "content": "Implement the feature"}],
    )


def test_agent_completion_requires_plan_and_verification(tmp_path):
    store = AgentRunStore(tmp_path / "agent.json")
    run = make_run(store)
    controls = AgentControlTools(store, run["id"])

    premature = json.loads(
        controls.execute("agent_complete", {"summary": "Finished"})
    )
    controls.execute("agent_set_plan", {"steps": ["Make the change"]})
    controls.execute(
        "agent_update_step",
        {"index": 0, "status": "completed", "note": "Change made"},
    )
    controls.execute("agent_begin_verification", {})
    unverified = json.loads(
        controls.execute("agent_complete", {"summary": "Finished"})
    )
    controls.execute(
        "agent_record_verification",
        {"evidence": "The focused test passed."},
    )
    completed = json.loads(
        controls.execute("agent_complete", {"summary": "Finished safely."})
    )

    assert premature["ok"] is False
    assert unverified["ok"] is False
    assert completed["ok"] is True
    current = store.get(run["id"])
    assert current["status"] == "completed"
    assert current["summary"] == "Finished safely."
    assert current["verification_evidence"] == ["The focused test passed."]


def test_agent_run_persists_and_active_run_becomes_stopped_after_restart(tmp_path):
    path = tmp_path / "agent.json"
    first = AgentRunStore(path)
    run = make_run(first)

    second = AgentRunStore(path)
    interrupted = second.mark_interrupted()

    assert interrupted["id"] == run["id"]
    assert interrupted["status"] == "stopped"
    assert "restarted" in interrupted["last_error"]
    assert AgentRunStore.public(interrupted)["resumable"] is True


def test_only_one_active_agent_run_is_allowed(tmp_path):
    store = AgentRunStore(tmp_path / "agent.json")
    make_run(store)

    try:
        make_run(store)
    except Exception as exc:
        assert "already active" in str(exc)
    else:
        raise AssertionError("Creating a second active run should fail")
