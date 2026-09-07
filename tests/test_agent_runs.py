import json

from app.agent_runs import AgentControlTools, AgentRunStore
from app.config import Settings


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


def test_long_run_history_survives_restart_with_expanded_budgets(tmp_path):
    settings = Settings("http://model.test", "model-test", None)
    store = AgentRunStore(tmp_path / "agent.json")
    messages = [{"role": "user", "content": "Inspect the workspace"}]
    for index in range(settings.agent_max_tool_rounds):
        calls = [
            {"id": f"call-{index}-{number}", "type": "function", "function": {
                "name": "list_files", "arguments": "{}",
            }}
            for number in range(2)
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        messages.extend({"role": "tool", "tool_call_id": call["id"], "content": '{"ok":true}'} for call in calls)
    run = store.create(
        "Inspect the workspace", model="model-test", temperature=0.0, max_tokens=128,
        budgets={
            "max_tool_rounds": settings.agent_max_tool_rounds,
            "max_tool_calls": settings.agent_max_tool_calls,
            "max_seconds": settings.agent_max_seconds,
            "max_consecutive_failures": settings.agent_max_consecutive_failures,
        },
        messages=messages,
    )
    recovered = AgentRunStore(tmp_path / "agent.json").get(run["id"])
    assert recovered["messages"] == messages
    assert recovered["budgets"] == run["budgets"]


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
    receipt, created = store.begin_tool_receipt(
        run["id"],
        "verify-call",
        "read_file",
        {"path": "result.txt"},
        {"path": "result.txt"},
        False,
    )
    assert created is True
    store.complete_tool_receipt(
        run["id"], "verify-call", '{"ok": true}', True
    )
    unverified = json.loads(
        controls.execute("agent_complete", {"summary": "Finished"})
    )
    controls.execute(
        "agent_record_verification",
        {
            "evidence": "The focused test passed.",
            "receipt_id": receipt["id"],
        },
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
    assert current["verification_evidence"][0]["text"] == "The focused test passed."
    assert current["verification_evidence"][0]["receipt_id"] == receipt["id"]


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


def test_run_history_and_event_journal_survive_new_runs(tmp_path):
    store = AgentRunStore(tmp_path / "agent.json")
    first = make_run(store)
    store.mark_interrupted()
    second = make_run(store)

    history = store.list_runs()
    first_events = store.events(first["id"])

    assert [run["id"] for run in history] == [second["id"], first["id"]]
    assert [event["seq"] for event in first_events] == list(
        range(1, len(first_events) + 1)
    )
    assert [event["type"] for event in first_events] == [
        "run_created",
        "run_interrupted",
    ]


def test_v1_run_state_is_migrated_in_memory(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentRunStore(path)
    run = make_run(store)
    document = json.loads(path.read_text())
    legacy = document["runs"][run["id"]]
    for key in (
        "plan_revisions",
        "tool_receipts",
        "grants",
        "pending_steering",
        "worker_tasks",
        "events",
        "next_event_seq",
    ):
        legacy.pop(key)
    legacy["verification_evidence"] = ["Legacy evidence"]
    path.write_text(json.dumps({"version": 1, "run": legacy}))

    migrated = AgentRunStore(path).current()

    assert migrated["id"] == run["id"]
    assert migrated["verification_evidence"][0]["text"] == "Legacy evidence"
    assert migrated["events"][0]["type"] == "run_migrated"


def test_side_effecting_receipts_are_reused_by_fingerprint(tmp_path):
    store = AgentRunStore(tmp_path / "agent.json")
    run = make_run(store)
    arguments = {"path": "note.txt", "content": "hello"}
    first, created = store.begin_tool_receipt(
        run["id"], "call-1", "create_file", arguments, arguments, True
    )
    store.complete_tool_receipt(run["id"], "call-1", '{"ok": true}', True)
    replay, replay_created = store.begin_tool_receipt(
        run["id"], "call-2", "create_file", arguments, arguments, True
    )

    assert created is True
    assert replay_created is False
    assert replay["id"] == first["id"]
    assert replay["status"] == "completed"


def test_revising_a_plan_requires_and_records_a_reason(tmp_path):
    store = AgentRunStore(tmp_path / "agent.json")
    run = make_run(store)
    controls = AgentControlTools(store, run["id"])
    controls.execute("agent_set_plan", {"steps": ["First approach"]})

    rejected = json.loads(
        controls.execute("agent_set_plan", {"steps": ["Better approach"]})
    )
    accepted = json.loads(
        controls.execute(
            "agent_set_plan",
            {"steps": ["Better approach"], "reason": "New evidence changed the path."},
        )
    )

    assert rejected["ok"] is False
    assert accepted["ok"] is True
    revision = store.get(run["id"])["plan_revisions"][0]
    assert revision["reason"] == "New evidence changed the path."


def test_v2_state_is_migrated_to_worker_task_schema(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentRunStore(path)
    run = make_run(store)
    document = json.loads(path.read_text())
    document["version"] = 2
    document["runs"][run["id"]].pop("worker_tasks")
    path.write_text(json.dumps(document))

    migrated_store = AgentRunStore(path)
    migrated = migrated_store.current()
    migrated_store.mark_interrupted()
    saved = json.loads(path.read_text())

    assert migrated["worker_tasks"] == {}
    assert migrated["events"][-1]["data"] == {"from_version": 9, "to_version": 10}
    assert saved["version"] == 10


def test_v3_worker_records_migrate_to_v4_integration_fields(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentRunStore(path)
    run = make_run(store)
    AgentControlTools(store, run["id"]).execute(
        "agent_set_plan", {"steps": ["Inspect"]}
    )
    task = store.create_worker_tasks(
        run["id"],
        [
            {
                "title": "Legacy reviewer",
                "instruction": "Review the implementation.",
                "role": "reviewer",
                "scope": ".",
                "mode": "read_only",
            }
        ],
    )[0]
    document = json.loads(path.read_text())
    document["version"] = 3
    legacy_task = document["runs"][run["id"]]["worker_tasks"][task["id"]]
    for key in (
        "mode",
        "change_set",
        "integration_status",
        "integrated_at",
        "rolled_back_at",
    ):
        legacy_task.pop(key)
    path.write_text(json.dumps(document))

    migrated = AgentRunStore(path).worker_tasks(run["id"])[0]

    assert migrated["mode"] == "read_only"
    assert migrated["change_set"] == []
    assert migrated["integration_status"] == "not_applicable"


def test_v4_run_records_migrate_to_v5_task_graph_field(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentRunStore(path)
    run = make_run(store)
    document = json.loads(path.read_text())
    document["version"] = 4
    document["runs"][run["id"]].pop("task_graph")
    path.write_text(json.dumps(document))

    migrated_store = AgentRunStore(path)
    migrated = migrated_store.current()
    migrated_store.mark_interrupted()
    saved = json.loads(path.read_text())

    assert migrated["task_graph"] is None
    assert migrated["events"][-1]["data"] == {"from_version": 9, "to_version": 10}
    assert saved["version"] == 10


def test_v5_worker_records_migrate_to_v6_review_field(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentRunStore(path)
    run = make_run(store)
    AgentControlTools(store, run["id"]).execute(
        "agent_set_plan", {"steps": ["Inspect"]}
    )
    task = store.create_worker_tasks(
        run["id"],
        [
            {
                "title": "Legacy implementer",
                "instruction": "Prepare a private change.",
                "role": "implementer",
                "scope": ".",
                "mode": "isolated_write",
            }
        ],
    )[0]
    document = json.loads(path.read_text())
    document["version"] = 5
    document["runs"][run["id"]]["worker_tasks"][task["id"]].pop(
        "review_decision"
    )
    path.write_text(json.dumps(document))

    migrated = AgentRunStore(path).worker_tasks(run["id"])[0]

    assert migrated["review_decision"] is None


def test_v6_run_records_migrate_to_v7_context_compactions(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentRunStore(path)
    run = make_run(store)
    document = json.loads(path.read_text())
    document["version"] = 6
    document["runs"][run["id"]].pop("context_compactions")
    path.write_text(json.dumps(document))

    migrated = AgentRunStore(path).current()

    assert migrated["context_compactions"] == []
    assert migrated["events"][-1]["data"] == {
        "from_version": 9,
        "to_version": 10,
    }


def test_v7_run_records_migrate_to_v8_rich_graph_and_artifact_fields(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentRunStore(path)
    run = make_run(store)
    AgentControlTools(store, run["id"]).execute(
        "agent_set_plan", {"steps": ["Inspect"]}
    )
    task = store.create_worker_tasks(
        run["id"],
        [
            {
                "title": "Legacy worker",
                "instruction": "Inspect the workspace.",
                "role": "researcher",
                "scope": ".",
                "mode": "read_only",
            }
        ],
    )[0]
    document = json.loads(path.read_text())
    document["version"] = 7
    document["runs"][run["id"]]["worker_tasks"][task["id"]].pop("artifacts")
    document["runs"][run["id"]]["task_graph"] = None
    path.write_text(json.dumps(document))

    migrated = AgentRunStore(path).current()

    assert migrated["worker_tasks"][task["id"]]["artifacts"] == []
    assert migrated["events"][-1]["data"] == {
        "from_version": 9,
        "to_version": 10,
    }


def test_v8_run_records_migrate_to_v9_operator_metadata(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentRunStore(path)
    run = make_run(store)
    document = json.loads(path.read_text())
    document["version"] = 8
    document["runs"][run["id"]].pop("parent_run_id")
    document["runs"][run["id"]]["task_graph"] = None
    path.write_text(json.dumps(document))

    migrated = AgentRunStore(path).current()

    assert migrated["parent_run_id"] is None
    assert migrated["events"][-1]["data"] == {
        "from_version": 9,
        "to_version": 10,
    }
