import pytest

from app.config import Settings


def test_terminal_mode_defaults_to_sandbox(monkeypatch):
    monkeypatch.delenv("MODEL_TERMINAL_MODE", raising=False)
    monkeypatch.delenv("MODEL_SANDBOX_IMAGE", raising=False)

    settings = Settings.from_env()

    assert settings.terminal_mode == "sandbox"
    assert settings.sandbox_image == "tff-harness-sandbox:latest"


def test_terminal_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("MODEL_TERMINAL_MODE", "unrestricted")

    with pytest.raises(ValueError, match="MODEL_TERMINAL_MODE"):
        Settings.from_env()


def test_agent_state_must_be_outside_model_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("MODEL_FILE_ROOT", str(workspace))
    monkeypatch.setenv("MODEL_AGENT_STATE_FILE", str(workspace / "agent.json"))

    with pytest.raises(ValueError, match="outside MODEL_FILE_ROOT"):
        Settings.from_env()


def test_memory_must_be_outside_model_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("MODEL_FILE_ROOT", str(workspace))
    monkeypatch.delenv("MODEL_AGENT_STATE_FILE", raising=False)
    monkeypatch.setenv("MODEL_MEMORY_FILE", str(workspace / "memory.json"))

    with pytest.raises(ValueError, match="outside MODEL_FILE_ROOT"):
        Settings.from_env()


def test_eval_store_must_be_outside_model_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("MODEL_FILE_ROOT", str(workspace))
    monkeypatch.delenv("MODEL_AGENT_STATE_FILE", raising=False)
    monkeypatch.delenv("MODEL_MEMORY_FILE", raising=False)
    monkeypatch.setenv("MODEL_EVAL_FILE", str(workspace / "evals.json"))

    with pytest.raises(ValueError, match="outside MODEL_FILE_ROOT"):
        Settings.from_env()


def test_model_pricing_is_bounded_and_configurable(monkeypatch):
    monkeypatch.setenv("MODEL_INPUT_COST_PER_MILLION", "2.5")
    monkeypatch.setenv("MODEL_OUTPUT_COST_PER_MILLION", "7.5")

    settings = Settings.from_env()

    assert settings.model_input_cost_per_million == 2.5
    assert settings.model_output_cost_per_million == 7.5

    monkeypatch.setenv("MODEL_INPUT_COST_PER_MILLION", "-1")
    with pytest.raises(ValueError, match="MODEL_INPUT_COST_PER_MILLION"):
        Settings.from_env()
