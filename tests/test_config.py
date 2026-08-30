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
