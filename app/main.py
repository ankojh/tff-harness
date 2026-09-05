from __future__ import annotations

from typing import Optional

import httpx
from fastapi import FastAPI

from app.agent_loop import AgentService
from app.agent_runs import AgentRunStore
from app.approvals import ApprovalBroker
from app.config import Settings, validate_agent_state_path
from app.evals import EvalStore
from app.file_tools import FileTools
from app.model_gateway import ModelGateway
from app.memory_tools import MemoryStore
from app.pdf_tools import PdfTools
from app.routes import create_router
from app.state_tools import StateTools
from app.terminal_tools import TerminalTools
from app.tool_loop import ToolLoop
from app.web_tools import WebTools


def create_app(
    model_transport: Optional[httpx.AsyncBaseTransport] = None,
    web_transport: Optional[httpx.AsyncBaseTransport] = None,
    settings: Optional[Settings] = None,
) -> FastAPI:
    app = FastAPI(title="Local Model Harness", version="0.1.0")
    configured = settings or Settings.from_env()
    gateway = ModelGateway(configured, model_transport)
    approvals = ApprovalBroker()
    state_file = configured.model_state_file or (
        configured.model_file_root / ".harness_state.json"
    )
    agent_state_file = validate_agent_state_path(
        configured.model_file_root,
        configured.agent_state_file or (
            configured.model_file_root.parent
            / ".tff_agent_runs"
            / f"{configured.model_file_root.name}.json"
        ),
    )
    memory_file = validate_agent_state_path(
        configured.model_file_root,
        configured.memory_file or (
            configured.model_file_root.parent
            / ".tff_memory"
            / f"{configured.model_file_root.name}.json"
        ),
    )
    eval_file = validate_agent_state_path(
        configured.model_file_root,
        configured.eval_file or (
            configured.model_file_root.parent
            / ".tff_evals"
            / f"{configured.model_file_root.name}.json"
        ),
    )
    file_tools = FileTools(configured.model_file_root)
    pdf_tools = PdfTools(configured.model_file_root)
    web_tools = WebTools(web_transport)
    terminal_tools = TerminalTools(
        configured.model_file_root,
        mode=configured.terminal_mode,
        sandbox_image=configured.sandbox_image,
    )
    state_tools = StateTools(state_file)
    tool_loop = ToolLoop(
        gateway,
        file_tools,
        pdf_tools,
        web_tools,
        terminal_tools,
        state_tools,
        approvals,
    )
    agent_store = AgentRunStore(agent_state_file)
    agent_service = AgentService(
        gateway,
        file_tools,
        pdf_tools,
        web_tools,
        terminal_tools,
        state_tools,
        approvals,
        agent_store,
        {
            "max_tool_rounds": configured.agent_max_tool_rounds,
            "max_tool_calls": configured.agent_max_tool_calls,
            "max_seconds": configured.agent_max_seconds,
            "max_consecutive_failures": configured.agent_max_consecutive_failures,
        },
        MemoryStore(memory_file, configured.model_file_root),
        EvalStore(eval_file),
        input_cost_per_million=configured.model_input_cost_per_million,
        output_cost_per_million=configured.model_output_cost_per_million,
    )
    app.include_router(create_router(gateway, tool_loop, agent_service, approvals))
    return app


app = create_app()
