from __future__ import annotations

from typing import Optional

import httpx
from fastapi import FastAPI

from app.approvals import ApprovalBroker
from app.config import Settings
from app.file_tools import FileTools
from app.model_gateway import ModelGateway
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
    tool_loop = ToolLoop(
        gateway,
        FileTools(configured.model_file_root),
        PdfTools(configured.model_file_root),
        WebTools(web_transport),
        TerminalTools(
            configured.model_file_root,
            mode=configured.terminal_mode,
            sandbox_image=configured.sandbox_image,
        ),
        StateTools(state_file),
        approvals,
    )
    app.include_router(create_router(gateway, tool_loop, approvals))
    return app


app = create_app()
