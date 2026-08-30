FROM python:3.12-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "fastapi>=0.115,<1" \
    "httpx>=0.27,<1" \
    "pypdf>=5,<7" \
    "pytest>=8,<9" \
    "pytest-asyncio>=0.24,<1" \
    "uvicorn[standard]>=0.30,<1"

WORKDIR /workspace

CMD ["/bin/sh"]
