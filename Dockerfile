# SimLiverse MCP Server — Container Build
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml /app/
COPY README.md /app/
COPY isaac_mcp/__init__.py /app/isaac_mcp/

# Install python dependencies.
#
# The upper bound on `mcp` is load-bearing. This image is built from source on
# every worker VM at boot, so an unbounded requirement does not mean "the version
# we tested" — it means "whatever was published this morning". The SDK's 2.0
# release renamed `FastMCP` to `MCPServer`, and `isaac_mcp/server.py` imports
# `from mcp.server.fastmcp import FastMCP`. The first worker to boot after that
# release built an image whose entrypoint died on import:
#
#     ModuleNotFoundError: No module named 'mcp.server.fastmcp'
#
# The container exited 1, nothing listened on 9905, and the agent lost every
# simulation tool — while the identical source had built and run correctly the
# day before. Lift this bound only together with the migration to `MCPServer`.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "mcp[cli]>=1.2.0,<2" fastmcp httpx uvicorn starlette

# Copy application source
COPY . /app

# Install package locally
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 9905

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=9905

CMD ["python", "-m", "isaac_mcp.server"]
