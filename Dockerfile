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

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "mcp[cli]>=1.2.0" fastmcp httpx uvicorn starlette

# Copy application source
COPY . /app

# Install package locally
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 9905

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=9905

CMD ["python", "-m", "isaac_mcp.server"]
