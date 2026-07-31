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

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mcp httpx uvicorn starlette

# Copy application source
COPY . /app

# Install package locally
RUN pip install --no-cache-dir -e .

EXPOSE 9905

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=9905

CMD ["python", "-m", "isaac_mcp.server"]
