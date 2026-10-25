FROM python:3.12-slim

# uv resolves and installs the dependencies (a single static binary, so no extra build tools are needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so code changes don't reinstall them
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

COPY src ./src

RUN useradd --system --no-create-home app
USER app

EXPOSE 2048

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:2048/api/game-data')"

CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "2048"]
