FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# WORKDIR /prod

# Copy over all dependencies
COPY prompts prompts
COPY parameters.toml parameters.toml
COPY pyproject.toml pyproject.toml
COPY src src

RUN uv pip install . --system

EXPOSE 8080

CMD ["streamlit", "run", "src/streamlit/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
