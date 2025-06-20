FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Copy over all dependencies
COPY prompts prompts
COPY parameters.toml parameters.toml
COPY pyproject.toml pyproject.toml
COPY src src

# Set up dependencies
RUN uv pip install . --system

# Run the app
CMD ["streamlit", "run", "src/streamlit/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
