# Lint files
lint:
    uv run tombi lint
    uv run ruff check
    uv check --preview-features check-command
