"""Configuration and environment handling."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def load_env(project_root: Path | None = None) -> None:
    """Load .env file from project root."""
    if project_root is None:
        project_root = get_project_root()

    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv(dotenv_path=False)


def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable value."""
    return os.getenv(key, default)


def require_env(key: str) -> str:
    """Get required environment variable, exit with error if not set."""
    value = os.getenv(key)
    if value is None:
        print(f"ERROR: Required environment variable '{key}' not set", file=sys.stderr)
        print(f"Create a .env file with {key}=your_value", file=sys.stderr)
        sys.exit(1)
    return value


def parse_file_arg(value: str) -> Path:
    """Parse file path argument."""
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def parse_output_dir(value: str) -> Path:
    """Parse output directory argument."""
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path
