# AGENTS.md - Agent Guidelines for Groq STT Project

## Project Overview
Audio preprocessing pipeline for local/private transcription using Groq Speech-to-Text. Focus on Unix philosophy: one focus per script, stdio for piping, minimal dependencies.

## Dependencies
All Python packages must be installed via `uv`:
```bash
source .venv/bin/activate
uv pip install <package>
```

Required system dependency: `ffmpeg` (must be installed separately)

---

## Commands

### Environment Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install pydub python-dotenv
```

### Running Scripts
```bash
# Single script usage
python scripts/<script-name> --input audio.m4a --output ./out -vvv

# Pipe usage (when supported)
cat audio.m4a | python scripts/<script-name> -vvv > output.txt
```

### Linting
```bash
# Run ruff (if configured)
ruff check .
ruff check --fix .
```

---

## Code Style Guidelines

### General
- Shebang: `#!/usr/bin/env python3`
- Encoding: UTF-8
- Line length: max 100 characters
- No trailing whitespace

### Imports
Order (alphabetical within groups):
1. Standard library (`argparse`, `logging`, `subprocess`, etc.)
2. Third-party (`pydub`, `dotenv`)
3. Local (`src.*`)

```python
# Good
import argparse
import sys
from pathlib import Path

from pydub import AudioSegment
from dotenv import load_dotenv

from src.audio import load_audio
from src.logging import setup_logging
```

### Naming
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private functions: `_leading_underscore`

### Type Hints
Required for all function signatures:
```python
def process_audio(input_path: Path, output_dir: Path) -> int:
    audio: AudioSegment = load_audio(input_path)
    return 0
```

### Docstrings
All public functions must have docstrings:
```python
def normalize_loudness(input_path: Path, output_path: Path, target_lufs: float = -16.0) -> None:
    """Apply EBU R128 loudness normalization using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file
        target_lufs: Target loudness in LUFS (default: -16)
    """
```

### Error Handling
- Use specific exceptions (`FileNotFoundError`, `ValueError`)
- Return exit codes: `0` = success, `1` = error, `2` = invalid args
- Log errors to stderr with context
- Never expose secrets in error messages

---

## Script Design Principles

### One Focus
Each script does exactly one thing:
- `audio-normalize` - EBU R128 loudness normalization
- `audio-enhance` - Voice enhancement filters (high/low pass)
- `audio-convert` - Format/samplerate conversion
- `audio-split` - Split by size with overlap

### Interface Convention
```
<command> --input <file> --output <dir> [-v|-vv|-vvv]
```

### Output
- Print result path to stdout on success
- Print errors to stderr
- Use verbose flag `-vvv` for debug logging

---

## Secrets & Configuration

### Environment Variables
- Store in `.env` (never commit)
- Template in `.env.example`
- Use `python-dotenv` for loading

```python
from src.config import load_env, require_env

load_env()
api_key = require_env("GROQ_API_KEY")
```

---

## File Structure
```
groq-stt/
├── src/
│   ├── __init__.py
│   ├── audio.py      # Audio loading utilities
│   ├── logging.py    # Verbose logging setup
│   └── config.py     # Environment/config helpers
├── scripts/
│   ├── audio-normalize
│   ├── audio-enhance
│   ├── audio-convert
│   └── audio-split
├── .env.example
├── .venv/            # Created by uv
├── pyproject.toml    # Optional project config
└── AGENTS.md         # This file
```

---

## Testing
No formal tests yet. Manual verification:
```bash
# Test audio-normalize
python scripts/audio-normalize -i input.m4a -o output.wav -vvv

# Test full pipeline
python scripts/audio-normalize -i input.m4a -o /tmp/normalized.wav
python scripts/audio-enhance -i /tmp/normalized.wav -o /tmp/enhanced.wav
```

---

## Commit Guidelines
- Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Focus on "why", not "what"
- Example: `feat: add audio-normalize script with EBU R128 support`
- Always verify working before committing

## Agent Rules
- when finished, commit your work as atomic units (they build individually allowing git bisect)
- NEVER read `.env` or secret files
