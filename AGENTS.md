# AGENTS.md - Agent Guidelines for Local-SST Project

## Project Overview
Audio preprocessing pipeline for transcription using local ONNX models (Parakeet, Canary) from Handy.app, with optional Groq cloud API. Focus on Unix philosophy: one focus per script, stdio for piping, minimal dependencies.

## Dependencies
All Python packages must be installed via `uv`:
```bash
source .venv/bin/activate
uv pip install <package>
```

Required: `ffmpeg` (system), `pydub`, `python-dotenv`, `onnx-asr[cpu,hub]`, `pyannote-audio`

---

## Commands

### Environment Setup
```bash
uv venv
source .venv/bin/activate
uv pip install pydub python-dotenv "onnx-asr[cpu,hub]" pyannote-audio
```

### Running Scripts
```bash
python scripts/<script-name> --input audio.m4a --output ./out -vvv
```

### Linting
```bash
ruff check .
ruff check --fix .
```

---

## File Structure
```
local-sst/
├── src/
│   ├── __init__.py
│   ├── audio.py           # Audio loading utilities
│   ├── logging.py         # Verbose logging setup
│   ├── config.py          # Environment/config helpers
│   ├── srt.py             # SRT/VTT generation + parsing
│   ├── transcribe.py      # Groq cloud transcription
│   ├── local_transcribe.py # Local ONNX transcription engine
│   └── diarize.py         # Speaker diarization (pyannote)
├── scripts/
│   ├── audio-normalize
│   ├── audio-enhance
│   ├── audio-convert
│   ├── audio-split
│   ├── audio-transcribe       # Groq cloud
│   ├── audio-transcribe-local  # Local ONNX
│   └── audio-process           # Full pipeline orchestrator
├── .env.example
├── pyproject.toml
└── AGENTS.md
```

---

## Code Style

- Shebang: `#!/usr/bin/env python3`, line length max 100, UTF-8
- Imports: stdlib → third-party → local (alphabetical within groups)
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, `_leading_underscore` private
- Type hints required on all function signatures
- Docstrings required on all public functions
- Return exit codes: `0` success, `1` error, `2` invalid args
- Log errors to stderr, print results to stdout

---

## Constraints

### Audio Processing
- **Do not load entire large audio files into memory** for duration/metadata. Use `ffprobe` or read headers only. Loading a 5-hour file into pydub consumes 20GB+ RAM.
- **Use ffmpeg subprocess directly** for filter operations (highpass, lowpass, loudnorm) on large files. pydub's in-memory filters crash on files >1 hour.
- **Convert compressed formats to 16kHz mono WAV** before passing to ONNX or pyannote. Compressed formats (Opus, MP3) have imprecise sample counts that cause chunking errors.

### Pipeline
- **Cache pipeline intermediates** between stages. Allow `--continue/-c` to resume a failed run without re-computing completed stages. Do not re-run a 5-minute normalize step because the script crashed at transcription.
- **Refuse to run on non-empty output directories** without an explicit `--continue` or `--force` flag.

### Subprocess & Interactive
- **Stream child process output in real-time** when running sub-scripts. Do not use `capture_output=True` for long-running child processes — it buffers all output until completion, hiding progress.
- **Check `sys.stdin.isatty()` before any interactive prompt.** If stdin is not a terminal, fail fast with a clear error. Do not let `input()` silently hang or EOF when piped or backgrounded.
- **Handle `KeyboardInterrupt` in interactive prompts.** Save partial progress, write output, and exit cleanly. Users must be able to resume from where they stopped.

### Transcription
- **Default engine is `groq`** (cloud). Use `--engine local` for offline ONNX transcription with Parakeet/Canary models.
- Models are auto-discovered from Handy.app's models directory on every run (lazy scan).
- Diarization caches: WAV conversion (`*.diarize.wav`), diarization results (`*.diarization.json`), speaker labels (`*.speakers.json`). All three are persisted and reused on re-runs.

---

## Secrets & Configuration
- Store in `.env` (never commit). Template in `.env.example`. Use `python-dotenv` for loading.
- Required: `GROQ_API_KEY` (for Groq engine), `HF_TOKEN` (for diarization)

## Commit Guidelines
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Focus on "why", not "what"
- Always verify working before committing
- Commit as atomic units (build individually, enabling git bisect)

## Agent Rules
- NEVER read `.env` or secret files
