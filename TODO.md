# TODO: Refactor Audio Processing Pipeline

## Project Overview
Refactor the audio processing pipeline to follow Unix philosophy: one focus per script, stdio for piping, proper error handling, and environment-based configuration.

## Current State
- `clear-transcript-audio.py` (405 lines) - Monolithic script doing everything
- `groq-to-vtt.py` - Already uses stdin (good)
- `swama-to-vtt.py` - Another conversion script
- `.venv/` - uv virtual environment
- No `.env` support yet

## Requirements
- Always use `uv` for package management
- No system changes (use virtual environment)
- Focus on the folder
- One focus per script (Linux philosophy)
- stdio for piping
- Keys in `.env`

---

## Implementation Plan

### Phase 1: Core Utilities & Configuration

- [ ] **1.1** Create `src/__init__.py` - Package marker
- [ ] **1.2** Create `src/audio.py` - Shared audio utilities (load, detect format)
- [ ] **1.3** Create `src/logging.py` - Verbose logging setup (-v, -vv, -vvv)
- [ ] **1.4** Create `src/config.py` - Load .env, argparse helpers
- [ ] **1.5** Create `.env.example` - Template for environment variables

### Phase 2: Standalone Scripts (One Focus Each)

- [ ] **2.1** Create `audio-normalize` - EBU R128 loudness normalization
  - Input: stdin (audio file path) or arg
  - Output: stdout (normalized audio path)
  - Args: --target-lufs, --verbose
  
- [ ] **2.2** Create `audio-enhance` - Voice enhancement filters
  - Input: stdin or arg
  - Output: stdout (enhanced audio path)
  - Args: --high-pass, --low-pass, --verbose
  
- [ ] **2.3** Create `audio-convert` - Convert to 16kHz mono Opus
  - Input: stdin or arg  
  - Output: stdout (converted audio path)
  - Args: --format, --verbose
  
- [ ] **2.4** Create `audio-split` - Split by size with overlap
  - Input: stdin or arg
  - Output: stdout (list of chunk paths)
  - Args: --size-mb, --overlap-seconds, --verbose

### Phase 3: Pipeline Orchestration

- [ ] **3.1** Create `audio-process` - Main orchestrator
  - Runs all stages in sequence
  - Can use stdin/stdout for piping
  - Args: --input, --output, --keep-temp, --verbose
  
- [ ] **3.2** Update `groq-to-vtt.py` to use .env for API key
- [ ] **3.3** Create `transcribe-pipeline` - Full pipeline (process + transcribe + convert)

### Phase 4: Project Configuration

- [ ] **4.1** Create `pyproject.toml` with uv configuration
- [ ] **4.2** Create `AGENTS.md` (~150 lines)
  - Build/lint/test commands using uv
  - Code style guidelines
  - Script design principles
  - Secrets handling
- [ ] **4.3** Add ruff configuration for linting

### Phase 5: Cleanup

- [ ] **5.1** Remove old `clear-transcript-audio.py` (replace with new scripts)
- [ ] **5.2** Update README.md if exists
- [ ] **5.3** Test full pipeline end-to-end
- [ ] **5.4** Commit all changes

---

## Script Interface Convention

```
# Single script usage
<command> --input audio.m4a --output ./out -vvv

# Pipe usage (stdin/stdout)
cat audio.m4a | <command> --verbose > output.txt

# Pipeline usage
audio-process -f input.m4a -o ./output | transcribe-pipeline --model whisper-large-v3
```

## Exit Codes
- `0` - Success
- `1` - General error
- `2` - Invalid arguments

## Dependencies
All installed via `uv`:
- pydub
- python-dotenv
- ffmpeg (system requirement)
