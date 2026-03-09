# groq-stt

Audio preprocessing and transcription pipeline for Groq Whisper API.

## Features

- **Loudness Normalization** - EBU R128 standard for consistent audio levels
- **Voice Enhancement** - High-pass and low-pass filters to clean up audio
- **Format Conversion** - Convert to 16kHz mono Opus (optimal for Whisper)
- **Smart Chunking** - Split large files with overlap for accurate transcription
- **Parallel Transcription** - Transcribe multiple chunks concurrently via Groq API
- **SRT/VTT Output** - Generate subtitle files with timestamps

## Prerequisites

- **ffmpeg** - Must be installed system-wide
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt install ffmpeg

  # Arch
  sudo pacman -S ffmpeg
  ```

- **Python 3.11+**

- **Groq API Key** - Get from https://console.groq.com/keys

## Installation

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install pydub python-dotenv

# Copy environment template and add your API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
```

## Quick Start

### Transcribe a 1-hour meeting

```bash
# Single command does it all: normalize -> enhance -> convert -> split -> transcribe
python scripts/audio-process -i recording.m4a -o output/
```

This creates:
- `output/processed.ogg` - Processed audio
- `output/chunks/` - Split audio chunks
- `output/chunks/*.srt` - Transcribed subtitles

## Usage

### Individual Scripts

Each script is independent and follows Unix philosophy:

```bash
# Normalize loudness (EBU R128)
python scripts/audio-normalize -i input.m4a -o normalized.wav -vvv

# Enhance voice (high/low pass filters)
python scripts/audio-enhance -i normalized.wav -o enhanced.wav -vvv

# Convert to 16kHz mono Opus
python scripts/audio-convert -i enhanced.wav -o output.ogg -vvv

# Split into chunks by size (default: 10MB chunks with 5s overlap)
python scripts/audio-split -i output.ogg -o chunks/ --size-mb 10 --overlap 5 -vvv

# Transcribe using Groq API
python scripts/audio-transcribe -i chunks/ -o output.srt -vvv
```

### Full Pipeline

```bash
# Run the complete pipeline
python scripts/audio-process -i meeting.m4a -o output/

# Keep temporary files for debugging
python scripts/audio-process -i meeting.m4a -o output/ --keep-temp

# Custom chunk size and overlap
python scripts/audio-process -i meeting.m4a -o output/ --size-mb 25 --overlap 10
```

## Options

All scripts support:

| Flag | Description |
|------|-------------|
| `-v`, `-vv`, `-vvv` | Increase verbosity (info, debug, trace) |
| `-j`, `--threads` | Number of threads for ffmpeg (0 = all cores, default: 0) |

### Threading

By default, all scripts use all available CPU cores for ffmpeg operations. Override with:

```bash
python scripts/audio-normalize -i input.m4a -o output.wav -j 4
```

### Transcription Options

```bash
# Specify language (faster, more accurate)
python scripts/audio-transcribe -i chunks/ -o output.srt -l en

# Use different model
python scripts/audio-transcribe -i chunks/ -o output.srt -m whisper-large-v3

# Parallel API calls (default: 4)
python scripts/audio-transcribe -i chunks/ -o output.srt -j 8
```

## Helper Scripts

### Convert to WebVTT

```bash
# From Groq JSON output
cat output.json | python groq-to-vtt.py > output.vtt

# From Swama format
cat input.txt | python swama-to-vtt.py > output.vtt
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key (required) |
| `GROQ_MODEL` | Default Whisper model (optional) |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  audio-normalize │ ──► │  audio-enhance  │
└─────────────────┘     └─────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  audio-convert  │
                         └─────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   audio-split   │
                         └─────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │audio-transcribe │
                         └─────────────────┘
```

Each script does one thing well and can be used independently or chained together via `audio-process`.

## License

MIT