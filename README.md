# local-sst

Offline speech-to-text pipeline with speaker diarization. Transcribes audio locally using ONNX models from [Handy.app](https://handy.computer), with optional Groq cloud API.

## Features

- **Local ONNX transcription** — Parakeet TDT and Canary AED models, CPU-optimized (~28x real-time)
- **Speaker diarization** — pyannote-audio with interactive speaker labeling and timestamps
- **VAD segmentation** — Silero VAD with configurable speech duration
- **Session resume** — Every stage cached (transcription, diarization, WAV, speaker labels)
- **Audio preprocessing** — Loudness normalization, voice enhancement, format conversion, chunking
- **SRT output** — Subtitle files with optional speaker prefixes (`[<speaker>]: ...`)
- **Groq API** — Optional cloud transcription via `--engine groq`

## Prerequisites

- **ffmpeg** (system)
- **Python 3.11+**
- **HF_TOKEN** — HuggingFace token for diarization (accept conditions at [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0))
- **GROQ_API_KEY** — Optional, only for `--engine groq`

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install pydub python-dotenv groq "onnx-asr[cpu,hub]" pyannote-audio

cp .env.example .env
# Edit .env: add HF_TOKEN=hf_xxx (and GROQ_API_KEY if using cloud)
```

## Quick Start

```bash
# Full pipeline with local models
python scripts/audio-process -i meeting.m4a -o output/ --engine local --model parakeet -vvv

# Resume a failed run (skips completed stages)
python scripts/audio-process -i meeting.m4a -o output/ --engine local -c -vvv

# With speaker diarization (interactive, requires terminal)
# Preferred: use audio-process with --diarize
python scripts/audio-process -i meeting.m4a -o output/ --engine local --diarize -vvv

# Or standalone: audio-transcribe-local with --diarize
python scripts/audio-transcribe-local -i output/chunks -o output/transcription.srt \
  --model parakeet --diarize -vvv

# Cloud transcription via Groq
python scripts/audio-process -i meeting.m4a -o output/ --engine groq -vvv
```

## Pipeline

1. `audio-normalize` — EBU R128 loudness normalization
2. `audio-enhance` — High-pass/low-pass voice filters (ffmpeg)
3. `audio-convert` — Convert to 16kHz mono Opus
4. `audio-split` — Split by size with overlap
5. `audio-transcribe-local` (or `audio-transcribe` for Groq) — Transcribe chunks (with optional `--diarize` for speaker labeling)

Output: `transcription.srt` + `.speakers.json` + `.diarization.json` + `.diarize.wav` (all cached for re-runs)

## Individual Scripts

```bash
python scripts/audio-normalize -i input.m4a -o normalized.wav -vvv
python scripts/audio-enhance -i normalized.wav -o enhanced.wav -vvv
python scripts/audio-convert -i enhanced.wav -o output.ogg -vvv
python scripts/audio-split -i output.ogg -o chunks/ --size-mb 10 --overlap 5 -vvv
python scripts/audio-transcribe-local -i chunks/ -o output.srt --model parakeet -vvv
python scripts/audio-transcribe -i chunks/ -o output.srt -vvv  # Groq
```

## Options

| Flag | Description |
|------|-------------|
| `--engine local\|groq` | Transcription engine (default: groq) |
| `--model` | Model name, fuzzy-matched for local (e.g. `parakeet`, `canary`) |
| `--diarize` | Run speaker diarization (requires interactive terminal) |
| `--num-speakers` | Known speaker count (diarization only) |
| `-c, --continue` | Resume into existing output dir |
| `--force` | Re-run all stages ignoring cache |
| `--keep-temp` | Keep intermediate files |
| `-v, -vv, -vvv` | Verbosity levels |
| `--language` | ISO 639-1 language code hint (e.g. `en`, `es`). Auto-detects if omitted. |
| `-j, --threads` | Parallel worker threads. 0 = all cores (default: 0). |
| `--overlap` | Chunk overlap in seconds (default: 5). |
| `--size-mb` | Max chunk size in MiB (default: 10). |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | `--diarize` | HuggingFace token for pyannote models |
| `GROQ_API_KEY` | `--engine groq` | Groq API key |

## Local Models

Models are auto-discovered from Handy.app's models directory:

| Model | Type | Languages |
|-------|------|-----------|
| `parakeet-tdt-0.6b-v3-int8` | nemo-conformer-tdt | 25 European |
| `canary-1b-v2` | nemo-conformer-aed | 25 European + translation |

List available models:
```bash
python scripts/audio-transcribe-local -i dummy -o dummy
```

## License

MIT
