# local-sst

Offline speech-to-text pipeline with speaker diarization. Transcribes audio locally using transcribe.cpp from [Handy.app](https://handy.computer) (ggml-accelerated, Metal/Vulkan/CUDA/CPU), with optional Groq cloud API.

## Features

- **Local transcription** — transcribe.cpp (Handy Computer's new ggml STT engine): whisper-large-v3-turbo runs ~50-60x realtime on Apple Silicon with Metal
- **Speaker diarization** — pyannote-audio with interactive speaker labeling and timestamps
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
uv sync
source .venv/bin/activate

cp .env.example .env
# Edit .env: add HF_TOKEN=hf_xxx (and GROQ_API_KEY if using cloud)
```

## Quick Start

```bash
# Full pipeline with local models
python scripts/audio-process -i meeting.m4a -o output/ --engine local --model whisper-large-v3-turbo -vvv

# Resume a failed run (skips completed stages; --model is still required
# for the transcription stage)
python scripts/audio-process -i meeting.m4a -o output/ --engine local \
  --model whisper-large-v3-turbo -c -vvv

# With speaker diarization (interactive, requires terminal)
# Preferred: use audio-process with --diarize
python scripts/audio-process -i meeting.m4a -o output/ --engine local --diarize --num-speakers 4 -vvv

# Or standalone: audio-transcribe-local with --diarize
python scripts/audio-transcribe-local -i output/chunks -o output/transcription.srt \
  --model whisper-large-v3-turbo --diarize -vvv

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
python scripts/audio-transcribe-local -i chunks/ -o output.srt --model whisper-large-v3-turbo -vvv
python scripts/audio-transcribe -i chunks/ -o output.srt -vvv  # Groq
```

## Options

| Flag | Description |
|------|-------------|
| `--engine local\|groq` | Transcription engine (default: groq) |
| `--model` | Model name or path, fuzzy-matched for local (e.g. `whisper-large-v3-turbo`); required for `--engine local` |
| `--diarize` | Run speaker diarization (requires interactive terminal) |
| `--num-speakers` | Known speaker count (diarization only) |
| `-c, --continue` | Resume into existing output dir |
| `--force` | Re-run all stages ignoring cache |
| `--clean-temp` | Delete intermediate files after pipeline finishes |
| `-v, -vv, -vvv` | Verbosity levels |
| `--language` | ISO 639-1 language code hint (e.g. `en`, `es`). Auto-detects if omitted. |
| `-j, --threads` | Parallel worker threads. 0 = all cores (default: 0). |
| `--overlap` | Chunk overlap in seconds (default: 5). |
| `--list-models` | List available GGUF models and exit |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | `--diarize` | HuggingFace token for pyannote models |
| `GROQ_API_KEY` | `--engine groq` | Groq API key |

## Local Models

There is no default local model. Pass `-m/--model` (a GGUF path, or a short name like `cohere` fuzzy-matched against the cache) or set `TCPP_MODEL`. Models come from [handy-computer](https://huggingface.co/handy-computer) and are cached under `~/.cache/huggingface/hub/models--handy-computer--*-gguf/` (e.g. `whisper-large-v3-turbo` Q8_0, ~845 MB).

List cached handy-computer GGUF models:
```bash
python scripts/audio-transcribe-local --list-models
```

Select another GGUF with `-m <path-or-name>`, or set the `TCPP_MODEL` (model path) and `TRANSCRIBE_CLI` (binary) environment variables.

## License

Dual-licensed under the Unlicense OR the MIT License, at your option
(see `UNLICENSE` and `LICENSE`). Contributions are accepted under the
same dual-license terms.
