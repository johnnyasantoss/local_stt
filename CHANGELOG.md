# Changelog

## [0.1.0] — 2026-07-08

### Breaking changes

- **Local engine migration**: Replaced ONNX transcription path with [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (Handy Computer's ggml STT engine). Local transcription now uses GGUF models from the HuggingFace cache (`~/.cache/huggingface/hub/models--handy-computer--*-gguf/`) instead of Handy.app ONNX models.
- **Default local model**: Changed from the previous ONNX model to `whisper-large-v3-turbo` (Q8_0, ~845 MB). Use `--list-models` to see available GGUF models, or `--model <name>` to select one.
- **Language support**: Regional language variants (e.g. `pt-BR`) are now rejected with a warning and normalized to the base code (e.g. `pt`). The `--language` flag now accepts only base codes.
- **Diarization**: `--diarize` now requires `HF_TOKEN` in `.env` for pyannote model access.
- **CLI**: Removed `--vad-threshold` flag (Whisper has internal VAD). Added `--list-models` flag to list available GGUF models.
