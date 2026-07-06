"""Local transcription via transcribe.cpp (Handy Computer's ggml STT engine).

transcribe.cpp is the new Handy Computer release that replaces the previous
ONNX transcription path. It runs GGUF models on Metal (Apple Silicon), Vulkan,
CUDA, and a tinyBLAS-accelerated CPU path, and is substantially faster than
ONNX on the same hardware. Whisper-large-v3-turbo at Q8_0 on an M-series Mac
runs ~55-60x realtime and emits native segment timestamps absolute to the
input audio (no external VAD, no manual chunking -- Whisper 30s-windows long
audio internally).
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from src.audio import get_audio_duration, load_audio_as_wav16k
from src.srt import deduplicate_segments

# Default transcribe.cpp checkout / build / model locations. Override with
# env vars for non-standard layouts.
DEFAULT_CLI = Path("../transcribe.cpp/build/bin/transcribe-cli")
DEFAULT_MODEL_GLOB = (
    "~/.cache/huggingface/hub/"
    "models--handy-computer--whisper-large-v3-turbo-gguf/snapshots/*/"
    "whisper-large-v3-turbo-Q8_0.gguf"
)

AUDIO_EXTENSIONS = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".webm", ".mp4"}


def resolve_cli() -> Path:
    """Resolve the transcribe-cli binary path. Fail fast if missing."""
    p = Path(os.environ.get("TRANSCRIBE_CLI", DEFAULT_CLI)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"transcribe-cli not found at {p}. "
            "Build transcribe.cpp (cmake -B build && cmake --build build) "
            "or set TRANSCRIBE_CLI to the binary path."
        )
    return p


def resolve_model(query: str | None = None) -> Path:
    """Resolve the GGUF model path. Fail fast if missing.

    Precedence: explicit query (path or short name) > TCPP_MODEL env > default
    whisper-large-v3-turbo Q8_0 in the HF cache.
    """
    if query:
        cand = Path(query).expanduser()
        if cand.is_file():
            return cand
        # short name like "whisper-large-v3-turbo" -> resolve from HF cache
        short = query.removesuffix(".gguf")
        for hit in (
            Path("~/.cache/huggingface/hub")
            .expanduser()
            .glob(f"models--handy-computer--{short}-gguf/snapshots/*/*.gguf")
        ):
            return hit
        raise FileNotFoundError(f"Model '{query}' not found as a path or HF cache entry.")

    env_model = os.environ.get("TCPP_MODEL")
    if env_model:
        cand = Path(env_model).expanduser()
        if not cand.is_file():
            raise FileNotFoundError(f"TCPP_MODEL={env_model} does not exist.")
        return cand

    hits = list(
        Path("~/.cache/huggingface/hub")
        .expanduser()
        .glob(
            "models--handy-computer--whisper-large-v3-turbo-gguf/snapshots/*/"
            "whisper-large-v3-turbo-Q8_0.gguf"
        )
    )
    if hits:
        return hits[-1]

    raise FileNotFoundError(
        "No GGUF model found. Download whisper-large-v3-turbo-Q8_0.gguf from "
        "https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf or set "
        "TCPP_MODEL to the path."
    )


def discover_models() -> list[dict]:
    """Scan the HuggingFace cache for handy-computer GGUF models.

    Returns a list of {name, path, size_mb} dicts, one per GGUF file found
    under ~/.cache/huggingface/hub/models--handy-computer--*-gguf/snapshots/*.
    """
    hub = Path("~/.cache/huggingface/hub").expanduser()
    if not hub.is_dir():
        return []

    models = []
    for repo in sorted(hub.glob("models--handy-computer--*-gguf")):
        short = repo.name.removeprefix("models--handy-computer--").removesuffix("-gguf")
        snapshots = repo / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in sorted(snapshots.iterdir()):
            if not snap.is_dir():
                continue
            for gguf in sorted(snap.glob("*.gguf")):
                models.append(
                    {
                        "name": short,
                        "path": gguf,
                        "quant": _quant_from_name(gguf.name),
                        "size_mb": gguf.stat().st_size / (1024 * 1024),
                    }
                )
    return models


def _quant_from_name(name: str) -> str:
    """Extract quantization tag from GGUF filename (e.g. 'Q8_0', 'F16')."""
    stem = name.removesuffix(".gguf")
    parts = stem.split("-")
    for p in parts:
        if p.startswith(("Q", "F", "I")):
            return p
    return "F32"


def _run_cli(
    cli: Path,
    model: Path,
    batch_file: Path,
    language: str | None,
    logger: logging.Logger,
) -> list[dict]:
    """Run transcribe-cli in batch-jsonl mode and return parsed per-file results.

    Returns a list of dicts: {"file": str, "segments": [...], "text": str, "error": str?}.
    The CLI loads the model once and reuses it across all files in the batch.
    """
    cmd = [
        str(cli),
        "-m",
        str(model),
        "--timestamps",
        "auto",
        "--batch",
        str(batch_file),
        "--batch-jsonl",
    ]
    if language:
        cmd += ["-l", language]

    logger.info(f"Running transcribe-cli: {' '.join(cmd[:3])} ... --batch {batch_file.name}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    results = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        # The CLI emits JSONL on stdout interleaved with [info]/[debug] log lines on
        # stderr-merged. Only lines starting with '{' are JSONL.
        if not line.startswith("{"):
            logger.debug(f"[tcpp] {line}")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"[tcpp] non-json: {line}")
            continue
        if obj.get("type") == "batch_header":
            logger.info(f"[tcpp] model loaded in {obj.get('load_ms', 0):.0f} ms")
            continue
        results.append(obj)
    proc.wait()
    if proc.returncode != 0:
        errs = [r["error"] for r in results if r.get("error")]
        detail = "; ".join(errs) if errs else "see [tcpp] log lines above"
        raise RuntimeError(f"transcribe-cli exited {proc.returncode}: {detail}")
    return results


def _segs_from_jsonl(obj: dict, offset_s: float = 0.0) -> list[dict]:
    """Convert one JSONL per-file object to our segment schema."""
    out = []
    for s in obj.get("segments", []):
        out.append(
            {
                "start": s["t0_ms"] / 1000.0 + offset_s,
                "end": s["t1_ms"] / 1000.0 + offset_s,
                "text": s.get("text", ""),
            }
        )
    return out


def _normalize_language(language: str | None, logger: logging.Logger) -> str | None:
    """Normalize a BCP-47-ish tag to the base code transcribe-cli expects.

    transcribe-cli accepts ISO 639-1/639-3 base codes (e.g. 'pt', 'en', 'yue');
    regional variants like 'pt-BR' or 'en-US' are rejected with 'unsupported
    language'. Strip the region subtag and warn.
    """
    if not language:
        return None
    short = language.replace("_", "-").split("-")[0]
    if short != language:
        logger.warning(
            f"Language '{language}' normalized to '{short}' "
            f"(transcribe-cli accepts base codes only, not regional tags)"
        )
    return short


def transcribe_local(
    input_path: Path,
    model_query: str | None = None,
    language: str | None = None,
    vad_threshold: float = 0.5,  # ignored: Whisper has internal VAD
    overlap_secs: float = 5.0,
    parallel: int = 1,  # ignored: CLI loads model once and serializes files
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Transcribe audio via transcribe.cpp. Returns [{"start","end","text"}].

    Args:
        input_path: Single audio file or directory of chunks.
        model_query: GGUF path or short model name. None = default turbo Q8_0.
        language: ISO 639-1 hint. None = auto-detect.
        vad_threshold: ignored (Whisper internal VAD). Kept for call-site compat.
        overlap_secs: overlap between chunks for offset math (chunk-dir mode).
        parallel: ignored (CLI serializes). Kept for call-site compat.
        logger: Logger instance.

    Returns:
        Segments [{"start": float, "end": float, "text": str}].
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    cli = resolve_cli()
    model = resolve_model(model_query)
    logger.info(f"Using transcribe.cpp: {cli.name}  model={model.parent.name}/{model.name}")
    language = _normalize_language(language, logger)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if input_path.is_file():
        return _transcribe_single_file(cli, model, input_path, language, logger)
    elif input_path.is_dir():
        return _transcribe_directory(cli, model, input_path, language, overlap_secs, logger)


def _transcribe_single_file(
    cli: Path,
    model: Path,
    file_path: Path,
    language: str | None,
    logger: logging.Logger,
) -> list[dict]:
    """Transcribe a single audio file. Whisper chunks long audio internally."""
    tmp_dir, wav_path = load_audio_as_wav16k(file_path)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as bf:
            bf.write(str(wav_path) + "\n")
            batch_file = Path(bf.name)
        try:
            results = _run_cli(cli, model, batch_file, language, logger)
        finally:
            batch_file.unlink(missing_ok=True)

        if not results:
            return []
        return _segs_from_jsonl(results[0])
    finally:
        tmp_dir.cleanup()


def _transcribe_directory(
    cli: Path,
    model: Path,
    input_dir: Path,
    language: str | None,
    overlap_secs: float,
    logger: logging.Logger,
) -> list[dict]:
    """Transcribe a directory of audio chunks with time offsets.

    The CLI loads the model once and transcribes all files in one batch, which
    is far cheaper than re-loading per chunk.
    """
    audio_files = sorted(
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in: {input_dir}")
    logger.info(f"Found {len(audio_files)} chunks")

    # Convert each chunk to 16k mono WAV and write a batch list. Keep temp dirs
    # alive for the duration of the CLI run.
    tmp_dirs = []
    wav_paths = []
    try:
        for f in audio_files:
            td, wp = load_audio_as_wav16k(f)
            tmp_dirs.append(td)
            wav_paths.append(wp)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as bf:
            for wp in wav_paths:
                bf.write(str(wp) + "\n")
            batch_file = Path(bf.name)
        try:
            results = _run_cli(cli, model, batch_file, language, logger)
        finally:
            batch_file.unlink(missing_ok=True)
    finally:
        for td in tmp_dirs:
            td.cleanup()

    # Calculate cumulative offsets from original chunk durations (pre-conversion)
    offsets = _calculate_chunk_offsets(audio_files, overlap_secs, logger)
    logger.info(f"Chunk offsets: {offsets[:5]}... (first 5)")

    # Map results back to chunks by file name. The CLI preserves batch order.
    all_segments = []
    for i, (obj, offset_s) in enumerate(zip(results, offsets, strict=True)):
        if "error" in obj:
            logger.error(f"  [{i + 1}/{len(audio_files)}] {audio_files[i].name}: {obj['error']}")
            continue
        segs = _segs_from_jsonl(obj, offset_s)
        all_segments.extend(segs)
        logger.info(f"  [{i + 1}/{len(audio_files)}] {audio_files[i].name}: {len(segs)} segments")

    if not all_segments:
        return []

    all_segments.sort(key=lambda x: x["start"])
    return deduplicate_segments(all_segments, overlap_threshold=0.3)


def _calculate_chunk_offsets(
    audio_files: list[Path],
    overlap_secs: float,
    logger: logging.Logger,
) -> list[float]:
    """Calculate cumulative time offsets for each chunk using ffprobe.

    Chunk N starts at: sum(duration[0..N-1]) - N * overlap_secs
    """
    offsets = []
    cumulative = 0.0
    for i, f in enumerate(audio_files):
        offsets.append(max(0.0, cumulative))
        duration_s = get_audio_duration(f)
        cumulative += duration_s - overlap_secs
        logger.debug(f"  Chunk {i + 1}: offset={offsets[i]:.1f}s, duration={duration_s:.1f}s")
    return offsets
