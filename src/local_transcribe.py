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
import struct
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

    Precedence: explicit query (path or short name) > TCPP_MODEL env.
    No model defaults to whisper; the caller must supply a model name or set
    TCPP_MODEL.
    """
    if query:
        cand = Path(query).expanduser()
        if cand.is_file():
            return cand
        # short name (e.g. "cohere") -> fuzzy search HF cache folders + gguf names
        short = query.removesuffix(".gguf").lower()
        hub = Path("~/.cache/huggingface/hub").expanduser()
        matches = []
        for gguf in sorted(hub.glob("models--handy-computer--*-gguf/snapshots/*/*.gguf")):
            folder_short = gguf.parent.parent.parent.name
            folder_short = folder_short.removeprefix("models--handy-computer--").removesuffix(
                "-gguf"
            )
            if short in folder_short.lower() or short in gguf.stem.lower():
                matches.append(gguf)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            listing = "\n  ".join(str(m) for m in matches)
            raise ValueError(
                f"Model query '{query}' is ambiguous; multiple matches found:\n  {listing}\n"
                "Pass a more specific query or the full path to the .gguf file."
            )
        raise FileNotFoundError(
            f"Model '{query}' not found in HF cache (no folder/gguf name match)."
        )
    env_model = os.environ.get("TCPP_MODEL")
    if env_model:
        cand = Path(env_model).expanduser()
        if not cand.is_file():
            raise FileNotFoundError(f"TCPP_MODEL={env_model} does not exist.")
        return cand

    raise ValueError(
        "No model specified. Pass a model name (e.g. 'cohere-transcribe-03-2026') "
        "or set TCPP_MODEL to the .gguf path."
    )


class _GGUFReader:
    """Sequential cursor over a binary GGUF stream (read-past, no buffering)."""

    def __init__(self, f):
        self.f = f

    def _read(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) < n:
            raise EOFError("unexpected end of GGUF file")
        return data

    def u8(self) -> int:
        return self._read(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self._read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self._read(4))[0]

    def string(self) -> str:
        n = self.u64()
        return self._read(n).decode("utf-8", "replace")


# GGUF metadata value type codes.
_GGUF_U8 = 0
_GGUF_I8 = 1
_GGUF_U16 = 2
_GGUF_I16 = 3
_GGUF_U32 = 4
_GGUF_I32 = 5
_GGUF_F32 = 6
_GGUF_BOOL = 7
_GGUF_STRING = 8
_GGUF_ARRAY = 9


def _read_gguf_value(reader: _GGUFReader, value_type: int) -> object:
    """Read one GGUF metadata value of the given type, advancing the cursor."""
    if value_type == _GGUF_U8:
        return reader.u8()
    if value_type == _GGUF_I8:
        return struct.unpack("<b", reader._read(1))[0]
    if value_type == _GGUF_U16:
        return reader.u16()
    if value_type == _GGUF_I16:
        return struct.unpack("<h", reader._read(2))[0]
    if value_type == _GGUF_U32:
        return reader.u32()
    if value_type == _GGUF_I32:
        return struct.unpack("<i", reader._read(4))[0]
    if value_type == _GGUF_F32:
        return reader.f32()
    if value_type == _GGUF_BOOL:
        return reader.u8() != 0
    if value_type == _GGUF_STRING:
        return reader.string()
    if value_type == _GGUF_ARRAY:
        sub_type = reader.u32()
        count = reader.u64()
        return [_read_gguf_value(reader, sub_type) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {value_type}")


def read_gguf_metadata(path: Path, wanted: set[str]) -> dict[str, object]:
    """Read ONLY the requested GGUF metadata keys, discarding all others.

    Reads past (but does not store) unwanted values, so large arrays such as
    tokenizer.ggml.tokens are skipped without being materialized. Raises on a
    malformed header or any parse error.
    """
    result: dict[str, object] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        reader = _GGUFReader(f)
        version = reader.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        reader.u64()  # tensor_count
        n_kv = reader.u64()
        for _ in range(n_kv):
            key = reader.string()
            value_type = reader.u32()
            if key in wanted and key not in result:
                result[key] = _read_gguf_value(reader, value_type)
            else:
                # Read and discard unwanted values (large arrays are skipped here).
                _read_gguf_value(reader, value_type)
    return result


# Maximum audio clip window (seconds) per GGUF architecture. None means no
# per-file cap (the engine chunks long audio internally).
#   cohere_asr: 30s is safely under its 50s positional limit
#     (pos_emb_max_len 5000 * hop 160 / sample_rate 16000) and the model card's
#     ~30-35s clip window.
#   whisper: None (transcribe.cpp chunks 30s windows internally).
GGUF_ARCH_MAX_AUDIO_SEC: dict[str, float | None] = {
    "cohere_asr": 15.0,
    "whisper": None,
}


def model_audio_window_seconds(path: Path) -> float | None:
    """Return the per-file audio cap (seconds) for a GGUF model, else None.

    Detects the architecture from GGUF metadata and maps it to the max clip
    window. Never raises: any parse error yields None (no cap applied).
    """
    try:
        meta = read_gguf_metadata(path, {"general.architecture"})
        arch = meta.get("general.architecture")
        if not isinstance(arch, str):
            return None
        return GGUF_ARCH_MAX_AUDIO_SEC.get(arch, None)
    except Exception:
        return None


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
    assert proc.stdout is not None, "Popen stdout must be iterable (PIPE mode)"
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
        model_query: GGUF path or short model name. Required: no default model.
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
    else:
        raise ValueError(f"Input path is neither a file nor directory: {input_path}")


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
    all_synthetic = True
    for i, (obj, offset_s) in enumerate(zip(results, offsets, strict=True)):
        if "error" in obj:
            logger.error(f"  [{i + 1}/{len(audio_files)}] {audio_files[i].name}: {obj['error']}")
            continue
        segs = _segs_from_jsonl(obj, offset_s)
        # Models like cohere/granite may return text but no per-word segments.
        # Create a single synthetic segment spanning the full chunk duration.
        if not segs and obj.get("text", "").strip():
            chunk_dur = get_audio_duration(audio_files[i])
            segs = [{"start": offset_s, "end": offset_s + chunk_dur, "text": obj["text"].strip()}]
        else:
            all_synthetic = False
        all_segments.extend(segs)
        logger.info(f"  [{i + 1}/{len(audio_files)}] {audio_files[i].name}: {len(segs)} segments")

    if not all_segments:
        return []

    all_segments.sort(key=lambda x: x["start"])
    # Synthetic segments (text-only models) represent full chunks — they must not
    # be merged across overlapping chunk boundaries.
    if all_synthetic:
        return all_segments
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
