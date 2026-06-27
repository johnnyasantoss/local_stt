"""Local ONNX transcription engine using Handy.app models."""

import json
import logging
import tempfile
from pathlib import Path

import onnx_asr
from pydub import AudioSegment

from src.audio import detect_format
from src.srt import deduplicate_segments

HANDY_MODELS_DIR = Path.home() / "Library/Application Support/com.pais.handy/models"
SUPPORTED_MODEL_TYPES = {"nemo-conformer-tdt", "nemo-conformer-aed"}
AUDIO_EXTENSIONS = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".webm", ".mp4"}
SAMPLE_RATE = 16000


def discover_models() -> list[dict]:
    """Scan Handy models dir for supported ONNX models.

    Returns list of {name, path, model_type, size_mb} dicts.
    """
    models = []
    if not HANDY_MODELS_DIR.is_dir():
        return models

    for subdir in sorted(HANDY_MODELS_DIR.iterdir()):
        if not subdir.is_dir():
            continue

        config_path = subdir / "config.json"
        if not config_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        model_type = config.get("model_type", "")
        if model_type not in SUPPORTED_MODEL_TYPES:
            continue

        encoder = _find_file(subdir, "encoder-model*.onnx")
        if encoder is None:
            continue

        size_mb = sum(f.stat().st_size for f in subdir.iterdir() if f.suffix == ".onnx") / (
            1024 * 1024
        )

        models.append(
            {
                "name": subdir.name,
                "path": subdir,
                "model_type": model_type,
                "size_mb": size_mb,
            }
        )

    return models


def resolve_model(query: str) -> dict:
    """Fuzzy-match model query against discovered models.

    1. Exact name match
    2. Case-insensitive substring match
    3. Direct path if valid
    """
    models = discover_models()

    for m in models:
        if m["name"] == query:
            return m

    query_lower = query.lower()
    for m in models:
        if query_lower in m["name"].lower():
            return m

    path = Path(query)
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_type = config.get("model_type", "")
            if model_type in SUPPORTED_MODEL_TYPES:
                size_mb = sum(f.stat().st_size for f in path.iterdir() if f.suffix == ".onnx") / (
                    1024 * 1024
                )
                return {
                    "name": path.name,
                    "path": path,
                    "model_type": model_type,
                    "size_mb": size_mb,
                }

    names = [m["name"] for m in models]
    raise ValueError(f"Model '{query}' not found. Available: {', '.join(names)}")


def _find_file(directory: Path, pattern: str) -> Path | None:
    """Find a file matching glob pattern in directory."""
    matches = list(directory.glob(pattern))
    onnx_files = [m for m in matches if m.is_file() and ".onnx.data" not in m.suffix]
    return onnx_files[0] if onnx_files else None


def _get_quantization(model_dir: Path) -> str | None:
    """Detect quantization level from model files."""
    for f in model_dir.iterdir():
        if "int8" in f.name and f.suffix == ".onnx":
            return "int8"
        if "int4" in f.name and f.suffix == ".onnx":
            return "int4"
    return None


def _is_quantized(model_dir: Path) -> bool:
    """Check if model has quantized files (int8/int4 in filename)."""
    return _get_quantization(model_dir) is not None


def load_model(model_info: dict, providers: list | None = None, logger=None):
    """Load ONNX ASR model via onnx-asr.

    Uses CoreML MLProgram format with GPU compute units for the encoder/decoder,
    and numpy preprocessors to avoid CoreML incompatibility with the preprocessor model.

    Args:
        model_info: From resolve_model() or discover_models()
        providers: ONNX Runtime providers. Auto-detects CoreML MLProgram on macOS.
        logger: Logger for reporting active providers
    """
    if providers is None:
        providers = _get_providers()

    quantization = (
        _get_quantization(model_info["path"]) if _is_quantized(model_info["path"]) else None
    )

    if logger:
        logger.info(f"ONNX providers: {providers}")
        logger.info(f"Quantization: {quantization or 'none'}")

    model = onnx_asr.load_model(
        model_info["model_type"],
        str(model_info["path"]),
        quantization=quantization,
        providers=providers,
        preprocessor_config={"use_numpy_preprocessors": True, "max_concurrent_workers": 1},
    )

    if logger:
        import onnxruntime as ort

        try:
            used = ort.get_available_providers()
            logger.info(f"Available providers: {used}")
        except Exception:
            pass

    return model


def load_vad():
    """Load Silero VAD via onnx-asr."""
    return onnx_asr.load_vad("silero")


def _get_providers() -> list:
    """Auto-detect best ONNX Runtime providers.

    On macOS, CPU-only is faster than CoreML for NeMo Conformer models
    because CoreML only supports ~47% of model nodes, causing expensive
    CPU↔GPU partition switching. CPU uses Apple's BNNS/AMX acceleration
    via the Accelerate framework which is highly optimized for these ops.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_audio_as_wav16k(file_path: Path) -> tuple[tempfile.TemporaryDirectory, Path]:
    """Convert any audio format to 16kHz mono WAV.

    Returns (temp_dir, wav_path). Caller must keep temp_dir alive while using wav_path.
    temp_dir auto-cleans on garbage collection or explicit cleanup.
    """
    audio = AudioSegment.from_file(str(file_path), format=detect_format(file_path))
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)

    tmp_dir = tempfile.TemporaryDirectory()
    wav_path = Path(tmp_dir.name) / "audio.wav"
    audio.export(str(wav_path), format="wav")
    return tmp_dir, wav_path


def transcribe_local(
    input_path: Path,
    model_query: str,
    language: str | None = None,
    vad_threshold: float = 0.5,
    overlap_secs: float = 5.0,
    parallel: int = 1,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Transcribe audio using local ONNX model with VAD always on.

    Args:
        input_path: Single audio file or directory of chunks
        model_query: Fuzzy model name
        language: Language code (required for Canary AED, optional for Parakeet TDT)
        vad_threshold: VAD speech detection threshold
        overlap_secs: Overlap between chunks for merging
        parallel: Number of parallel workers for chunk directory
        logger: Logger instance

    Returns:
        Segments: [{"start": float, "end": float, "text": str}]
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    model_info = resolve_model(model_query)
    logger.info(
        f"Using model: {model_info['name']} ({model_info['model_type']}, {model_info['size_mb']:.0f} MB)"
    )

    asr_model = load_model(model_info, logger=logger)
    vad = load_vad()

    if input_path.is_file():
        return _transcribe_file(asr_model, vad, input_path, language, vad_threshold, logger)
    elif input_path.is_dir():
        return _transcribe_directory(
            asr_model, vad, input_path, language, vad_threshold, overlap_secs, parallel, logger
        )
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


CHUNK_DURATION_MS = 10 * 60 * 1000  # 10 minutes per chunk
CHUNK_OVERLAP_MS = 5 * 1000  # 5 seconds overlap


def _transcribe_file(
    asr_model,
    vad,
    file_path: Path,
    language: str | None,
    vad_threshold: float,
    logger: logging.Logger,
) -> list[dict]:
    """Transcribe a single audio file with VAD.

    For files > 10 minutes, auto-splits into temporary WAV chunks,
    transcribes each sequentially with correct time offsets, then merges.
    """
    logger.info(f"Loading {file_path.name}...")
    audio = AudioSegment.from_file(str(file_path), format=detect_format(file_path))
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
    duration_ms = len(audio)

    if duration_ms <= CHUNK_DURATION_MS:
        return _transcribe_wav_segment(
            audio, file_path.name, 0.0, asr_model, vad, vad_threshold, language, logger
        )

    logger.info(
        f"File is {duration_ms / 1000:.0f}s, splitting into "
        f"~{duration_ms // CHUNK_DURATION_MS + 1} chunks of {CHUNK_DURATION_MS // 1000}s"
    )

    all_segments = []
    chunk_num = 0
    pos_ms = 0

    while pos_ms < duration_ms:
        end_ms = min(pos_ms + CHUNK_DURATION_MS, duration_ms)
        chunk = audio[pos_ms:end_ms]
        offset_s = pos_ms / 1000.0

        chunk_num += 1
        logger.info(f"  Transcribing chunk {chunk_num} at {offset_s:.0f}s...")

        segs = _transcribe_wav_segment(
            chunk,
            f"{file_path.name}#chunk{chunk_num}",
            offset_s,
            asr_model,
            vad,
            vad_threshold,
            language,
            logger,
        )
        all_segments.extend(segs)

        pos_ms += CHUNK_DURATION_MS - CHUNK_OVERLAP_MS

    if not all_segments:
        return []

    all_segments.sort(key=lambda x: x["start"])
    return deduplicate_segments(all_segments, overlap_threshold=0.3)


def _transcribe_wav_segment(
    audio: AudioSegment,
    label: str,
    time_offset_s: float,
    asr_model,
    vad,
    vad_threshold: float,
    language: str | None,
    logger: logging.Logger,
) -> list[dict]:
    """Transcribe an in-memory AudioSegment via temp WAV file."""
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name) / "segment.wav"

    try:
        audio.export(str(tmp_path), format="wav")

        model_with_vad = asr_model.with_vad(
            vad,
            threshold=vad_threshold,
            max_speech_duration_s=30.0,
            min_silence_duration_ms=300.0,
        )
        kwargs = {}
        if language:
            kwargs["language"] = language

        segments = []
        for result in model_with_vad.recognize(str(tmp_path), **kwargs):
            segments.append(
                {
                    "start": result.start + time_offset_s,
                    "end": result.end + time_offset_s,
                    "text": result.text,
                }
            )

        logger.info(f"    {label}: {len(segments)} segments")
        return segments

    finally:
        tmp_dir.cleanup()


def _transcribe_directory(
    asr_model,
    vad,
    input_dir: Path,
    language: str | None,
    vad_threshold: float,
    overlap_secs: float,
    parallel: int,
    logger: logging.Logger,
) -> list[dict]:
    """Transcribe a directory of audio chunks in parallel with time offsets."""
    audio_files = sorted(
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in: {input_dir}")

    logger.info(f"Found {len(audio_files)} audio files")

    model_with_vad = asr_model.with_vad(
        vad,
        threshold=vad_threshold,
        max_speech_duration_s=30.0,
        min_silence_duration_ms=300.0,
    )

    # Calculate cumulative time offsets from chunk durations
    offsets = _calculate_chunk_offsets(audio_files, overlap_secs, logger)
    logger.info(f"Chunk offsets: {offsets[:5]}... (first 5)")

    def _transcribe_one(file_path: Path, offset_s: float) -> tuple[str, list[dict]]:
        tmp_dir, wav_path = _load_audio_as_wav16k(file_path)
        try:
            kwargs = {}
            if language:
                kwargs["language"] = language

            segments = []
            for result in model_with_vad.recognize(str(wav_path), **kwargs):
                segments.append(
                    {
                        "start": result.start + offset_s,
                        "end": result.end + offset_s,
                        "text": result.text,
                    }
                )
            return file_path.name, segments
        finally:
            tmp_dir.cleanup()

    all_segments = []
    total = len(audio_files)
    for i, (file_path, offset_s) in enumerate(zip(audio_files, offsets, strict=True)):
        logger.info(
            f"  [{i + 1}/{total}] Transcribing {file_path.name} (offset {offset_s:.0f}s)..."
        )
        try:
            name, segments = _transcribe_one(file_path, offset_s)
            all_segments.append((name, segments))
            logger.info(f"  [{i + 1}/{total}] Done: {len(segments)} segments")
        except Exception as e:
            logger.error(f"  [{i + 1}/{total}] Failed: {file_path.name}: {e}")

    if not all_segments:
        return []

    all_segments.sort(key=lambda x: x[0])
    return _merge_chunks(all_segments, overlap_secs)


def _get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds using ffprobe (no full load into memory)."""
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


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
        duration_s = _get_audio_duration(f)
        cumulative += duration_s - overlap_secs
        logger.debug(f"  Chunk {i + 1}: offset={offsets[i]:.1f}s, duration={duration_s:.1f}s")

    return offsets


def _merge_chunks(
    chunk_results: list[tuple[str, list[dict]]],
    overlap_secs: float,
) -> list[dict]:
    """Merge transcription results from multiple chunks.

    With proper time offsets applied, segments from different chunks only
    overlap in the overlap region. Deduplicate by dropping segments that
    fall entirely within the overlap of the previous chunk.
    """
    all_segments = []

    for _chunk_name, segments in chunk_results:
        for seg in segments:
            all_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
            )

    if not all_segments:
        return []

    all_segments.sort(key=lambda x: x["start"])

    # Drop segments that fall entirely within the previous segment's time range
    # (these are duplicates from chunk overlap)
    merged = []
    for seg in all_segments:
        if merged and seg["end"] <= merged[-1]["end"]:
            # Segment is entirely within previous - skip duplicate
            continue
        if merged and seg["start"] < merged[-1]["end"]:
            # Partial overlap - trim the start
            overlap = merged[-1]["end"] - seg["start"]
            if overlap < overlap_secs:
                # Small overlap from chunk boundary - keep both, just trim
                merged.append(seg)
            else:
                # Large overlap - skip
                continue
        else:
            merged.append(seg)

    return merged
