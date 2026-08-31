"""Groq API transcription client and merging logic."""

import json
import logging
import os
import typing as _typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from groq import Groq

from src import audio
from src.srt import segments_to_srt

MAX_FILE_SIZE_MB = 25
DEFAULT_MODEL = "whisper-large-v3-turbo"


@dataclass
class ChunkResult:
    """Result of transcribing a single audio chunk."""

    chunk_path: Path
    start_time: float
    segments: list[dict]
    success: bool
    error: str | None = None


class TranscriptionError(Exception):
    """Custom exception for transcription errors."""

    pass


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def validate_file_size(path: Path, max_mb: float = MAX_FILE_SIZE_MB) -> None:
    """Validate file doesn't exceed max size."""
    size_mb = get_file_size_mb(path)
    if size_mb > max_mb:
        raise TranscriptionError(
            f"File {path.name} is {size_mb:.1f}MB, exceeds {max_mb}MB limit. "
            "Split into smaller chunks first."
        )


def get_audio_files_from_dir(directory: Path) -> list[Path]:
    """Get all audio files from directory, in numeric filename order."""
    audio_extensions = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".webm"}
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]
    return sorted(files, key=audio.numeric_sort_key)


def find_existing_transcripts(directory: Path, prefix: str) -> set[Path]:
    """Find existing JSON transcript files matching prefix."""
    transcripts = set()
    for f in directory.iterdir():
        if f.suffix == ".json" and f.stem.startswith(prefix):
            transcripts.add(f)
    return transcripts


def transcribe_chunk(
    client: Groq,
    chunk_path: Path,
    model: str,
    language: str | None,
    output_dir: Path,
) -> ChunkResult:
    """Transcribe a single audio chunk and save JSON result."""
    transcript_path = output_dir / f"{chunk_path.stem}.json"

    if transcript_path.exists():
        try:
            with open(transcript_path) as f:
                data = json.load(f)
            return ChunkResult(
                chunk_path=chunk_path,
                start_time=data.get("start_time", 0.0),
                segments=data.get("segments", []),
                success=True,
            )
        except (OSError, json.JSONDecodeError):
            pass

    validate_file_size(chunk_path)

    try:
        with open(chunk_path, "rb") as f:
            response = client.audio.transcriptions.create(
                file=(chunk_path.name, f.read()),
                model=model,
                **({"language": language} if language else {}),
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments: list[dict] = []
        if hasattr(response, "segments") and response.segments:
            segs: list[dict] = list(_typing.cast(_typing.Iterable, response.segments))

            for seg in segs:
                if isinstance(seg, dict):
                    segments.append(
                        {
                            "start": seg.get("start", 0.0),
                            "end": seg.get("end", 0.0),
                            "text": seg.get("text", ""),
                        }
                    )
                else:
                    segments.append(
                        {
                            "start": seg.start,  # type: ignore[unresolved-attribute]  # type: ignore[unresolved-attribute]
                            "end": seg.end,  # type: ignore[unresolved-attribute]  # type: ignore[unresolved-attribute]
                            "text": seg.text,  # type: ignore[unresolved-attribute]  # type: ignore[unresolved-attribute]
                        }
                    )
        result_data = {
            "chunk": str(chunk_path),
            "start_time": 0.0,
            "segments": segments,
        }

        with open(transcript_path, "w") as f:
            json.dump(result_data, f, indent=2)

        return ChunkResult(
            chunk_path=chunk_path,
            start_time=0.0,
            segments=segments,
            success=True,
        )

    except Exception as e:
        return ChunkResult(
            chunk_path=chunk_path,
            start_time=0.0,
            segments=[],
            success=False,
            error=str(e),
        )


def extract_chunk_start_time(filename: str, default: float = 0.0) -> float:
    """Extract start time from chunk filename (e.g., audio_001.ogg -> 0, audio_002 -> 10s)."""
    import re

    match = re.search(r"_(\d+)\.", filename)
    if match:
        idx = int(match.group(1))
        return (idx - 1) * 10.0
    return default


def merge_chunk_results(
    results: list[ChunkResult],
    audio_files: list[Path] | None = None,
    chunk_duration: float = 600.0,
    overlap_secs: float = 5.0,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Merge transcription results from multiple chunks into the source timeline.

    Offsets come from the shared src.audio.chunk_offsets helper (derived from
    the split sidecar) so the Groq cloud path uses the exact same timeline
    math as the local engine. When ``audio_files`` is omitted (e.g. legacy
    callers) it falls back to ``i * (chunk_duration - overlap_secs)``.

    Args:
        results: List of ChunkResult from each chunk, sorted by chunk name.
        audio_files: Chunk paths in the same order as ``results`` (used to read
            the split sidecar for the true per-chunk step).
        chunk_duration: Fallback nominal chunk duration when no sidecar exists.
        overlap_secs: Overlap between chunks in seconds.

    Returns:
        Merged and deduplicated list of segments.
    """
    if not results:
        return []

    if audio_files is not None and len(audio_files) == len(results):
        offsets = audio.chunk_offsets(audio_files, overlap_secs, logger)
    else:
        offsets = [i * (chunk_duration - overlap_secs) for i in range(len(results))]

    all_segments = []
    for i, result in enumerate(results):
        if not result.success or not result.segments:
            continue

        offset = offsets[i]
        for seg in result.segments:
            adjusted_seg = {
                "start": seg["start"] + offset,
                "end": seg["end"] + offset,
                "text": seg["text"],
            }
            all_segments.append(adjusted_seg)

    if not all_segments:
        return []

    all_segments.sort(key=lambda x: x["start"])

    merged = [all_segments[0]]
    for seg in all_segments[1:]:
        prev = merged[-1]

        if seg["start"] < prev["end"]:
            text_overlap = _compute_text_overlap(prev["text"], seg["text"])
            if text_overlap > 0.3:
                continue

            if seg["start"] < prev["end"]:
                prev["end"] = seg["end"]
                prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            merged.append(seg)

    return merged


def _compute_text_overlap(text1: str, text2: str) -> float:
    """Compute similarity between two text strings using word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def transcribe_audio(
    input_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    parallel: int = 4,
    overlap_secs: float = 5.0,
    chunk_duration: float = 600.0,
    logger: logging.Logger | None = None,
    fail_fast: bool = False,
) -> Path:
    """Transcribe audio file or directory of chunks.

    Args:
        input_path: Single audio file or directory of audio chunks
        output_path: Output .srt file path
        model: Whisper model to use
        language: Language code (e.g., 'en')
        parallel: Number of parallel API calls
        overlap_secs: Overlap seconds used when splitting
        chunk_duration: Expected chunk duration in seconds
        logger: Optional logger instance
        fail_fast: Stop on first error

    Returns:
        Path to output SRT file
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    if input_path.is_file():
        return _transcribe_single_file(client, input_path, output_path, model, language, logger)
    elif input_path.is_dir():
        return _transcribe_directory(
            client,
            input_path,
            output_path,
            model,
            language,
            parallel,
            overlap_secs,
            chunk_duration,
            logger,
            fail_fast,
        )
    else:
        raise TranscriptionError(f"Input not found: {input_path}")


def _transcribe_single_file(
    client: Groq,
    input_path: Path,
    output_path: Path,
    model: str,
    language: str | None,
    logger: logging.Logger,
) -> Path:
    """Transcribe a single audio file."""
    validate_file_size(input_path)

    logger.info(f"Transcribing: {input_path.name}")

    with open(input_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(input_path.name, f.read()),
            model=model,
            **({"language": language} if language else {}),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments: list[dict] = []
    if hasattr(response, "segments") and response.segments:
        segments = [
            {
                "start": _typing.cast(dict, seg).get("start", 0.0)
                if isinstance(seg, dict)
                else seg.start,  # noqa: F821
                "end": _typing.cast(dict, seg).get("end", 0.0)
                if isinstance(seg, dict)
                else seg.end,  # noqa: F821
                "text": _typing.cast(dict, seg).get("text", "")
                if isinstance(seg, dict)
                else seg.text,  # noqa: F821
            }
            for seg in _typing.cast(_typing.Iterable, response.segments)
        ]

    srt_content = segments_to_srt(segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(srt_content)

    logger.info(f"Saved: {output_path}")
    return output_path


def _transcribe_directory(
    client: Groq,
    input_dir: Path,
    output_path: Path,
    model: str,
    language: str | None,
    parallel: int,
    overlap_secs: float,
    chunk_duration: float,
    logger: logging.Logger,
    fail_fast: bool = False,
) -> Path:
    """Transcribe a directory of audio chunks."""
    audio_files = get_audio_files_from_dir(input_dir)

    if not audio_files:
        raise TranscriptionError(f"No audio files found in: {input_dir}")

    logger.info(f"Found {len(audio_files)} audio files")

    existing = find_existing_transcripts(input_dir, audio_files[0].stem.split("_")[0])
    if existing:
        logger.info(f"Resuming: found {len(existing)} existing transcripts")

    results: list[ChunkResult] = []
    failed = False

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(transcribe_chunk, client, f, model, language, input_dir): f
            for f in audio_files
        }

        for future in as_completed(futures):
            chunk_path = futures[future]
            try:
                result = future.result()
                results.append(result)

                if result.success:
                    logger.debug(f"Transcribed: {chunk_path.name}")
                else:
                    logger.warning(f"Failed: {chunk_path.name} - {result.error}")
            except Exception as e:
                logger.error(f"Error processing {chunk_path.name}: {e}")
                if fail_fast:
                    failed = True
                    break

    if fail_fast and failed:
        raise TranscriptionError("Transcription failed for one or more chunks")

    results.sort(key=lambda x: x.chunk_path.name)

    merged = merge_chunk_results(results, audio_files, chunk_duration, overlap_secs, logger)

    srt_content = segments_to_srt(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(srt_content)

    logger.info(f"Saved: {output_path}")
    return output_path
