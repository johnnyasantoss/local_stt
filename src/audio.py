"""Shared audio utilities."""

import logging
import subprocess
import tempfile
from pathlib import Path

from pydub import AudioSegment

SUPPORTED_EXTENSIONS: set[str] = {
    "flac",
    "mp3",
    "mp4",
    "mpeg",
    "mpga",
    "m4a",
    "ogg",
    "wav",
    "webm",
}


SAMPLE_RATE = 16000


def numeric_sort_key(path: Path) -> list[tuple[str | int]]:
    """Sort key that orders filenames with embedded numbers numerically.

    Plain lexicographic sorting mis-orders chunk files once indices exceed the
    zero-padding width (e.g. processed_1000.ogg sorts before processed_999.ogg
    because "1" < "9"). That scrambles chunk order and, since offsets are
    computed as index*step, assigns every later chunk the wrong timestamp.
    This key splits the name into text/digit runs so 1000 > 999.
    """
    import re

    return [
        (int(tok) if tok.isdigit() else tok.lower(),)
        for tok in re.split(r"(\d+)", path.name)
        if tok != ""
    ]


def sorted_audio_chunks(directory: Path, pattern: str = "*.ogg") -> list[Path]:
    """Return audio chunk files in numeric order (see numeric_sort_key)."""
    return sorted(
        (p for p in directory.glob(pattern) if p.is_file()),
        key=numeric_sort_key,
    )


def chunk_step_seconds(chunk_duration_s: float, overlap_s: float) -> float:
    """Nominal source advance between consecutive chunk starts.

    A splitter slices the source into ``chunk_duration_s`` windows that
    overlap by ``overlap_s`` seconds, so each chunk (except the first) starts
    ``chunk_duration_s - overlap_s`` later than the previous one. This is the
    timeline step used to place a chunk's transcript segments in the continuous
    source, and it must match exactly what audio-split used when slicing.
    """
    return max(0.0, float(chunk_duration_s) - float(overlap_s))


def chunk_offsets(
    audio_files: list[Path],
    overlap_secs: float,
    logger: logging.Logger | None = None,
) -> list[float]:
    """Cumulative start offsets (seconds) for each chunk in the source timeline.

    Each Opus chunk is encoded with a fixed pre-skip (312 samples @ 48kHz ~=
    6.5ms) that ffprobe reports inside the raw stream duration. Summing those
    per-chunk decoded durations therefore over-counts every step and drifts
    the transcript timeline forward of the continuous diarize WAV (which
    decodes the single source stream once, so it steps by the true slice
    length). Over a long file this reaches several seconds of desync between
    the displayed timestamp and the audio it seeks to.

    To stay on the source timeline we step by the NOMINAL slice length the
    splitter used: chunk_duration = min(size_mb-derived, max_seconds),
    step = chunk_duration - overlap, offset[k] = k * step. These come from the
    split sidecar (chunks/.split-meta.json) so they always match how
    audio-split actually sliced the source. Falls back to summing real
    per-chunk durations when no sidecar is present (no worse than before).
    """
    sidecar = audio_files[0].parent / ".split-meta.json" if audio_files else None
    if sidecar is not None and sidecar.is_file():
        try:
            import json

            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            size_mb = float(meta["size_mb"])
            bitrate = int(meta["bitrate"])
            max_seconds = meta.get("max_seconds")
            chunk_s = int(size_mb * 1024 * 1024 * 8 / bitrate)
            if max_seconds is not None:
                chunk_s = int(min(chunk_s, float(max_seconds)))
            step = chunk_step_seconds(chunk_s, overlap_secs)
            return [max(0.0, k * step) for k in range(len(audio_files))]
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            if logger is not None:
                logger.warning(f"  split sidecar unreadable ({e}); summing chunk durations")
    offsets = []
    cumulative = 0.0
    for f in audio_files:
        offsets.append(max(0.0, cumulative))
        duration_s = get_audio_duration(f)
        cumulative += duration_s - overlap_secs
    return offsets


def detect_format(file_path: Path) -> str:
    """Detect audio format from file extension."""
    ext = file_path.suffix.lstrip(".").lower()
    if ext == "m4a":
        return "m4a"
    elif ext == "mp4":
        return "mp4"
    elif ext == "mpga":
        return "mpga"
    elif ext == "webm":
        return "webm"
    return ext


def load_audio(file_path: Path) -> AudioSegment:
    """Load audio file using pydub."""
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    fmt = detect_format(file_path)
    audio = AudioSegment.from_file(str(file_path), format=fmt)

    return audio


def get_audio_info(audio: AudioSegment) -> dict[str, object]:
    """Get audio metadata as dict."""
    return {
        "duration_seconds": len(audio) / 1000,
        "channels": audio.channels,
        "sample_rate": audio.frame_rate,
        "sample_width": audio.sample_width,
    }


def load_audio_as_wav16k(file_path: Path) -> tuple[tempfile.TemporaryDirectory, Path]:
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


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds using ffprobe (no full load into memory)."""
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
