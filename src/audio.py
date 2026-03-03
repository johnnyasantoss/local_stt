"""Shared audio utilities."""

from pathlib import Path
from typing import Optional

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
