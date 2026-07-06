"""Shared audio utilities."""

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
