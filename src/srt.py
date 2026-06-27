"""SRT file generation utilities."""

import math
import re
from dataclasses import dataclass


@dataclass
class SRTEntry:
    """Represents a single SRT entry."""

    index: int
    start_time: float
    end_time: float
    text: str

    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format: HH:MM:SS,mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_srt_line(self) -> str:
        """Convert to SRT format string."""
        start = self.format_timestamp(self.start_time)
        end = self.format_timestamp(self.end_time)
        return f"{self.index}\n{start} --> {end}\n{self.text}\n"


def format_seconds_to_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp format: HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[dict], include_speakers: bool = False) -> str:
    """Convert segments to SRT format.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys
        include_speakers: If True and segment has 'speaker' field,
                          prefix text with 'Speaker {label}: '

    Returns:
        SRT formatted string
    """
    entries = []
    for idx, seg in enumerate(segments, start=1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if include_speakers and "speaker" in seg:
            text = f"Speaker {seg['speaker']}: {text}"
        entry = SRTEntry(
            index=idx,
            start_time=seg.get("start", 0.0),
            end_time=seg.get("end", 0.0),
            text=text,
        )
        entries.append(entry)

    return "\n".join(entry.to_srt_line() for entry in entries)


def deduplicate_segments(segments: list[dict], overlap_threshold: float = 0.5) -> list[dict]:
    """Deduplicate segments that appear in overlapping regions.

    Args:
        segments: List of segment dicts sorted by start time
        overlap_threshold: Minimum text similarity to consider as duplicate

    Returns:
        Deduplicated list of segments
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]

        if seg["start"] >= prev["end"]:
            merged.append(seg.copy())
            continue

        overlap_ratio = (prev["end"] - seg["start"]) / (seg["end"] - seg["start"])

        if overlap_ratio > overlap_threshold:
            continue

        if seg["start"] < prev["end"]:
            prev["end"] = seg["end"]
            if seg.get("text"):
                prev["text"] = prev.get("text", "") + " " + seg["text"]

    return merged


_SRT_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_SPEAKER_RE = re.compile(r"^Speaker\s+(.+?):\s*(.*)")


def _parse_srt_time(time_str: str) -> float:
    """Parse SRT timestamp HH:MM:SS,mmm to seconds."""
    match = _SRT_TIME_RE.match(time_str.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {time_str}")
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000.0


def parse_srt(srt_content: str) -> list[dict]:
    """Parse SRT content into segment dicts.

    Args:
        srt_content: SRT formatted string

    Returns:
        List of {"start": float, "end": float, "text": str, "speaker": str|None}
    """
    blocks = re.split(r"\n\s*\n", srt_content.strip())
    segments = []

    for block in blocks:
        lines = [l for l in block.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            continue

        time_line = lines[1]
        time_match = _SRT_TIME_RE.findall(time_line)
        if len(time_match) != 2:
            continue

        start = _parse_srt_time(time_line)
        end = _parse_srt_time(time_line.split("-->")[1])

        text_lines = lines[2:]
        text = " ".join(text_lines).strip()

        speaker = None
        sp_match = _SPEAKER_RE.match(text)
        if sp_match:
            speaker = sp_match.group(1).strip()
            text = sp_match.group(2).strip()

        segments.append({"start": start, "end": end, "text": text, "speaker": speaker})

    return segments
