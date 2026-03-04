"""SRT file generation utilities."""

import math
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


def segments_to_srt(segments: list[dict]) -> str:
    """Convert Groq API segments to SRT format.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys

    Returns:
        SRT formatted string
    """
    entries = []
    for idx, seg in enumerate(segments, start=1):
        text = seg.get("text", "").strip()
        if not text:
            continue
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
