"""Speaker diarization using pyannote-audio."""

import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import torch


def _ensure_wav(
    audio_path: Path, cache_dir: Path | None = None
) -> tuple[tempfile.TemporaryDirectory | None, Path]:
    """Convert audio to 16kHz mono WAV if not already WAV.

    Compressed formats (Opus, MP3) may have sample counts that don't exactly
    match expected duration, causing pyannote chunking errors. WAV guarantees
    exact sample alignment.

    If cache_dir is provided, the WAV is cached there as <stem>.diarize.wav
    and reused on subsequent runs. Otherwise a temp dir is used.

    Returns (temp_dir_or_none, wav_path). Caller must keep temp_dir alive
    if it is not None.
    """
    if audio_path.suffix.lower() == ".wav":
        return None, audio_path

    # Use persistent cache if provided, otherwise temp dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        wav_path = cache_dir / f"{audio_path.stem}.diarize.wav"
        if wav_path.exists():
            return None, wav_path
        tmp_dir = None
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        wav_path = Path(tmp_dir.name) / "diarize_input.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(wav_path),
        ],
        check=True,
    )
    return tmp_dir, wav_path


def diarize_audio(
    audio_path: Path,
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int = 10,
    cache_dir: Path | None = None,
    logger=None,
) -> list[dict]:
    """Run speaker diarization on audio file.

    Requires HF_TOKEN env var with accepted conditions for:
      - pyannote/speaker-diarization-3.1
      - pyannote/segmentation-3.0

    Args:
        audio_path: Path to audio file
        num_speakers: Known number of speakers (optional)
        min_speakers: Minimum speaker count
        max_speakers: Maximum speaker count
        logger: Logger instance

    Returns:
        [{"start": float, "end": float, "speaker": "SPEAKER_01"}, ...]
    """
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN env var required for diarization. "
            "Set it with a HuggingFace token that has accepted "
            "pyannote/speaker-diarization-3.1 conditions."
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if logger:
        logger.info(f"Loading diarization pipeline on device: {device}")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    assert pipeline is not None, "Failed to load pyannote pipeline"
    pipeline.to(torch.device(device))

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        kwargs["min_speakers"] = min_speakers
        kwargs["max_speakers"] = max_speakers

    if logger:
        logger.info(f"Diarizing {audio_path}...")

    tmp_dir, wav_path = _ensure_wav(audio_path, cache_dir=cache_dir)
    try:
        if tmp_dir is not None and logger:
            logger.info(f"Converted to temp WAV: {wav_path}")
        elif cache_dir is not None and logger and wav_path.exists():
            logger.info(f"Using cached WAV: {wav_path}")

        result = pipeline(str(wav_path), **kwargs)

        # pyannote 4.x returns DiarizeOutput, older versions return Annotation directly
        if hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        else:
            annotation = result
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            }
        )

    if logger:
        speakers = set(s["speaker"] for s in segments)
        logger.info(f"Detected {len(speakers)} speakers across {len(segments)} segments")

    return segments


def assign_speakers_to_segments(
    transcription_segments: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:
    """Assign speaker labels to transcription segments by maximum time overlap."""
    result = []
    for tseg in transcription_segments:
        best_speaker = None
        best_overlap = 0.0

        for dseg in diarization_segments:
            overlap_start = max(tseg["start"], dseg["start"])
            overlap_end = min(tseg["end"], dseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

        new_seg = dict(tseg)
        new_seg["speaker"] = best_speaker or "UNKNOWN"
        result.append(new_seg)

    return result


def get_speaker_samples(
    segments: list[dict],
    max_samples: int = 3,
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Group segments by speaker, return the best samples for labeling.

    Prefers long, substantive segments (>=2s duration, >=10 chars text)
    so the user can clearly hear and identify each speaker's voice.
    Falls back to shorter segments if not enough long ones exist.

    Returns (samples_by_speaker, segment_counts).
    """
    from collections import defaultdict

    all_segs: dict[str, list[dict]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    for seg in segments:
        speaker = seg["speaker"]
        counts[speaker] += 1
        all_segs[speaker].append(
            {
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
            }
        )

    picked: dict[str, list[dict]] = {}
    for speaker, segs in all_segs.items():
        # Prefer long segments with substantive text
        substantive = [
            s for s in segs if (s["end"] - s["start"]) >= 2.0 and len(s["text"].strip()) >= 10
        ]
        # Sort by duration descending — longest segments are clearest for ID
        substantive.sort(key=lambda s: s["end"] - s["start"], reverse=True)

        if len(substantive) >= max_samples:
            picked[speaker] = substantive[:max_samples]
        else:
            # Use all substantive, fill remaining with shorter ones
            remaining = [s for s in segs if s not in substantive]
            remaining.sort(key=lambda s: s["end"] - s["start"], reverse=True)
            picked[speaker] = (substantive + remaining)[:max_samples]

    return picked, dict(counts)


def _format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS for display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def prompt_speaker_labels(
    samples: dict[str, list[dict]],
    counts: dict[str, int],
    existing_labels: dict[str, str] | None = None,
) -> dict[str, str]:
    """Interactive CLI prompt for labeling speakers.

    Shows sample text with timestamps so the user can navigate to that
    point in the audio to hear the speaker's voice.

    If existing_labels is provided, shows the saved label as default
    (press Enter to keep it). Re-asks all speakers each run so the user
    can correct previous labels.

    On Ctrl+C, returns whatever labels were entered so far (partial).
    """
    labels = {}
    prev = existing_labels or {}

    all_speakers = sorted(samples.keys())
    if not all_speakers:
        return labels

    print(f"\n  {len(all_speakers)} speaker(s) to label (Ctrl+C to save and exit):\n")

    try:
        for speaker_id in all_speakers:
            segs = samples[speaker_id]
            count = counts[speaker_id]
            default = prev.get(speaker_id, speaker_id)

            print(f"  {speaker_id} ({count} segments)")
            for seg in segs:
                ts = _format_timestamp(seg["start"])
                preview = seg["text"][:80] + ("..." if len(seg["text"]) > 80 else "")
                print(f'    [{ts}] "{preview}"')

            label = input(f"  Label [{default}]: ").strip()
            labels[speaker_id] = label if label else default
            print()

    except KeyboardInterrupt:
        print("\n  Interrupted. Saving partial labels...")
        # Merge: keep any not-yet-asked speakers from previous labels
        for speaker_id in all_speakers:
            if speaker_id not in labels:
                labels[speaker_id] = prev.get(speaker_id, speaker_id)

    return labels


def apply_speaker_labels(
    segments: list[dict],
    labels: dict[str, str],
) -> list[dict]:
    """Replace speaker IDs with human-readable labels."""
    result = []
    for seg in segments:
        new_seg = dict(seg)
        new_seg["speaker"] = labels.get(seg["speaker"], seg["speaker"])
        result.append(new_seg)
    return result


def save_speaker_mapping(labels: dict[str, str], output_path: Path) -> None:
    """Save speaker mapping to JSON sidecar file."""
    output_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_speaker_mapping(input_path: Path) -> dict[str, str] | None:
    """Load speaker mapping from JSON sidecar file if it exists."""
    if input_path.exists():
        return json.loads(input_path.read_text(encoding="utf-8"))
    return None
