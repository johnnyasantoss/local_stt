"""Speaker diarization using pyannote-audio."""

import json
import os
import shutil
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

        # Build kwargs explicitly so pyright can narrow the types (avoids int/float vs bool mismatch).
        _pipeline_kwargs: dict = {}
        if kwargs.get("num_speakers") is not None:
            _pipeline_kwargs["num_speakers"] = kwargs["num_speakers"]
        elif kwargs.get("min_speakers") is not None:
            _pipeline_kwargs["min_speakers"] = kwargs["min_speakers"]
            _pipeline_kwargs["max_speakers"] = kwargs["max_speakers"]
        result = pipeline(str(wav_path), **_pipeline_kwargs)  # type: ignore[arg-type]
        if hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization  # type: ignore[union-attr]
        else:
            annotation = result
            assert hasattr(annotation, "itertracks"), (
                "Expected pyannote Annotation or DiarizationOutput"
            )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    segments = []
    # Guard against pyannote versions that do not expose itertracks (e.g. DiarizationOutput).
    if hasattr(annotation, "itertracks"):
        it = annotation.itertracks(yield_label=True)  # type: ignore[call-non-callable]
    else:
        it = []

    for turn, _, speaker in it:
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


def _play_audio_snippet(wav_path: Path, start_s: float, duration_s: float = 5.0) -> None:
    """Play a short audio snippet from *wav_path* starting at *start_s*.

    Uses ``afplay -start`` on macOS (which accepts fractional seconds), falling
    back to ``ffplay -ss`` / ``ffplay -t`` on other platforms. Silently skips if
    no player is available — the user can still type the speaker ID without
    hearing a sample.
    """
    duration_s = max(2.0, min(duration_s, 8.0))
    try:
        if Path("/usr/bin/afplay").is_file():
            subprocess.run(
                ["afplay", "-start", str(start_s), "-length", str(duration_s), str(wav_path)],
                check=True,
                timeout=duration_s + 2,
            )
        elif shutil.which("ffplay") is not None:
            subprocess.run(
                [
                    "ffplay",
                    "-nodisp",
                    "-loglevel",
                    "quiet",
                    "-ss",
                    str(start_s),
                    "-t",
                    str(duration_s),
                    str(wav_path),
                ],
                check=True,
                timeout=duration_s + 2,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _speaker_diarization_segments(
    speaker_id: str,
    diarization_segments: list[dict],
    limit: int = 15,
    min_duration_s: float = 0.5,
) -> list[dict]:
    """Return up to *limit* contiguous diarization segments for one speaker with duration >= *min_duration_s*."""
    out: list[dict] = []
    for s in diarization_segments:
        if s["speaker"] != speaker_id:
            continue
        dur = s["end"] - s["start"]
        if dur < min_duration_s:
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def prompt_speaker_labels(
    samples_by_transcription: dict[str, list[dict]],
    diarization_segments: list[dict],
    wav_path: Path | None = None,
    existing_labels: dict[str, str] | None = None,
) -> tuple[dict[str, str], bool]:
    """Interactive CLI prompt for labeling speakers.

    Shows every pyannote-detected speaker — including those whose time ranges
    did not overlap any transcription segment (e.g. when transcribe-cli emits
    one long segment). For each speaker: display up to three short audio snippets
    from their diarization segments and accept a name label, with an optional
    "h" key to replay the last sample.

    If existing_labels is provided, shows the saved label as default
    (press Enter to keep it). Re-asks all speakers each run so the user
    can correct previous labels.

    Returns (labels, finalized). finalized is True only if the user
    labeled all speakers without interruption.
    """
    import re

    labels: dict[str, str] = {}
    finalized = True
    prev = existing_labels or {}

    # Merge speaker IDs from transcription samples and diarization segments so no
    # pyannote-detected voice is silently dropped when transcribe-cli outputs few
    # segments.
    all_speakers: set[str] = set(samples_by_transcription.keys())
    for dseg in diarization_segments:
        all_speakers.add(dseg["speaker"])
    sorted_ids = sorted(all_speakers)
    if not sorted_ids:
        return labels, finalized

    print(f"\n  {len(sorted_ids)} speaker(s) found.\n")
    print("  For each speaker, listen to the audio snippets below and enter a name.")
    print("  Press 'h' + Enter to replay the last snippet, or Enter alone to skip listening.\n")
    print("  Ctrl+C to save partial labels and exit.\n")

    try:
        for speaker_id in sorted_ids:
            default = prev.get(speaker_id, "")
            count = sum(1 for s in diarization_segments if s["speaker"] == speaker_id)
            print(f"  {speaker_id} ({count} diarization segments)")
            # Show transcription samples when available (with text preview).
            trans_segs: list[dict] = samples_by_transcription.get(speaker_id, [])
            if trans_segs:
                for seg in trans_segs[:3]:
                    ts = _format_timestamp(seg["start"])
                    text = re.sub(r"^\[.+?\]:\s*", "", seg.get("text", ""))
                    preview = (text or "<silence>")[:80] + ("..." if len(text) > 80 else "")
                    print(f'    [{ts}] "{preview}"')

            # Show diarization audio snippets for speakers without transcription overlap.
            if not trans_segs:
                dsegs = _speaker_diarization_segments(speaker_id, diarization_segments)
                for seg in dsegs[:3]:
                    dur = max(2.0, min(seg["end"] - seg["start"], 5.0))
                    print(f"    [{ts}] (~{dur:.1f}s) — press h+Enter to hear")
                    if wav_path is not None and wav_path.is_file():
                        try:
                            _play_audio_snippet(wav_path, seg["start"], dur)
                        except (OSError, subprocess.SubprocessError):
                            pass

            prompt = f"  Enter name for {speaker_id} [{default or 'Speaker'}]: "
            last_played: float | None = None
            while True:
                try:
                    reply = input(prompt).strip()
                except EOFError:
                    return labels, False
                if reply == "" and default:
                    prompt = f"  Enter name for {speaker_id} [{default}] (replay with h+Enter): "
                    continue
                if reply == "h":
                    # Replay last played snippet or pick the first diarization segment.
                    target = None
                    if last_played is not None:
                        target = last_played
                    else:
                        for d in diarization_segments:
                            if d["speaker"] == speaker_id and (d["end"] - d["start"]) >= 1.0:
                                target = d["start"]
                                break
                    if wav_path is not None and target is not None and wav_path.is_file():
                        try:
                            _play_audio_snippet(wav_path, target)
                            last_played = target
                        except (OSError, subprocess.SubprocessError):
                            pass
                    prompt = f"  Enter name for {speaker_id} [{default or 'Speaker'}] (replay with h+Enter): "
                    continue
                if reply:
                    labels[speaker_id] = reply
                    break
                # Empty reply with no default — ask again.
                prompt = f"  Enter name for {speaker_id}: "

            print()
    except KeyboardInterrupt:
        finalized = False
        print("\n  Interrupted. Saving partial labels...")
        for speaker_id in sorted_ids:
            if speaker_id not in labels:
                labels[speaker_id] = prev.get(speaker_id, speaker_id)

    return labels, finalized


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


def save_speaker_mapping(
    labels: dict[str, str],
    output_path: Path,
    finalized: bool = False,
) -> None:
    """Save speaker mapping to JSON sidecar file.

    Args:
        labels: Mapping of speaker IDs to human-readable names.
        output_path: Path to write the JSON file.
        finalized: True if the user completed labeling (not interrupted).
    """
    data = {"finalized": finalized, "labels": labels}
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_speaker_mapping(
    input_path: Path,
) -> tuple[dict[str, str], bool] | None:
    """Load speaker mapping from JSON sidecar file if it exists.

    Returns (labels, finalized) or None if file doesn't exist.
    Handles both old flat format {speaker: label} and new format
    {"finalized": bool, "labels": {speaker: label}}.
    """
    if not input_path.exists():
        return None
    data = json.loads(input_path.read_text(encoding="utf-8"))
    # New format: {"finalized": bool, "labels": {...}}
    if isinstance(data, dict) and "labels" in data:
        return data["labels"], data.get("finalized", False)
    # Old flat format: {"SPEAKER_00": "Alice", ...} (backward compat)
    return data, False
