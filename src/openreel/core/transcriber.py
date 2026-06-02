"""Transcription backends and speaker diarization."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openreel.exceptions import OpenReelError

logger = logging.getLogger(__name__)


class TranscriptionError(OpenReelError):
    """Transcription failed."""


@dataclass
class TimedWord:
    """A single word with precise timestamps and optional speaker label."""

    word: str
    start: float  # seconds
    end: float  # seconds
    speaker: str | None = None  # e.g., "SPEAKER_00", "SPEAKER_01"


async def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    """Extract audio from a video file as 16kHz mono WAV for transcription."""
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav", prefix="openreel_audio_"))

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise TranscriptionError(f"Audio extraction failed: {stderr.decode().strip()[-200:]}")

    logger.debug(
        "Extracted audio: %s (%.1f MB)",
        output_path.name,
        output_path.stat().st_size / (1024 * 1024),
    )
    return output_path


async def _diarize(audio_path: Path, hf_token: str = "") -> list[tuple[float, float, str]]:
    """Run speaker diarization with pyannote.audio.

    Returns list of (start, end, speaker_label) segments.
    Requires a HuggingFace token for model access.
    """
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        raise TranscriptionError("pyannote.audio is not installed. Run: pip install pyannote.audio")

    logger.info("Running speaker diarization...")
    loop = asyncio.get_event_loop()

    def _run():
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **kwargs,
        )
        result = pipeline(str(audio_path))

        # pyannote 4.x returns DiarizeOutput; access the Annotation via .speaker_diarization
        annotation = getattr(result, "speaker_diarization", result)

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))

        return segments

    segments = await loop.run_in_executor(None, _run)
    speakers = set(s[2] for s in segments)
    logger.info("Diarization complete: %d segments, %d speakers", len(segments), len(speakers))
    return segments


def _assign_speakers(
    words: list[TimedWord],
    diarization: list[tuple[float, float, str]],
) -> list[TimedWord]:
    """Assign speaker labels to words by matching timestamps to diarization segments."""
    for word in words:
        word_mid = (word.start + word.end) / 2
        best_speaker = None
        best_overlap = 0.0

        for seg_start, seg_end, speaker in diarization:
            # Calculate overlap between word and diarization segment
            overlap_start = max(word.start, seg_start)
            overlap_end = min(word.end, seg_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

            # Also check if word midpoint falls within segment
            if seg_start <= word_mid <= seg_end and best_speaker is None:
                best_speaker = speaker

        word.speaker = best_speaker

    return words


async def transcribe_local(
    audio_path: Path,
    model_size: str = "base",
) -> list[TimedWord]:
    """Transcribe audio using faster-whisper (local, no API key needed)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscriptionError("faster-whisper is not installed. Run: pip install openreel[captions]")

    logger.info("Transcribing with faster-whisper (model: %s)...", model_size)

    loop = asyncio.get_event_loop()

    def _transcribe():
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
        )

        words: list[TimedWord] = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words.append(
                        TimedWord(
                            word=w.word.strip(),
                            start=w.start,
                            end=w.end,
                        )
                    )

        return words

    words = await loop.run_in_executor(None, _transcribe)
    logger.info("Transcribed %d words", len(words))
    return words


async def transcribe_openai(
    audio_path: Path,
    api_key: str,
) -> list[TimedWord]:
    """Transcribe audio using OpenAI Whisper API (cloud, requires key)."""
    import httpx

    logger.info("Transcribing with OpenAI Whisper API...")

    async with httpx.AsyncClient(timeout=300) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (audio_path.name, f, "audio/wav")},
                data={
                    "model": "whisper-1",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
            )

    if response.status_code != 200:
        raise TranscriptionError(f"OpenAI API error ({response.status_code}): {response.text[:200]}")

    data = response.json()
    words: list[TimedWord] = []

    for w in data.get("words", []):
        words.append(
            TimedWord(
                word=w["word"].strip(),
                start=w["start"],
                end=w["end"],
            )
        )

    logger.info("Transcribed %d words via OpenAI", len(words))
    return words


def _load_cached_transcript(cache_path: Path | None) -> list[TimedWord] | None:
    """Load cached transcript from disk if available."""
    if not cache_path or not cache_path.exists():
        return None
    try:
        import json

        data = json.loads(cache_path.read_text())
        return [TimedWord(**w) for w in data.get("words", [])]
    except Exception as e:
        logger.warning("Failed to load cached transcript: %s", e)
        return None


def _save_cached_transcript(words: list[TimedWord], cache_path: Path) -> None:
    """Save transcript to cache."""
    try:
        import json
        from dataclasses import asdict

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"words": [asdict(w) for w in words]}
        cache_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning("Failed to cache transcript: %s", e)


async def transcribe(
    video_path: Path,
    model_size: str = "base",
    openai_api_key: str = "",
    diarize: bool = False,
    hf_token: str = "",
    cache_path: Path | None = None,
) -> list[TimedWord]:
    """Transcribe a video file, optionally with speaker diarization.

    If cache_path is provided and the cache exists with matching diarization
    state, the cached transcript is returned instead of re-transcribing.
    """
    # Check cache — match diarization state
    cached = _load_cached_transcript(cache_path)
    if cached is not None:
        has_speakers = any(w.speaker for w in cached)
        if has_speakers == diarize:
            logger.info("Using cached transcript (%d words)", len(cached))
            return cached

    audio_path = await extract_audio(video_path)

    try:
        if openai_api_key:
            words = await transcribe_openai(audio_path, openai_api_key)
        else:
            words = await transcribe_local(audio_path, model_size)

        if diarize and words:
            try:
                segments = await _diarize(audio_path, hf_token=hf_token)
                words = _assign_speakers(words, segments)
            except Exception as e:
                logger.warning("Diarization failed, continuing without speaker labels: %s", e)

        if cache_path:
            _save_cached_transcript(words, cache_path)

        return words
    finally:
        audio_path.unlink(missing_ok=True)


def summarize_speakers(words: list[TimedWord]) -> list[dict]:
    """Build a summary of speakers from a transcribed word list."""
    speakers: dict[str, dict] = {}
    for w in words:
        if not w.speaker:
            continue
        if w.speaker not in speakers:
            speakers[w.speaker] = {
                "speaker_id": w.speaker,
                "sample_words": [],
                "word_count": 0,
                "total_duration_seconds": 0.0,
            }
        s = speakers[w.speaker]
        s["word_count"] += 1
        s["total_duration_seconds"] += w.end - w.start
        if len(s["sample_words"]) < 8:
            s["sample_words"].append(w.word)

    return sorted(speakers.values(), key=lambda x: -x["word_count"])
