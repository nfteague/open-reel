"""Gemini video analysis: upload, prompt, and parse structured output."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from openreel.config import PRESET_CRITERIA_TEXT, OpenReelSettings, PresetCriteria
from openreel.exceptions import AnalysisError, FileUploadError
from openreel.models import ChunkAnalysisResult, ChunkWindow, HighlightMoment

logger = logging.getLogger(__name__)

# Maximum file size for Gemini File API upload (2 GB)
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_prompt(
    chunk: ChunkWindow,
    total_chunks: int,
    settings: OpenReelSettings,
    criteria: str | None = None,
    preset: PresetCriteria | None = None,
) -> str:
    """Build the analysis prompt for a single chunk."""
    # Resolve criteria text
    if criteria:
        criteria_section = f"Highlight criteria:\n{criteria}"
    elif preset:
        criteria_section = f"Highlight criteria:\n{PRESET_CRITERIA_TEXT[preset]}"
    else:
        criteria_section = f"Highlight criteria:\n{PRESET_CRITERIA_TEXT[PresetCriteria.GENERAL]}"

    chunk_duration_hours = chunk.duration_seconds / 3600
    count_by_rate = round(settings.target_clips_per_hour * chunk_duration_hours)
    # Spread the minimum total evenly across chunks, rounding up
    count_by_minimum = -(-settings.min_clips_total // total_chunks)  # ceiling division
    target_count = max(count_by_minimum, count_by_rate)

    return f"""You are analyzing a segment of a longer video recording (segment {chunk.index + 1} of {total_chunks}).
This segment covers timestamps {_format_timestamp(chunk.start_seconds)} to {_format_timestamp(chunk.end_seconds)} of the full video.

Your task: Identify the most highlight-worthy moments in this segment.

{criteria_section}

Guidelines:
- Find approximately {target_count} highlights in this segment (roughly {settings.target_clips_per_hour} per hour).
- Each highlight must be at least {settings.min_clip_seconds} seconds long but NO LONGER than 120 seconds. Most clips should be 15-60 seconds. Only exceed 60 seconds if the moment truly justifies it (e.g., an extended clutch play or back-and-forth fight).
- When identifying a moment, pinpoint the tightest possible window around the most exciting part. For example, a winning fight should be the final 20-40 seconds of action, not the entire match.
- Return timestamps as absolute positions from the start of this video segment.
- For each highlight, provide: a short title, a description of why it's noteworthy, start and end timestamps in seconds, a confidence score from 0.0 to 1.0, and categorization tags.
- If a noteworthy moment occurs near the segment boundaries (within {settings.overlap_seconds} seconds of the start or end), still include it — duplicates will be handled later.
- Prefer quality over quantity. Only flag genuinely compelling moments.
- If there are no highlight-worthy moments, return an empty moments list."""


async def _split_chunk_file(
    input_path: Path,
    chunk: ChunkWindow,
    tmp_dir: Path,
    downscale: bool = False,
    analysis_fps: int = 0,
) -> Path:
    """Extract a chunk from the video for Gemini analysis.

    Modes (applied in combination):
    - analysis_fps > 0: Drop to N fps (e.g., 2). Dramatically reduces file size
      with minimal CPU load since the encoder outputs very few frames. Gemini
      samples at ~1fps so low FPS is fine for highlight detection.
    - downscale=True: Also resize to 720p (more CPU-intensive).
    - Neither: Stream copy (no re-encoding, largest files).
    """
    output_path = tmp_dir / f"chunk_{chunk.index:03d}.mp4"

    needs_reencode = downscale or analysis_fps > 0

    if needs_reencode:
        vf_filters = []
        if analysis_fps > 0:
            vf_filters.append(f"fps={analysis_fps}")
        if downscale:
            vf_filters.append("scale=-2:720")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(chunk.start_seconds),
            "-i",
            str(input_path),
            "-t",
            str(chunk.duration_seconds),
            "-threads",
            "4",
            "-vf",
            ",".join(vf_filters),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(chunk.start_seconds),
            "-i",
            str(input_path),
            "-t",
            str(chunk.duration_seconds),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise FileUploadError(f"Failed to split chunk {chunk.index}: {stderr.decode().strip()}")

    return output_path


def _upload_file_sync(client: genai.Client, file_path: Path) -> types.File:
    """Upload a file to Gemini File API (synchronous)."""
    logger.info("Uploading %s (%.1f MB)...", file_path.name, file_path.stat().st_size / (1024 * 1024))

    uploaded = client.files.upload(file=file_path)

    # Poll until the file is ACTIVE
    import time

    max_wait = 600  # 10 minutes
    poll_interval = 5
    waited = 0

    while uploaded.state == "PROCESSING":
        if waited >= max_wait:
            raise FileUploadError(
                f"File {file_path.name} still processing after {max_wait}s. "
                "The file may be too large or Gemini is overloaded."
            )
        time.sleep(poll_interval)
        waited += poll_interval
        uploaded = client.files.get(name=uploaded.name)
        poll_interval = min(poll_interval * 1.5, 15)

    if uploaded.state == "FAILED":
        raise FileUploadError(f"File upload failed for {file_path.name}")

    logger.info("Upload complete: %s -> %s", file_path.name, uploaded.name)
    return uploaded


@retry(
    retry=retry_if_exception_type((AnalysisError,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "Retrying analysis (attempt %d): %s",
        retry_state.attempt_number + 1,
        retry_state.outcome.exception() if retry_state.outcome else "unknown",
    ),
)
def _generate_content_sync(
    client: genai.Client,
    model: str,
    uploaded_file: types.File,
    prompt: str,
) -> ChunkAnalysisResult:
    """Call Gemini generate_content with structured output (synchronous)."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ChunkAnalysisResult,
                temperature=0.2,
            ),
        )
    except Exception as e:
        logger.error("Gemini API call failed: %s", e)
        error_str = str(e).lower()
        if "429" in error_str or "rate" in error_str:
            raise AnalysisError(f"Rate limited by Gemini API: {e}") from e
        if "503" in error_str or "overloaded" in error_str:
            raise AnalysisError(f"Gemini API overloaded: {e}") from e
        raise AnalysisError(f"Gemini API error: {e}") from e

    if not response.text:
        raise AnalysisError("Gemini returned empty response")

    try:
        result = ChunkAnalysisResult.model_validate_json(response.text)
    except Exception as e:
        raise AnalysisError(f"Failed to parse Gemini response: {e}") from e

    return result


async def split_chunk(
    input_path: Path,
    chunk: ChunkWindow,
    tmp_dir: Path,
    downscale: bool = False,
    analysis_fps: int = 0,
) -> Path:
    """Split a single chunk from the source video. Run sequentially to avoid CPU overload."""
    tags = []
    if analysis_fps > 0:
        tags.append(f"{analysis_fps}fps")
    if downscale:
        tags.append("720p")
    tag_str = f" [{', '.join(tags)}]" if tags else ""

    logger.info(
        "Splitting chunk %d (%s - %s)%s...",
        chunk.index + 1,
        _format_timestamp(chunk.start_seconds),
        _format_timestamp(chunk.end_seconds),
        tag_str,
    )
    return await _split_chunk_file(input_path, chunk, tmp_dir, downscale=downscale, analysis_fps=analysis_fps)


async def analyze_chunk(
    chunk: ChunkWindow,
    chunk_file: Path,
    total_chunks: int,
    settings: OpenReelSettings,
    criteria: str | None = None,
    preset: PresetCriteria | None = None,
    needs_offset: bool = False,
) -> list[HighlightMoment]:
    """Analyze a pre-split video chunk with Gemini and return highlight moments.

    Timestamps in returned moments are adjusted to be absolute positions
    within the full source video when needs_offset is True.
    """
    api_key = settings.resolve_api_key()
    client = genai.Client(api_key=api_key)

    # Upload to Gemini (run sync upload in thread pool)
    loop = asyncio.get_event_loop()
    uploaded_file = await loop.run_in_executor(None, _upload_file_sync, client, chunk_file)

    try:
        # Build prompt
        prompt = _build_prompt(chunk, total_chunks, settings, criteria, preset)

        logger.info(
            "Analyzing chunk %d/%d (%s - %s)...",
            chunk.index + 1,
            total_chunks,
            _format_timestamp(chunk.start_seconds),
            _format_timestamp(chunk.end_seconds),
        )

        # Generate content (run sync call in thread pool)
        result = await loop.run_in_executor(
            None, _generate_content_sync, client, settings.gemini_model.value, uploaded_file, prompt
        )

        logger.info(
            "Chunk %d/%d: found %d moments. Summary: %s",
            chunk.index + 1,
            total_chunks,
            len(result.moments),
            result.chunk_summary[:100],
        )

        # Adjust timestamps: if we split the file, the model sees timestamps
        # relative to the chunk start. Offset them to absolute positions.
        moments = []
        for moment in result.moments:
            adjusted = moment.model_copy(
                update={
                    "start_seconds": moment.start_seconds + chunk.start_seconds
                    if needs_offset
                    else moment.start_seconds,
                    "end_seconds": moment.end_seconds + chunk.start_seconds if needs_offset else moment.end_seconds,
                }
            )
            moments.append(adjusted)

        return moments

    finally:
        # Clean up uploaded file from Gemini
        try:
            client.files.delete(name=uploaded_file.name)
            logger.debug("Cleaned up Gemini file: %s", uploaded_file.name)
        except Exception:
            logger.warning("Failed to clean up Gemini file: %s", uploaded_file.name)
