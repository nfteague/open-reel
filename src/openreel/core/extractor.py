"""Extract video clips using ffmpeg."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from openreel.config import AspectRatio, OpenReelSettings
from openreel.exceptions import ClipExtractionError
from openreel.models import DeduplicatedHighlight, ExtractedClip


def _get_crop_filter(aspect_ratio: AspectRatio) -> str | None:
    """Return the ffmpeg crop filter for the given aspect ratio, or None for 16:9 (no crop)."""
    if aspect_ratio == AspectRatio.PORTRAIT:
        # 9:16 — crop width to height*9/16, centered
        return "crop=ih*9/16:ih"
    elif aspect_ratio == AspectRatio.SQUARE:
        # 1:1 — crop to square using the shorter dimension
        return "crop=min(iw\\,ih):min(iw\\,ih)"
    return None


def _get_output_dimensions(source_width: int, source_height: int, aspect_ratio: AspectRatio) -> tuple[int, int]:
    """Calculate output dimensions after cropping."""
    if aspect_ratio == AspectRatio.PORTRAIT:
        out_w = int(source_height * 9 / 16)
        # Ensure even dimensions
        out_w = out_w - (out_w % 2)
        return out_w, source_height
    elif aspect_ratio == AspectRatio.SQUARE:
        size = min(source_width, source_height)
        size = size - (size % 2)
        return size, size
    return source_width, source_height


logger = logging.getLogger(__name__)


def _format_timestamp_filename(seconds: float) -> str:
    """Format seconds as 00h12m34s for filenames."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _sanitize_title(title: str, max_length: int = 50) -> str:
    """Sanitize a title for use in a filename."""
    sanitized = title.lower().strip()
    sanitized = re.sub(r"[^a-z0-9\s]", "", sanitized)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized[:max_length]


def _build_output_path(
    output_dir: Path,
    highlight: DeduplicatedHighlight,
) -> Path:
    """Build the output file path for a clip."""
    timestamp = _format_timestamp_filename(highlight.start_seconds)
    title = _sanitize_title(highlight.title)
    filename = f"clip_{highlight.index + 1:03d}_{timestamp}_{title}.mp4"
    return output_dir / filename


async def extract_clip(
    input_path: Path,
    highlight: DeduplicatedHighlight,
    output_dir: Path,
    settings: OpenReelSettings,
    highlight_video_width: int | None = None,
    highlight_video_height: int | None = None,
) -> ExtractedClip:
    """Extract a single clip from the source video.

    Uses stream copy by default (fast but keyframe-aligned).
    Uses re-encoding if settings.accurate_cuts is True.

    Raises:
        ClipExtractionError: If ffmpeg fails to extract the clip.
    """
    output_path = _build_output_path(output_dir, highlight)
    duration = highlight.padded_end_seconds - highlight.padded_start_seconds

    crop_filter = _get_crop_filter(settings.aspect_ratio)
    needs_reencode = settings.accurate_cuts or crop_filter is not None

    if needs_reencode:
        vf = crop_filter or ""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            str(highlight.padded_start_seconds),
            "-to",
            str(highlight.padded_end_seconds),
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    else:
        # Fast mode: seek before input (keyframe-aligned), copy streams
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(highlight.padded_start_seconds),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]

    logger.debug("Extracting clip %d: %s", highlight.index + 1, " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise ClipExtractionError(
            f"Failed to extract clip {highlight.index + 1} '{highlight.title}': {stderr.decode().strip()[-200:]}"
        )

    if not output_path.exists():
        raise ClipExtractionError(f"Clip file not created for highlight {highlight.index + 1}: {output_path}")

    actual_size = output_path.stat().st_size
    logger.info(
        "Extracted clip %d: %s (%.1fs, %.1f MB)",
        highlight.index + 1,
        output_path.name,
        duration,
        actual_size / (1024 * 1024),
    )

    # Add captions if enabled
    if settings.captions:
        from openreel.core.captioner import add_captions

        src_w = highlight_video_width or 1920
        src_h = highlight_video_height or 1080
        out_w, out_h = _get_output_dimensions(src_w, src_h, settings.aspect_ratio)

        await add_captions(
            clip_path=output_path,
            model_size=settings.caption_model_size,
            openai_api_key=settings.openai_api_key,
            diarize=settings.diarize,
            hf_token=settings.hf_token,
            video_width=out_w,
            video_height=out_h,
        )

    return ExtractedClip(
        highlight=highlight,
        output_path=output_path,
        duration_seconds=duration,
    )


async def extract_all_clips(
    input_path: Path,
    highlights: list[DeduplicatedHighlight],
    output_dir: Path,
    settings: OpenReelSettings,
    on_progress: callable | None = None,
    video_width: int | None = None,
    video_height: int | None = None,
) -> tuple[list[ExtractedClip], list[str]]:
    """Extract all clips concurrently, respecting concurrency limits.

    Returns a tuple of (successful clips, error messages for failed clips).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # If captions enabled, serialize extraction (transcription is CPU-heavy)
    concurrency = 1 if settings.captions else settings.max_extraction_concurrency
    semaphore = asyncio.Semaphore(concurrency)
    clips: list[ExtractedClip] = []
    errors: list[str] = []

    async def _extract_with_semaphore(highlight: DeduplicatedHighlight) -> None:
        async with semaphore:
            try:
                clip = await extract_clip(
                    input_path,
                    highlight,
                    output_dir,
                    settings,
                    highlight_video_width=video_width,
                    highlight_video_height=video_height,
                )
                clips.append(clip)
            except ClipExtractionError as e:
                logger.error("Clip extraction failed: %s", e)
                errors.append(str(e))
            if on_progress:
                on_progress(len(clips), len(errors))

    tasks = [_extract_with_semaphore(h) for h in highlights]
    await asyncio.gather(*tasks)

    # Sort clips by index to maintain order
    clips.sort(key=lambda c: c.highlight.index)

    return clips, errors
